module Fitter

using ..Models: ModelParameters, DataParameters, ClusterParameters, GaussianParameters, NtlParameters
using ..Models: DataSufficientStatistics, ClusterSufficientStatistics, GaussianSufficientStatistics
using ..Models: NtlSufficientStatistics
using Distributions
using LinearAlgebra
using SpecialFunctions
using Statistics

function prepare_data_sufficient_statistics(data::Matrix{Float64}, data_parameters::GaussianParameters)
    dim = size(data)[1]
    n = size(data)[2]
    data_means = zeros(Float64, dim, n)
    data_means[:, 1] = Statistics.mean(data, dims=2)
    posterior_means = Matrix{Float64}(undef, dim, n)
    for i = 1:n
        posterior_means[:, i] = data_parameters.prior_mean
    end
    posterior_covs = Matrix{Float64}(undef, dim*dim, n)
    for i = 1:n
        posterior_covs[:, i] = vec(data_parameters.prior_covariance)
    end
    return GaussianSufficientStatistics(data_means, posterior_means, posterior_covs)
end

function prepare_cluster_sufficient_statistics(data::Matrix{Float64}, cluster_parameters::ClusterParameters)
    n = size(data)[2]
    cluster_num_observations = Vector{Int64}(zeros(Int64, n))
    cluster_num_observations[1] = n
    cluster_sufficient_stats = NtlSufficientStatistics(cluster_num_observations, 1)
    return cluster_sufficient_stats
end

function fit(data::Matrix, data_parameters::DataParameters, cluster_parameters::ClusterParameters, 
             num_instances::Int64, method::String)
    n = size(data)[2]
    dim = size(data)[1]
    instances = Array{Int64}(undef, n, num_instances)
    instances[:, 1] .= 1
    cluster_sufficient_stats = prepare_cluster_sufficient_statistics(data, cluster_parameters)
    data_sufficient_stats = prepare_data_sufficient_statistics(data, data_parameters)
    compute_data_posterior_parameters!(1, cluster_sufficient_stats, data_sufficient_stats, data_parameters)
    if method === "gibbs"
        gibbs_sample!(instances, data, cluster_sufficient_stats, data_sufficient_stats, data_parameters, 
                      cluster_parameters)
    else
        message = "$method is not an appropriate fit method."
        throw(ArgumentError(message))
    end
    return instances
end

function gibbs_sample!(instances::Matrix{Int64}, data::Matrix{Float64}, 
                       cluster_sufficient_stats::ClusterSufficientStatistics,
                       data_sufficient_stats::DataSufficientStatistics, 
                       data_parameters::DataParameters,
                       cluster_parameters::ClusterParameters)
    iterations = size(instances)[2]
    for iteration = 2:iterations
        println("Iteration: $iteration/$iterations")
        instances[:, iteration] = instances[:, iteration-1]
        for observation = 1:size(instances)[1]
            remove_observation!(observation, instances, iteration, data, cluster_sufficient_stats, 
                                data_sufficient_stats, data_parameters, cluster_parameters)
            assignment = vec(instances[:, iteration])
            (cluster, weight) = gibbs_move(observation, data, assignment, cluster_sufficient_stats, 
                                           data_sufficient_stats, data_parameters, cluster_parameters)
            add_observation!(observation, cluster, instances, iteration, data, cluster_sufficient_stats, 
                             data_sufficient_stats, data_parameters, cluster_parameters)
        end
    end
end

function gumbel_max(objects::Vector{Int64}, weights::Vector{Float64})
    gumbel_values = rand(Gumbel(0, 1), length(weights))
    index = argmax(gumbel_values + weights)
    return (objects[index], weights[index])
end

function update_data_sufficient_statistics!(data_sufficient_stats::GaussianSufficientStatistics, 
                                            cluster::Int64, datum::Vector{Float64}, 
                                            cluster_sufficient_stats::ClusterSufficientStatistics, 
                                            update_type::String)
    cluster_mean = vec(data_sufficient_stats.data_means[:, cluster])
    n = cluster_sufficient_stats.num_observations[cluster]
    if update_type == "add"
        cluster_mean = (cluster_mean.*(n-1) + datum)./n
    elseif update_type == "remove"
        if n > 0
            cluster_mean = (cluster_mean.*(n+1) - datum)./n
        else
            cluster_mean .= 0
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
    data_sufficient_stats.data_means[:, cluster] = cluster_mean
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::ClusterSufficientStatistics, 
                                              new_time::Int64, old_time::Int64)
    cluster_sufficient_stats.num_observations[new_time] = cluster_sufficient_stats.num_observations[old_time]
    cluster_sufficient_stats.num_observations[old_time] = 0
end

function update_data_stats_new_birth_time!(data_sufficient_stats::GaussianSufficientStatistics, 
                                           new_time::Int64, old_time::Int64, 
                                           data_parameters::GaussianParameters)
    data_sufficient_stats.data_means[:, new_time] = data_sufficient_stats.data_means[:, old_time]
    data_sufficient_stats.data_means[:, old_time] .= 0
    data_sufficient_stats.posterior_means[:, new_time] = data_sufficient_stats.posterior_means[:, old_time]
    data_sufficient_stats.posterior_covs[:, new_time] = data_sufficient_stats.posterior_covs[:, old_time]
    data_sufficient_stats.posterior_means[:, old_time] = data_parameters.prior_mean
    data_sufficient_stats.posterior_covs[:, old_time] = vec(data_parameters.prior_covariance)
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, 
                             data::Matrix{Float64}, cluster_sufficient_stats::ClusterSufficientStatistics, 
                             data_sufficient_stats::DataSufficientStatistics,
                             data_parameters::DataParameters, cluster_parameters::ClusterParameters)
    cluster = instances[observation, iteration]
    instances[observation, iteration] = size(instances)[1] + 1 
    datum = vec(data[:, observation])
    cluster_sufficient_stats.num_observations[cluster] -= 1
    if cluster_sufficient_stats.num_observations[cluster] === 0
        cluster_sufficient_stats.num_clusters -= 1
    end
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, "remove")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, data_parameters)
    if (cluster === observation) && (cluster_sufficient_stats.num_observations[cluster] > 0)
        dim = length(datum)
        assigned_to_cluster = instances[:, iteration] .=== cluster
        new_cluster_time = findfirst(assigned_to_cluster)
        instances[assigned_to_cluster, iteration] .= new_cluster_time 
        update_cluster_stats_new_birth_time!(cluster_sufficient_stats, new_cluster_time, cluster)
        update_data_stats_new_birth_time!(data_sufficient_stats, new_cluster_time, cluster, data_parameters)
    end
end

function add_observation!(observation::Int64, cluster::Int64, instances::Matrix{Int64}, iteration::Int64, 
                          data::Matrix{Float64}, cluster_sufficient_stats::ClusterSufficientStatistics,
                          data_sufficient_stats::DataSufficientStatistics, 
                          data_parameters::DataParameters, cluster_parameters::ClusterParameters)
    if observation < cluster
        dim = size(data)[1]
        assigned_to_cluster = (instances[:, iteration] .=== cluster)
        old_cluster_time = cluster
        cluster = observation
        instances[assigned_to_cluster, iteration] .= cluster
        update_cluster_stats_new_birth_time!(cluster_sufficient_stats, cluster, old_cluster_time)
        update_data_stats_new_birth_time!(data_sufficient_stats, cluster, old_cluster_time, data_parameters)
    end
    instances[observation, iteration] = cluster
    datum = vec(data[:, observation])
    cluster_sufficient_stats.num_observations[cluster] += 1
    if cluster_sufficient_stats.num_observations[cluster] === 1
        cluster_sufficient_stats.num_clusters += 1
    end
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, "add")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, data_parameters)
end

function new_cluster_log_predictive(cluster_sufficient_stats::NtlSufficientStatistics, 
                                    cluster_parameters::ClusterParameters)
    n = length(cluster_sufficient_stats.num_observations)
    num_clusters = cluster_sufficient_stats.num_clusters
    phi_prior_a = cluster_parameters.arrival_distribution.prior[1]
    phi_prior_b = cluster_parameters.arrival_distribution.prior[2]
    return log(num_clusters - 1 + phi_prior_a) - log(n - 1 + phi_prior_a + phi_prior_b)
end

function existing_cluster_log_predictive(cluster_sufficient_stats::NtlSufficientStatistics, 
                                         cluster_parameters::ClusterParameters)
    n = length(cluster_sufficient_stats.num_observations)
    num_clusters = cluster_sufficient_stats.num_clusters
    phi_prior_a = cluster_parameters.arrival_distribution.prior[1]
    phi_prior_b = cluster_parameters.arrival_distribution.prior[2]
    return log(n - num_clusters + phi_prior_a) - log(n - 1 + phi_prior_a + phi_prior_b)
end

function compute_num_observations(cluster::Int64, assignment::Vector{Int64})
    assigned_to_cluster = assignment .=== cluster
    return length(assignment[assigned_to_cluster])
end

function compute_num_complement(cluster::Int64, cluster_num_observations::Vector{Int64}, observation::Int64)
    younger_clusters = 1:(cluster-1)
    num_complement = sum(cluster_num_observations[younger_clusters]) - (cluster - 1)
    if observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_psi_posterior(cluster::Int64, ntl_sufficient_stats::NtlSufficientStatistics, observation::Int64, 
                               ntl_parameters::NtlParameters)
    psi_posterior = Array{Int64}(undef, 2)
    psi_prior_a = ntl_parameters.prior[1]
    psi_prior_b = ntl_parameters.prior[2] 
    psi_posterior[1] = ntl_sufficient_stats.num_observations[cluster] - 1 + psi_prior_a
    psi_posterior[2] = compute_num_complement(cluster, ntl_sufficient_stats.num_observations, observation) + psi_prior_b
    return psi_posterior
end

function compute_cluster_log_predictives!(log_weights::Vector{Float64}, observation::Int64, clusters::Vector{Int64}, 
                                          ntl_sufficient_stats::NtlSufficientStatistics, ntl_parameters::NtlParameters)
    # Not strictly the number of observations
    n = maximum(clusters)
    n = maximum([n, observation])
    cluster_log_weights = zeros(Float64, n)
    complement_log_weights = zeros(Float64, n)
    psi_parameters = Array{Int64}(undef, 2, n)
    logbetas = Array{Float64}(undef, n)
    new_num_complement = compute_num_complement(observation, ntl_sufficient_stats.num_observations, observation)

    younger_cluster_new_psi = Array{Float64}(undef, 2)
    cluster_new_psi = Array{Float64}(undef, 2)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_psi_posterior(cluster, ntl_sufficient_stats, observation, ntl_parameters)
            psi_parameters[:, cluster] = cluster_psi
            denom = cluster_psi[1] + cluster_psi[2]
            log_denom = log(denom)
            cluster_log_weights[cluster] = log(cluster_psi[1]) - log_denom
            complement_log_weights[cluster] = log(cluster_psi[2]) - log_denom
            if observation < cluster
                logbetas[cluster] = logbeta(cluster_psi[1], cluster_psi[2])
            end
        end
    end

    for (i, cluster) = enumerate(clusters)
        if cluster < observation
            log_weight = 0
            if cluster > 1
                log_weight += cluster_log_weights[cluster]
            end
            # Clusters younger than the current cluster
            younger_clusters = (cluster+1):(observation-1)
            log_weight += sum(complement_log_weights[younger_clusters])
            log_weights[i] = log_weight
        else # observation < cluster
            cluster_num_obs = ntl_sufficient_stats.num_observations[cluster]
            cluster_old_psi = psi_parameters[:, cluster]

            # Clusters younger than the observation
            younger_clusters = (observation+1):(cluster-1)
            younger_clusters = younger_clusters[ntl_sufficient_stats.num_observations[younger_clusters] .> 0]

            if ntl_sufficient_stats.num_observations[1] === 0
                youngest_cluster = 2
            else
                youngest_cluster = 1
            end

            younger_clusters_log_weight = 0
            for younger_cluster = younger_clusters
                younger_cluster_old_psi = psi_parameters[:, younger_cluster]
                younger_cluster_new_psi[1] = younger_cluster_old_psi[1]
                younger_cluster_new_psi[2] = younger_cluster_old_psi[2]
                if observation === 1 && younger_cluster === youngest_cluster
                    younger_cluster_new_psi[1] = cluster_num_obs + 1 
                    younger_cluster_old_psi = cluster_old_psi
                else
                    younger_cluster_new_psi[1] += cluster_num_obs
                end
                new_a = younger_cluster_new_psi[1]
                new_b = younger_cluster_new_psi[2]
                younger_clusters_log_weight += logbeta(new_a, new_b) - logbetas[younger_cluster]
            end

            if observation > youngest_cluster
                cluster_new_psi[1] = cluster_old_psi[1] + 1
                cluster_new_psi[2] = new_num_complement + 1
                new = cluster_new_psi
                cluster_log_weight = logbeta(new[1], new[2]) - logbetas[cluster]
                log_weight = cluster_log_weight + younger_clusters_log_weight
            else
                log_weight = younger_clusters_log_weight
            end
            log_weights[i] = log_weight
        end
    end
end

function compute_existing_cluster_log_predictives(observation::Int64, assignment::Vector{Int64}, 
                                                  cluster_sufficient_stats::ClusterSufficientStatistics, 
                                                  cluster_parameters::ClusterParameters)
    clusters = get_clusters(assignment)
    num_clusters = length(clusters)
    log_weights = Array{Float64}(undef, length(clusters))
    compute_cluster_log_predictives!(log_weights, observation, clusters, cluster_sufficient_stats, cluster_parameters)
    log_weights .+= existing_cluster_log_predictive(cluster_sufficient_stats, cluster_parameters)
    return log_weights
end

function compute_data_posterior_parameters!(cluster::Int64, ntl_sufficient_stats::NtlSufficientStatistics,
                                            data_sufficient_stats::GaussianSufficientStatistics,
                                            data_parameters::GaussianParameters)
    data_mean = vec(data_sufficient_stats.data_means[:, cluster])
    n = ntl_sufficient_stats.num_observations[cluster]
    dim = length(data_mean)
    prior_mean = data_parameters.prior_mean
    prior_cov = data_parameters.prior_covariance
    prior_precision = inv(prior_cov)
    if n > 0
        data_precision = inv(data_parameters.data_covariance)
        cov = inv(prior_precision + n*data_precision)
        posterior_mean = cov*(prior_cov*prior_mean + n*data_precision*data_mean)
    else
        posterior_mean = prior_mean
        cov = prior_cov
    end
    data_sufficient_stats.posterior_means[:, cluster] = vec(posterior_mean)
    data_sufficient_stats.posterior_covs[:, cluster] = vec(cov)
end

function data_log_predictive(datum::Vector{Float64}, cluster::Int64, 
                             data_sufficient_stats::GaussianSufficientStatistics, 
                             data_parameters::GaussianParameters)
    posterior_mean = vec(data_sufficient_stats.posterior_means[:, cluster])
    dim = length(posterior_mean)
    posterior_cov = reshape(data_sufficient_stats.posterior_covs[:, cluster], dim, dim)
    posterior = MvNormal(posterior_mean, data_parameters.data_covariance + posterior_cov)
    return logpdf(posterior, datum)
end

function compute_data_log_predictives(observation::Int64, clusters::Vector{Int64}, data::Matrix{Float64}, 
                                      data_sufficient_stats::DataSufficientStatistics, 
                                      data_parameters::DataParameters)
    num_clusters = length(clusters)
    log_predictive = Array{Float64}(undef, num_clusters)
    datum = vec(data[:, observation])
    dim = length(datum)
    for i = 1:num_clusters
        cluster = clusters[i]
        log_predictive[i] = data_log_predictive(datum, cluster, data_sufficient_stats, data_parameters)
    end
    return log_predictive
end

function get_clusters(assignment::Vector{Int64})
    # We represent the observation not assigned to a cluster, if it exists, by n+1
    assigned = assignment .<= length(assignment)
    return unique(sort(assignment[assigned]))
end

function gibbs_move(observation::Int64, data::Matrix{Float64}, assignment::Vector{Int64},
                    cluster_sufficient_stats::ClusterSufficientStatistics, 
                    data_sufficient_stats::DataSufficientStatistics,
                    data_parameters::DataParameters, cluster_parameters::ClusterParameters)
    clusters = get_clusters(assignment)
    n = size(data)[2]
    num_clusters = length(clusters)

    choices = Array{Int64}(undef, num_clusters+1)
    choices[1:num_clusters] = clusters
    choices[num_clusters+1] = observation

    weights = Array{Float64}(undef, num_clusters+1)
    weights[1:num_clusters] = compute_existing_cluster_log_predictives(observation, assignment, 
                                                                       cluster_sufficient_stats, cluster_parameters)
    weights[num_clusters+1] = new_cluster_log_predictive(cluster_sufficient_stats, cluster_parameters)
    weights += compute_data_log_predictives(observation, choices, data, data_sufficient_stats, data_parameters)
    return gumbel_max(choices, weights)
end

end
