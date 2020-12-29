module NtlMixture

using DataFrames
using Distributions
using LinearAlgebra
using Plots
using SpecialFunctions
using Statistics

# Mixture Generator

function generate_log_pmfs(psi_variates::Array{Float64})
    num_clusters = length(psi_variates)
    complement_psi_variates = 1 .- psi_variates
    log_psi_variates = log.(psi_variates)
    log_comple_psi_variates = log.(complement_psi_variates)

    log_pmfs = -ones(num_clusters, num_clusters)./0
    log_pmfs_diag = diagind(log_pmfs)
    log_pmfs[log_pmfs_diag] = log_psi_variates

    for j = 1:num_clusters
        log_pmfs[(j+1):num_clusters, j] .= log_psi_variates[j]
    end

    for i = 2:num_clusters
        log_pmfs[i:num_clusters, 1:(i-1)] .+= log_comple_psi_variates[i]
    end

    return log_pmfs
end

function sample_clusters(arrivals::Array{Int64})
    n = length(arrivals)
    cluster_assignments = Array{Int64}(undef, n)
    num_clusters = sum(arrivals) 
    new_cluster_indices = arrivals .=== 1
    arrival_times = (1:n)[new_cluster_indices]
    psi_variates  = rand(Beta(1, 1), num_clusters)
    psi_variates[1] = 1 

    log_pmfs = generate_log_pmfs(psi_variates)
    pmfs = exp.(log_pmfs)

    cluster_assignments[new_cluster_indices] = 1:num_clusters
    for i = 1:num_clusters
        arrival_time = arrival_times[i]
        if i < num_clusters
            next_arrival_time = arrival_times[i+1]
        else
            next_arrival_time = n + 1
        end
        observation_range = (arrival_time + 1):(next_arrival_time - 1)
        num_obs = length(observation_range)
        pmf = vec(pmfs[i, :])
        observation_assignments = rand(Categorical(pmf), num_obs)
        cluster_assignments[observation_range] = observation_assignments
    end

    return cluster_assignments 
end

function generate(n::Int64, dim::Int64, variance::Float64=0.1)
    phi_distribution = Beta(1, 1)
    phi = reshape(rand(phi_distribution, 1), 1)[1] 
    arrivals = rand(Binomial(1, phi), n-1)
    prepend!(arrivals, [1])
    num_clusters = sum(arrivals)

    assignments = sample_clusters(arrivals)

    prior_mean = vec(zeros(dim, 1))
    prior_cov = I
    cluster_means = rand(MvNormal(prior_mean, prior_cov), num_clusters)

    data_covariance = variance*I 
    data = Array{Float64}(undef, dim, n)

    for cluster = 1:num_clusters
        assigned_to_cluster = (assignments .=== cluster) 
        num_assigned = count(assigned_to_cluster)
        cluster_mean = vec(cluster_means[:, cluster])
        normal_distribution = MvNormal(cluster_mean, data_covariance)
        cluster_data = rand(normal_distribution, num_assigned)
        data[:, assigned_to_cluster] = cluster_data
    end

    assignments = DataFrame(reshape(assignments, length(assignments), 1))
    assignments = select(assignments, "x1" => "cluster")
    data = DataFrame(transpose(data))

    mixture = hcat(assignments, data)
    return mixture
end

function plot_assignments(assignments::Vector{Int64})
    plot(1:length(assignments), assignments, seriestype = :scatter)
end

## Mixture Fitter

function fit(data::Matrix, iterations::Int64)
    n = size(data)[2]
    dim = size(data)[1]
    instances = Array{Int64}(undef, n, iterations)
    instances[:, 1] .= 1
    cluster_num_observations = zeros(Int64, n)
    cluster_num_observations[1] = n
    cluster_means = zeros(Float64, dim, n)
    cluster_means[:, 1] = Statistics.mean(data, dims=2)
    posterior_means = zeros(Float64, dim, n)
    posterior_covs = Matrix{Float64}(undef, dim*dim, n)
    for i = 1:n
        posterior_covs[:, i] = vec(Matrix{Float64}(I, dim, dim))
    end
    compute_gaussian_posterior_parameters!(1, cluster_means, cluster_num_observations, posterior_means, posterior_covs)
    gibbs_sample!(instances, iterations, data, cluster_num_observations, cluster_means, posterior_means, 
                  posterior_covs)
    instances = DataFrame(instances)
    return instances
end

function gibbs_sample!(instances::Matrix{Int64}, iterations::Int64, data::Matrix{Float64}, 
                       cluster_num_observations::Vector{Int64}, cluster_means::Matrix{Float64}, 
                       posterior_means::Matrix{Float64}, posterior_covs::Matrix{Float64})
    for iteration = 2:iterations
        instances[:, iteration] = instances[:, iteration-1]
        for observation = 1:size(instances)[1]
            remove_observation!(observation, instances, iteration, data, cluster_num_observations, cluster_means, 
                                posterior_means, posterior_covs)
            (cluster, weight) = gibbs_move(observation, data, instances, iteration, cluster_num_observations, 
                                           posterior_means, posterior_covs)
            add_observation!(observation, cluster, instances, iteration, data, cluster_num_observations, cluster_means,
                             posterior_means, posterior_covs)
        end
    end
end

function gumbel_max(objects::Vector{Int64}, weights::Vector{Float64})
    gumbel_values = rand(Gumbel(0, 1), length(weights))
    index = argmax(gumbel_values + weights)
    return (objects[index], weights[index])
end

function update_gaussian_sufficient_statistics!(cluster::Int64, cluster_means::Matrix{Float64}, datum::Vector{Float64}, 
                                                cluster_num_observations::Vector{Int64}, update_type::String)
    cluster_mean = vec(cluster_means[:, cluster])
    n = cluster_num_observations[cluster]
    if update_type == "add"
        cluster_mean = (cluster_mean.*(n-1) + datum)./n
    elseif update_type == "remove"
        if n > 0
            cluster_mean = (cluster_mean.*(n+1) - datum)./n
        else
            cluster_mean .= 0
        end
    end
    cluster_means[:, cluster] = cluster_mean
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{Float64},
                             cluster_num_observations::Vector{Int64}, cluster_means::Matrix{Float64},
                             posterior_means::Matrix{Float64}, posterior_covs::Matrix{Float64})
    cluster = instances[observation, iteration]
    instances[observation, iteration] = size(instances)[1] + 1 
    datum = vec(data[:, observation])
    cluster_num_observations[cluster] -= 1
    update_gaussian_sufficient_statistics!(cluster, cluster_means, datum, cluster_num_observations, "remove")
    data_mean = vec(cluster_means[:, cluster])
    num_obs = cluster_num_observations[cluster]
    compute_gaussian_posterior_parameters!(cluster, cluster_means, cluster_num_observations, posterior_means, 
                                           posterior_covs)
    if (cluster === observation) && (cluster_num_observations[cluster] > 0)
        dim = length(datum)
        assigned_to_cluster = instances[:, iteration] .=== cluster
        new_cluster_time = findfirst(assigned_to_cluster)
        instances[assigned_to_cluster, iteration] .= new_cluster_time 
        cluster_num_observations[new_cluster_time] = cluster_num_observations[cluster]
        cluster_num_observations[cluster] = 0
        cluster_means[:, new_cluster_time] = cluster_means[:, cluster]
        cluster_means[:, cluster] .= 0
        posterior_means[:, new_cluster_time] = posterior_means[:, cluster]
        posterior_covs[:, new_cluster_time] = posterior_covs[:, cluster]
        posterior_means[:, cluster] .= 0
        posterior_covs[:, cluster] = vec(Matrix{Float64}(I, dim, dim))
    end
end

function add_observation!(observation::Int64, cluster::Int64, instances::Matrix{Int64}, iteration::Int64, 
                          data::Matrix{Float64}, cluster_num_observations::Vector{Int64}, 
                          cluster_means::Matrix{Float64}, posterior_means::Matrix{Float64}, 
                          posterior_covs::Matrix{Float64})
    if observation < cluster
        dim = size(data)[1]
        assigned_to_cluster = (instances[:, iteration] .=== cluster)
        old_cluster_time = cluster
        cluster = observation
        instances[assigned_to_cluster, iteration] .= cluster
        cluster_num_observations[cluster] = cluster_num_observations[old_cluster_time]
        cluster_num_observations[old_cluster_time] = 0
        cluster_means[:, cluster] = cluster_means[:, old_cluster_time]
        cluster_means[:, old_cluster_time] .= 0
        posterior_means[:, cluster] = posterior_means[:, old_cluster_time] 
        posterior_covs[:, cluster] = posterior_covs[:, old_cluster_time] 
        posterior_means[:, old_cluster_time] .= 0
        posterior_covs[:, old_cluster_time] = vec(Matrix{Float64}(I, dim, dim))
    end
    instances[observation, iteration] = cluster
    datum = vec(data[:, observation])
    cluster_num_observations[cluster] += 1
    update_gaussian_sufficient_statistics!(cluster, cluster_means, datum, cluster_num_observations, "add")
    compute_gaussian_posterior_parameters!(cluster, cluster_means, cluster_num_observations, posterior_means, 
                                           posterior_covs)
end

function geometric_new_log_weight(n::Int64, num_clusters::Int64)
    phi_prior_a = 1
    phi_prior_b = 1
    return log(num_clusters - 1 + phi_prior_a) - log(n - 1 + phi_prior_a + phi_prior_b)
end

function geometric_existing_log_weight(n::Int64, num_clusters::Int64)
    phi_prior_a = 1
    phi_prior_b = 1
    return log(n - num_clusters + phi_prior_a) - log(n - 1 + phi_prior_a + phi_prior_b)
end

function compute_num_observations(cluster::Int64, assignment::Vector{Int64})
    assigned_to_cluster = assignment .=== cluster
    return length(assignment[assigned_to_cluster])
end

function compute_num_complement(cluster::Int64, cluster_num_observations::Vector{Int64}, observation::Int64)
    younger_clusters = 1:(cluster-1)
    num_complement = sum(cluster_num_observations[younger_clusters]) -(cluster - 1)
    if observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_psi_posterior(cluster::Int64, cluster_num_observations::Vector{Int64}, observation::Int64)
    psi_posterior = Array{Int64}(undef, 2)
    psi_prior_a = 1
    psi_prior_b = 1 
    psi_posterior[1] = cluster_num_observations[cluster] - 1 + psi_prior_a
    psi_posterior[2] = compute_num_complement(cluster, cluster_num_observations, observation) + psi_prior_b
    return psi_posterior
end

function compute_ntl_predictive!(observation::Int64, clusters::Vector{Int64}, cluster_num_observations::Vector{Int64},
                                 log_weights::Vector{Float64})
    # Not strictly the number of observations
    n = maximum(clusters)
    n = maximum([n, observation])
    cluster_log_weights = zeros(Float64, n)
    complement_log_weights = zeros(Float64, n)
    psi_parameters = Array{Int64}(undef, 2, n)
    logbetas = Array{Float64}(undef, n)
    new_num_complement = compute_num_complement(observation, cluster_num_observations, observation)

    younger_cluster_new_psi = Array{Float64}(undef, 2)
    cluster_new_psi = Array{Float64}(undef, 2)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_psi_posterior(cluster, cluster_num_observations, observation)
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
            cluster_num_obs = cluster_num_observations[cluster]
            cluster_old_psi = psi_parameters[:, cluster]

            # Clusters younger than the observation
            younger_clusters = (observation+1):(cluster-1)
            younger_clusters = younger_clusters[cluster_num_observations[younger_clusters] .> 0]

            if cluster_num_observations[1] === 0
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

function compute_existing_cluster_weights(observation::Int64, assignment::Vector{Int64}, 
                                          cluster_num_observations::Vector{Int64})
    clusters = get_clusters(assignment)
    num_clusters = length(clusters)
    log_weights = Array{Float64}(undef, length(clusters))
    compute_ntl_predictive!(observation, clusters, cluster_num_observations, log_weights)
    log_weights .+= geometric_existing_log_weight(length(assignment), num_clusters)
    return log_weights
end

function compute_gaussian_posterior_parameters!(cluster::Int64, cluster_means::Matrix{Float64}, 
                                                cluster_num_observations::Vector{Int64}, 
                                                posterior_means::Matrix{Float64}, posterior_covs::Matrix{Float64})
    data_mean = vec(cluster_means[:, cluster])
    n = cluster_num_observations[cluster]
    dim = length(data_mean)
    prior_mean = zeros(dim)
    if n > 0
        data_precision = inv(0.1I)
        cov = inv(I + n*data_precision)
        posterior_mean = cov*(I*prior_mean + n*data_precision*data_mean)
    else
        posterior_mean = prior_mean
        cov = I
    end
    posterior_means[:, cluster] = vec(posterior_mean)
    posterior_covs[:, cluster] = vec(Matrix{Float64}(cov, dim, dim))
end

function compute_gaussian_log_predictive(datum::Vector{Float64}, posterior_mean::Vector{Float64}, 
                                         posterior_cov::Matrix{Float64})
    posterior = MvNormal(posterior_mean, 0.1I + posterior_cov)
    return logpdf(posterior, datum)
end

function compute_data_log_predictive(observation::Int64, clusters::Vector{Int64}, data::Matrix{Float64}, 
                                     posterior_means::Matrix{Float64}, posterior_covs::Matrix{Float64})
    num_clusters = length(clusters)
    log_predictive = Array{Float64}(undef, num_clusters)
    datum = vec(data[:, observation])
    dim = length(datum)
    for i = 1:num_clusters
        cluster = clusters[i]
        posterior_mean = vec(posterior_means[:, cluster])
        posterior_cov = reshape(posterior_covs[:, cluster], dim, dim)
        log_predictive[i] = compute_gaussian_log_predictive(datum, posterior_mean, posterior_cov)
    end
    return log_predictive
end

function get_clusters(assignment::Vector{Int64})
    # We represent the observation not assigned to a cluster, if it exists, by n+1
    assigned = assignment .<= length(assignment)
    return unique(sort(assignment[assigned]))
end

function gibbs_move(observation::Int64, data::Matrix{Float64}, instances::Matrix{Int64}, iteration::Int64, 
                    cluster_num_observations::Vector{Int64}, posterior_means::Matrix{Float64}, 
                    posterior_covs::Matrix{Float64})
    assignment = instances[:, iteration]
    clusters = get_clusters(assignment)
    n = size(data)[2]
    num_clusters = length(clusters)

    choices = Array{Int64}(undef, num_clusters+1)
    choices[1:num_clusters] = clusters
    choices[num_clusters+1] = observation

    weights = Array{Float64}(undef, num_clusters+1)
    weights[1:num_clusters] = compute_existing_cluster_weights(observation, assignment, cluster_num_observations)
    weights[num_clusters+1] = geometric_new_log_weight(n, num_clusters)
    weights += compute_data_log_predictive(observation, choices, data, posterior_means, posterior_covs)
    return gumbel_max(choices, weights)
end

function one_hot_encode(assignments::Vector{Int64})
    n = length(assignments)
    youngest_cluster = maximum(assignments)
    encoding_matrix = zeros(Int64, youngest_cluster, n)
    for observation = 1:n
        cluster = assignments[observation]
        encoding_matrix[cluster, observation] = 1
    end
    return Matrix(encoding_matrix)
end

function compute_co_occurrence_matrix(markov_chain::Matrix{Int64})
    n = size(markov_chain)[1]
    co_occurrence_matrix = zeros(Int64, n, n)
    num_instances = size(markov_chain)[2]
    for i = 2:num_instances
        assignment = markov_chain[:, i]
        ohe_assignment = one_hot_encode(assignment)
        co_occurrence_matrix += transpose(ohe_assignment) * ohe_assignment
    end
    return co_occurrence_matrix
end

end