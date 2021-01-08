module Fitter

using ..Models: ModelParameters, DataParameters, ClusterParameters, GaussianParameters, NtlParameters
using ..Models: DataSufficientStatistics, ClusterSufficientStatistics, GaussianSufficientStatistics
using ..Models: NtlSufficientStatistics, MultinomialParameters, MultinomialSufficientStatistics, GeometricArrivals
using Distributions
using LinearAlgebra
using SpecialFunctions: loggamma, logfactorial
import SpecialFunctions: logbeta
using Statistics
using ProgressBars

logbeta(parameters::Vector{T}) where {T<:Real} = logbeta(parameters[1], parameters[2])

function logmvbeta(parameters::Vector{T}) where {T <: Real}
    return sum(loggamma.(parameters)) - loggamma(sum(parameters))
end

function log_multinomial_coeff(counts::Vector{Int64})
    return logfactorial(sum(counts)) - sum(logfactorial.(counts)) 
end

function prepare_data_sufficient_statistics(data::Matrix{Float64}, data_parameters::GaussianParameters)
    dim = size(data)[1]
    n = size(data)[2]
    data_means = zeros(Float64, dim, n)
    data_precision_quadratic_sums = zeros(Float64, n)
    posterior_means = Matrix{Float64}(undef, dim, n)
    for i = 1:n
        posterior_means[:, i] = data_parameters.prior_mean
    end
    posterior_covs = Matrix{Float64}(undef, dim*dim, n)
    for i = 1:n
        posterior_covs[:, i] = vec(data_parameters.prior_covariance)
    end
    return GaussianSufficientStatistics(data_means, data_precision_quadratic_sums, posterior_means, posterior_covs)
end

function prepare_data_sufficient_statistics(data::Matrix{Int64}, data_parameters::MultinomialParameters)
    dim = size(data)[1]
    n = size(data)[2]
    counts = zeros(Int64, dim, n)
    posterior_dirichlet_scale = Matrix{Float64}(undef, dim, n)
    for i = 1:n
        posterior_dirichlet_scale[:, i] = vec(data_parameters.prior_dirichlet_scale)
    end
    return MultinomialSufficientStatistics(counts, posterior_dirichlet_scale)
end

function prepare_cluster_sufficient_statistics(::Type{T}, n::Int64) where {T <: NtlParameters}
    cluster_num_observations = Vector{Int64}(zeros(Int64, n))
    cluster_sufficient_stats = NtlSufficientStatistics(cluster_num_observations, 1)
    return cluster_sufficient_stats
end

function fit(data::Matrix{T}, data_parameters::DataParameters, cluster_parameters::ClusterParameters;
             num_instances::Int64=100, method::String="gibbs") where {T <: Real}
    n = size(data)[2]
    dim = size(data)[1]
    instances = Array{Int64}(undef, n, num_instances)
    if method === "gibbs"
        gibbs_sample!(instances, data, data_parameters, cluster_parameters)
    elseif method === "smc" 
        smc!(instances, data, data_parameters, cluster_parameters)
    else
        message = "$method is not an appropriate fit method."
        throw(ArgumentError(message))
    end
    return instances
end

function gibbs_sample!(instances::Matrix{Int64}, data::Matrix{T}, data_parameters::DataParameters,
                       cluster_parameters::ClusterParameters) where {T <: Real}
    n = size(data)[2]
    cluster_sufficient_stats = prepare_cluster_sufficient_statistics(typeof(cluster_parameters), n)
    data_sufficient_stats = prepare_data_sufficient_statistics(data, data_parameters)
    # Assign all of the observations to the first cluster
    for observation = 1:n
        add_observation!(observation, 1, instances, 1, data, cluster_sufficient_stats, data_sufficient_stats, 
                         data_parameters)
    end
    iterations = size(instances)[2]
    for iteration = ProgressBar(2:iterations)
        instances[:, iteration] = instances[:, iteration-1]
        for observation = 1:size(instances)[1]
            remove_observation!(observation, instances, iteration, data, cluster_sufficient_stats, 
                                data_sufficient_stats, data_parameters)
            (cluster, weight) = gibbs_move(observation, data, cluster_sufficient_stats, data_sufficient_stats, 
                                           data_parameters, cluster_parameters)
            add_observation!(observation, cluster, instances, iteration, data, cluster_sufficient_stats, 
                             data_sufficient_stats, data_parameters)
        end
    end
end

function compute_cluster_data_log_likelihood(cluster::Int64, data_sufficient_stats::MultinomialSufficientStatistics,
                                             data_parameters::MultinomialParameters, 
                                             ::ClusterSufficientStatistics)
    dirichlet_posterior = data_sufficient_stats.posterior_dirichlet_scale[:, cluster]
    dirichlet_prior = data_parameters.prior_dirichlet_scale
    counts = convert(Vector{Int64}, dirichlet_posterior - dirichlet_prior)
    log_likelihood = log_multinomial_coeff(counts) + logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
end

function compute_cluster_data_log_likelihood(cluster::Int64, data_sufficient_stats::GaussianSufficientStatistics, 
                                             data_parameters::GaussianParameters, 
                                             cluster_sufficient_stats::ClusterSufficientStatistics)
    posterior_mean = vec(data_sufficient_stats.posterior_means[:, cluster])
    dim = length(posterior_mean)
    posterior_covariance = reshape(data_sufficient_stats.posterior_covs[:, cluster], dim, dim)
    posterior_precision = inv(posterior_covariance)
    prior_mean = data_parameters.prior_mean
    prior_covariance = data_parameters.prior_covariance
    prior_precision = inv(prior_covariance)

    data_precision_quadratic_sum = data_sufficient_stats.data_precision_quadratic_sums[cluster]
    posterior_quadratic = transpose(posterior_mean) * posterior_covariance * posterior_mean 
    prior_quadratic = transpose(prior_mean) * prior_precision * prior_mean

    log_likelihood = -(1/2)*(data_precision_quadratic_sum + prior_quadratic - posterior_quadratic)

    data_log_determinant = log(abs(det(data_parameters.data_covariance)))
    prior_log_determinant = log(abs(det(prior_covariance)))
    posterior_log_determinant = log(abs(det(posterior_covariance)))

    n = cluster_sufficient_stats.num_observations[cluster]
    log_likelihood += log(2*pi)*dim*(n + 1)/2 - (n/2)data_log_determinant - (1/2)prior_log_determinant
    return log_likelihood
end

function compute_data_log_likelihood(data_sufficient_stats::DataSufficientStatistics, 
                                     data_parameters::DataParameters, 
                                     cluster_sufficient_stats::ClusterSufficientStatistics,
                                     clusters::Vector{Int64})
    log_likelihood = 0
    for cluster = clusters
        log_likelihood += compute_cluster_data_log_likelihood(cluster, data_sufficient_stats, data_parameters,
                                                              cluster_sufficient_stats)
    end
    return log_likelihood
end

function compute_cluster_assignment_log_likelihood(cluster::Int64, cluster_sufficient_stats::NtlSufficientStatistics, 
                                                   cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    psi_posterior = compute_stick_breaking_posterior(cluster, cluster_sufficient_stats, cluster_parameters)
    return logbeta(psi_posterior) - logbeta(cluster_parameters.prior)
end

function compute_arrivals_log_likelihood(cluster_sufficient_stats::NtlSufficientStatistics, 
                                         cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return logbeta(phi_posterior) - logbeta(cluster_parameters.arrival_distribution.prior)
end

function compute_assignment_log_likelihood(cluster_sufficient_stats::ClusterSufficientStatistics, 
                                           cluster_parameters::ClusterParameters, clusters::Vector{Int64})
    log_likelihood = 0
    for cluster = clusters
        log_likelihood += compute_cluster_assignment_log_likelihood(cluster, cluster_sufficient_stats, 
                                                                    cluster_parameters)
    end
    log_likelihood += compute_arrivals_log_likelihood(cluster_sufficient_stats, cluster_parameters)
end

function compute_joint_log_likelihood(data_sufficient_stats::DataSufficientStatistics, 
                                      cluster_sufficient_stats::ClusterSufficientStatistics,
                                      data_parameters::DataParameters,
                                      cluster_parameters::ClusterParameters)
    clusters = get_clusters(cluster_sufficient_stats.num_observations)
    log_likelihood = compute_data_log_likelihood(data_sufficient_stats, data_parameters, cluster_sufficient_stats, 
                                                 clusters)
    log_likelihood += compute_assignment_log_likelihood(cluster_sufficient_stats, cluster_parameters, clusters)
    return log_likelihood
end

function compute_log_weight!(log_weights::Vector{Float64}, log_likelihoods::Vector{Float64}, 
                             proposal_log_weight::Float64, 
                             data_sufficient_stats::DataSufficientStatistics, 
                             cluster_sufficient_stats::ClusterSufficientStatistics,
                             data_parameters::DataParameters, cluster_parameters::ClusterParameters, particle::Int64)
    previous_log_weight = log_weights[particle]
    previous_log_likelihood = log_likelihoods[particle]
    log_likelihood = compute_joint_log_likelihood(data_sufficient_stats, cluster_sufficient_stats, data_parameters, 
                                                  cluster_parameters)
    log_weights[particle] = previous_log_weight + log_likelihood - previous_log_likelihood - proposal_log_weight
    log_likelihoods[particle] = log_likelihood
end

function make_data_sufficient_stats_array(::Type{T}, num_particles::Int64) where {T<:GaussianParameters} 
    return Array{GaussianSufficientStatistics}(undef, num_particles)
end

function make_cluster_sufficient_stats_array(::Type{T}, num_particles::Int64) where {T<:NtlParameters} 
    return Array{NtlSufficientStatistics}(undef, num_particles)
end

function smc!(particles::Matrix{Int64}, data::Matrix{T}, data_parameters::DataParameters, 
              cluster_parameters::ClusterParameters) where {T <: Real}
    n = size(data)[2]
    particles .= n+1
    num_particles = size(particles)[2]
    log_likelihoods = zeros(Float64, num_particles)
    log_weights = Array{Float64}(undef, num_particles)

    # Prepare sufficient statistics for each particle
    data_sufficient_stats_array = make_data_sufficient_stats_array(typeof(data_parameters), num_particles)
    cluster_sufficient_stats_array = make_cluster_sufficient_stats_array(typeof(cluster_parameters), num_particles)
    for particle = 1:num_particles
        data_sufficient_stats_array[particle] = prepare_data_sufficient_statistics(data, data_parameters)
        cluster_sufficient_stats_array[particle] = prepare_cluster_sufficient_statistics(typeof(cluster_parameters), n)
    end
    # Assign first observation to first cluster
    for particle = 1:num_particles
        cluster_sufficient_stats = cluster_sufficient_stats_array[particle]
        data_sufficient_stats = data_sufficient_stats_array[particle]
        add_observation!(1, 1, particles, particle, data, cluster_sufficient_stats, data_sufficient_stats,
                         data_parameters)
    end
    for observation = ProgressBar(2:n)
        for particle = 1:num_particles
            cluster_sufficient_stats = cluster_sufficient_stats_array[particle]
            data_sufficient_stats = data_sufficient_stats_array[particle]
            (cluster, weight) = gibbs_move(observation, data, cluster_sufficient_stats, data_sufficient_stats, 
                                           data_parameters, cluster_parameters)
            add_observation!(observation, cluster, particles, particle, data, cluster_sufficient_stats, 
                             data_sufficient_stats, data_parameters)
            compute_log_weight!(log_weights, log_likelihoods, weight, data_sufficient_stats, cluster_sufficient_stats, 
                                data_parameters, cluster_parameters, particle)
        end
        weights = exp.(log_weights)
        normalized_weights = weights./sum(weights)
        ess = 1/sum(normalized_weights.^2)
        if ess < num_particles/2
            resampled_indices = rand(Categorical(normalized_weights), num_particles)
            particles = particles[:, resampled_indices]
            log_weights = log_weights[resampled_indices]
            log_likelihoods = log_likelihoods[resampled_indices]
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
                                            data_parameters::GaussianParameters; update_type::String="add")
    cluster_mean = vec(data_sufficient_stats.data_means[:, cluster])
    data_precision_quadratic_sum = data_sufficient_stats.data_precision_quadratic_sums[cluster]
    n = cluster_sufficient_stats.num_observations[cluster]
    datum_precision_quadratic_sum = transpose(datum)*data_parameters.data_precision*datum
    if update_type == "add"
        cluster_mean = (cluster_mean.*(n-1) + datum)./n
        data_precision_quadratic_sum += datum_precision_quadratic_sum
    elseif update_type == "remove"
        if n > 0
            cluster_mean = (cluster_mean.*(n+1) - datum)./n
            data_precision_quadratic_sum -= datum_precision_quadratic_sum
        else
            cluster_mean .= 0
            data_precision_quadratic_sum = 0
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
    data_sufficient_stats.data_means[:, cluster] = cluster_mean
    data_sufficient_stats.data_precision_quadratic_sums[cluster] = data_precision_quadratic_sum
end

function update_data_sufficient_statistics!(data_sufficient_stats::MultinomialSufficientStatistics,
                                            cluster::Int64, datum::Vector{Int64}, ::ClusterSufficientStatistics,
                                            ::MultinomialParameters; update_type::String="add")
    cluster_counts = vec(data_sufficient_stats.total_counts[:, cluster])
    if update_type === "add"
        cluster_counts += datum 
    elseif update_type === "remove"
        cluster_counts -= datum
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
    data_sufficient_stats.total_counts[:, cluster] = cluster_counts
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::ClusterSufficientStatistics, 
                                              new_time::Int64, old_time::Int64)
    cluster_sufficient_stats.num_observations[new_time] = cluster_sufficient_stats.num_observations[old_time]
    cluster_sufficient_stats.num_observations[old_time] = 0
end

function update_data_stats_new_birth_time!(data_sufficient_stats::GaussianSufficientStatistics, 
                                           new_time::Int64, old_time::Int64, data_parameters::GaussianParameters)
    data_sufficient_stats.data_means[:, new_time] = data_sufficient_stats.data_means[:, old_time]
    data_sufficient_stats.data_means[:, old_time] .= 0
    data_sufficient_stats.posterior_means[:, new_time] = data_sufficient_stats.posterior_means[:, old_time]
    data_sufficient_stats.posterior_covs[:, new_time] = data_sufficient_stats.posterior_covs[:, old_time]
    data_sufficient_stats.posterior_means[:, old_time] = data_parameters.prior_mean
    data_sufficient_stats.posterior_covs[:, old_time] = vec(data_parameters.prior_covariance)
    data_sufficient_stats.data_precision_quadratic_sums[new_time] = data_sufficient_stats.data_precision_quadratic_sums[old_time]
    data_sufficient_stats.data_precision_quadratic_sums[old_time] = 0
end

function update_data_stats_new_birth_time!(data_sufficient_stats::MultinomialSufficientStatistics, 
                                           new_time::Int64, old_time::Int64, data_parameters::MultinomialParameters)
    data_sufficient_stats.total_counts[:, new_time] = data_sufficient_stats.total_counts[:, old_time]
    data_sufficient_stats.total_counts[:, old_time] .= 0
    data_sufficient_stats.posterior_dirichlet_scale[:, new_time] = data_sufficient_stats.posterior_dirichlet_scale[:, old_time]
    data_sufficient_stats.posterior_dirichlet_scale[:, old_time] = data_parameters.prior_dirichlet_scale
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             cluster_sufficient_stats::ClusterSufficientStatistics, 
                             data_sufficient_stats::DataSufficientStatistics,
                             data_parameters::DataParameters) where {T <: Real}
    cluster = instances[observation, iteration]
    instances[observation, iteration] = size(instances)[1] + 1 
    datum = vec(data[:, observation])
    cluster_sufficient_stats.num_observations[cluster] -= 1
    if cluster_sufficient_stats.num_observations[cluster] === 0
        cluster_sufficient_stats.num_clusters -= 1
    end
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, data_parameters, 
                                       update_type="remove")
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
                          data::Matrix{T}, cluster_sufficient_stats::ClusterSufficientStatistics,
                          data_sufficient_stats::DataSufficientStatistics, 
                          data_parameters::DataParameters) where {T <: Real}
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
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, data_parameters, 
                                       update_type="add")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, data_parameters)
end

function compute_arrival_distribution_posterior(cluster_sufficient_stats::NtlSufficientStatistics,
                                                cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    n = length(cluster_sufficient_stats.num_observations)
    num_clusters = cluster_sufficient_stats.num_clusters
    phi_posterior = cluster_parameters.arrival_distribution.prior
    phi_posterior[1] += num_clusters - 1
    phi_posterior[2] += n - num_clusters 
    return phi_posterior
end

function new_cluster_log_predictive(cluster_sufficient_stats::NtlSufficientStatistics, 
                                    cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return log(phi_posterior[1]) - log(sum(phi_posterior))
end

function existing_cluster_log_predictive(cluster_sufficient_stats::NtlSufficientStatistics, 
                                         cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return log(phi_posterior[2]) - log(sum(phi_posterior))
end

function compute_num_complement(cluster::Int64, cluster_num_observations::Vector{Int64}; 
                                observation::Union{Int64, Nothing}=nothing)
    younger_clusters = 1:(cluster-1)
    num_complement = sum(cluster_num_observations[younger_clusters]) - (cluster - 1)
    if observation !== nothing && observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_stick_breaking_posterior(cluster::Int64, ntl_sufficient_stats::NtlSufficientStatistics, 
                               ntl_parameters::NtlParameters; observation::Union{Int64, Nothing}=nothing)
    psi_posterior = Array{Int64}(undef, 2)
    psi_prior_a = ntl_parameters.prior[1]
    psi_prior_b = ntl_parameters.prior[2] 
    psi_posterior[1] = ntl_sufficient_stats.num_observations[cluster] - 1 + psi_prior_a
    psi_posterior[2] = compute_num_complement(cluster, ntl_sufficient_stats.num_observations, observation=observation) 
    psi_posterior[2] += psi_prior_b
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
    new_num_complement = compute_num_complement(observation, ntl_sufficient_stats.num_observations, 
                                                observation=observation)
    younger_cluster_new_psi = Array{Float64}(undef, 2)
    cluster_new_psi = Array{Float64}(undef, 2)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_stick_breaking_posterior(cluster, ntl_sufficient_stats, ntl_parameters, 
                                                           observation=observation)
            psi_parameters[:, cluster] = cluster_psi
            denom = cluster_psi[1] + cluster_psi[2]
            log_denom = log(denom)
            cluster_log_weights[cluster] = log(cluster_psi[1]) - log_denom
            complement_log_weights[cluster] = log(cluster_psi[2]) - log_denom
            if observation < cluster
                logbetas[cluster] = logbeta(cluster_psi)
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

function compute_existing_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, 
                                                  cluster_sufficient_stats::ClusterSufficientStatistics, 
                                                  cluster_parameters::ClusterParameters)
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

function compute_data_posterior_parameters!(cluster::Int64, ::NtlSufficientStatistics,
                                            data_sufficient_stats::MultinomialSufficientStatistics,
                                            data_parameters::MultinomialParameters)
    cluster_counts = data_sufficient_stats.total_counts[:, cluster]
    prior_dirichlet_scale = data_parameters.prior_dirichlet_scale
    data_sufficient_stats.posterior_dirichlet_scale[:, cluster] = cluster_counts + prior_dirichlet_scale 
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

function data_log_predictive(datum::Vector{Int64}, cluster::Int64, 
                             data_sufficient_stats::MultinomialSufficientStatistics,
                             ::MultinomialParameters)
    posterior_scale = vec(data_sufficient_stats.posterior_dirichlet_scale[:, cluster])
    n = sum(datum)
    posterior = DirichletMultinomial(n, posterior_scale)
    return logpdf(posterior, datum)
end

function compute_data_log_predictives(observation::Int64, clusters::Vector{Int64}, data::Matrix{T}, 
                                      data_sufficient_stats::DataSufficientStatistics, 
                                      data_parameters::DataParameters) where {T <: Real}
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

function get_clusters(num_observations::Vector{Int64})
    # We represent the observation not assigned to a cluster, if it exists, by n+1
    return findall(>(0), num_observations)
end

function gibbs_move(observation::Int64, data::Matrix{T}, 
                    cluster_sufficient_stats::ClusterSufficientStatistics, 
                    data_sufficient_stats::DataSufficientStatistics, data_parameters::DataParameters, 
                    cluster_parameters::ClusterParameters) where {T <: Real}
    clusters = get_clusters(cluster_sufficient_stats.num_observations)
    n = size(data)[2]
    num_clusters = length(clusters)

    choices = Array{Int64}(undef, num_clusters+1)
    choices[1:num_clusters] = clusters
    choices[num_clusters+1] = observation

    weights = Array{Float64}(undef, num_clusters+1)
    weights[1:num_clusters] = compute_existing_cluster_log_predictives(observation, clusters, 
                                                                       cluster_sufficient_stats, cluster_parameters)
    weights[num_clusters+1] = new_cluster_log_predictive(cluster_sufficient_stats, cluster_parameters)
    weights += compute_data_log_predictives(observation, choices, data, data_sufficient_stats, data_parameters)
    return gumbel_max(choices, weights)
end

end
