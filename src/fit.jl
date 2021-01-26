module Fitter

using ..Models: DataParameters, ClusterParameters, GaussianParameters, NtlParameters, DpParameters
using ..Models: DataSufficientStatistics, MixtureSufficientStatistics, GaussianSufficientStatistics
using ..Models: MultinomialParameters, MultinomialSufficientStatistics, GeometricArrivals
using ..Models: NtlHmmSufficientStatistics, HmmSufficientStatistics
using ..Models: Model, Mixture, HiddenMarkovModel, ClusterSufficientStatistics, Changepoint
using ..Models: ParametricArrivalsClusterParameters, BetaNtlParameters
using ..Models: ChangepointSufficientStatistics, PitmanYorArrivals 
using ..Samplers: Sampler, GibbsSampler, SequentialMonteCarlo
using ..Utils: logbeta, logmvbeta, log_multinomial_coeff, gumbel_max, isnothing

using Distributions
using LinearAlgebra
using Statistics
using ProgressBars

asserting() = true # when set to true, this will enable all `@assertion`s

macro assertion(test)
    esc(:(if $(@__MODULE__).asserting()
        @assert($test)
    end))
end

function fit(data::Matrix{T}, model::Union{Mixture, Changepoint}, sampler::GibbsSampler) where {T <: Real}
    iterations = sampler.num_iterations
    n = size(data)[2]
    dim = size(data)[1]
    instances = Array{Int64}(undef, n, iterations)
    n = size(data)[2]
    cluster_sufficient_stats = prepare_cluster_sufficient_statistics(model, n)
    data_sufficient_stats = prepare_data_sufficient_statistics(data, model.data_parameters)
    # Assign all of the observations to the first cluster
    for observation = 1:n
        add_observation!(observation, 1, instances, 1, data, cluster_sufficient_stats, data_sufficient_stats, model)
    end
    for iteration = ProgressBar(2:iterations)
        instances[:, iteration] = instances[:, iteration-1]
        for observation = 1:n
            remove_observation!(observation, instances, iteration, data, cluster_sufficient_stats, 
                                data_sufficient_stats, model)
            (cluster, weight) = gibbs_move(observation, data, cluster_sufficient_stats, data_sufficient_stats, model)
            add_observation!(observation, cluster, instances, iteration, data, cluster_sufficient_stats, 
                             data_sufficient_stats, model)

            @assertion sum(cluster_sufficient_stats.num_observations) === n
        end
    end
    return instances
end

function fit(data::Matrix{T}, model::Mixture, sampler::SequentialMonteCarlo) where {T <: Real}
    num_particles = sampler.num_particles
    ess_threshold = sampler.ess_threshold
    n = size(data)[2]
    dim = size(data)[1]
    particles = Array{Int64}(undef, n, num_particles)
    particles .= n+1
    num_particles = size(particles)[2]
    log_likelihoods = zeros(Float64, num_particles)
    log_weights = Array{Float64}(undef, num_particles)

    cluster_sufficient_stats_array = prepare_cluster_sufficient_statistics(model, num_particles, n)
    # Prepare sufficient statistics for each particle
    data_sufficient_stats_array = prepare_data_sufficient_statistics(num_particles, data, model.data_parameters)

    # Assign first observation to first cluster
    for particle = 1:num_particles
        cluster_sufficient_stats = cluster_sufficient_stats_array[particle]
        data_sufficient_stats = data_sufficient_stats_array[particle]
        cluster = 1
        observation = 1
        add_observation!(observation, cluster, particles, particle, data, cluster_sufficient_stats, 
                         data_sufficient_stats, model)
        cluster_sufficient_stats_array[particle] = cluster_sufficient_stats
        data_sufficient_stats_array[particle] = data_sufficient_stats
    end
    for observation = ProgressBar(2:n)
        for particle = 1:num_particles
            cluster_sufficient_stats = cluster_sufficient_stats_array[particle]
            data_sufficient_stats = data_sufficient_stats_array[particle]

            prev_cluster = particles[observation-1, particle]
            (cluster, weight) = propose(observation, prev_cluster, data, cluster_sufficient_stats, 
                                        data_sufficient_stats, model)

            add_observation!(observation, cluster, particles, particle, data, cluster_sufficient_stats, 
                             data_sufficient_stats, model)

            compute_log_weight!(log_weights, log_likelihoods, weight, data_sufficient_stats, cluster_sufficient_stats, 
                                model, particle)

            cluster_sufficient_stats_array[particle] = cluster_sufficient_stats
            data_sufficient_stats_array[particle] = data_sufficient_stats
        end
        weights = exp.(log_weights)
        normalized_weights = weights./sum(weights)
        ess = 1/sum(normalized_weights.^2)
        if ess < ess_threshold 
            resampled_indices = rand(Categorical(normalized_weights), num_particles)
            particles = particles[:, resampled_indices]
            log_weights .= 1/num_particles
            log_likelihoods = log_likelihoods[resampled_indices]
            cluster_sufficient_stats_array = cluster_sufficient_stats_array[resampled_indices]
            data_sufficient_stats_array = data_sufficient_stats_array[resampled_indices]
        end
    end
    return particles
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

function prepare_data_sufficient_statistics(num_particles::Int64, data::Matrix{Float64}, 
                                            data_parameters::GaussianParameters)
    sufficient_stats = Array{GaussianSufficientStatistics}(undef, num_particles)
    for particle = 1:num_particles
        sufficient_stats[particle] = prepare_data_sufficient_statistics(data, data_parameters)
    end
    return sufficient_stats
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

function prepare_data_sufficient_statistics(num_particles::Int64, data::Matrix{Int64}, 
                                            data_parameters::MultinomialParameters)
    sufficient_stats = Array{MultinomialSufficientStatistics}(undef, num_particles)
    for particle = 1:num_particles
        sufficient_stats[particle] = prepare_data_sufficient_statistics(data, data_parameters)
    end
    return sufficient_stats
end

function prepare_cluster_sufficient_statistics(::Changepoint, n::Int64)
    changepoint_num_observations = Vector{Int64}(zeros(Int64, n))
    changepoints = BitArray(undef, n)
    changepoints .= false
    changepoint_sufficient_stats = ChangepointSufficientStatistics(changepoint_num_observations, changepoints)
    return changepoint_sufficient_stats
end

function prepare_cluster_sufficient_statistics(::Mixture, n::Int64)
    cluster_num_observations = Vector{Int64}(zeros(Int64, n))
    clusters = BitArray(undef, n)
    clusters .= false
    cluster_sufficient_stats = MixtureSufficientStatistics(cluster_num_observations, clusters)
    return cluster_sufficient_stats
end

function prepare_cluster_sufficient_statistics(::HiddenMarkovModel{T}, n::Int64) where {T <: NtlParameters}
    cluster_num_observations = zeros(Int64, n+1, n+1) # (source state) -> (target state)
    clusters = BitArray(undef, n)
    clusters .= false
    cluster_sufficient_stats = NtlHmmSufficientStatistics(cluster_num_observations, clusters)
end

function prepare_cluster_sufficient_statistics(model::Mixture, num_states::Int64, n::Int64)
    sufficient_stats = Array{MixtureSufficientStatistics}(undef, num_states)
    for state = 1:num_states
        sufficient_stats[state] = prepare_cluster_sufficient_statistics(model, n)
    end
    return sufficient_stats
end

function prepare_cluster_sufficient_statistics(model::HiddenMarkovModel{T}, num_particles::Int64, n::Int64) where {T <: NtlParameters}
    # The (n+1)th state corresponds to the non-existent null state
    sufficient_stats = Array{NtlHmmSufficientStatistics}(undef, num_particles)
    for state = 1:num_states
        sufficient_stats[state] = prepare_cluster_sufficient_statistics(model, n)
    end
    return sufficient_stats
end

function prepare_cluster_sufficient_statistics(::ClusterParameters, num_observations::Vector{Int64}, clusters::BitArray)
    cluster_sufficient_stats = MixtureSufficientStatistics(num_observations, clusters)
    return cluster_sufficient_stats
end

function compute_cluster_data_log_likelihood(cluster::Int64, data_sufficient_stats::MultinomialSufficientStatistics,
                                             data_parameters::MultinomialParameters, ::MixtureSufficientStatistics)
    dirichlet_posterior = data_sufficient_stats.posterior_dirichlet_scale[:, cluster]
    dirichlet_prior = data_parameters.prior_dirichlet_scale
    counts = convert(Vector{Int64}, dirichlet_posterior - dirichlet_prior)
    log_likelihood = log_multinomial_coeff(counts) + logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
    return log_likelihood
end

function compute_cluster_data_log_likelihood(cluster::Int64, data_sufficient_stats::GaussianSufficientStatistics, 
                                             data_parameters::GaussianParameters, 
                                             cluster_sufficient_stats::MixtureSufficientStatistics)
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

function compute_data_log_likelihood(data_sufficient_stats::DataSufficientStatistics, data_parameters::DataParameters, 
                                     cluster_sufficient_stats::MixtureSufficientStatistics, clusters::Vector{Int64})
    log_likelihood = 0
    for cluster = clusters
        log_likelihood += compute_cluster_data_log_likelihood(cluster, data_sufficient_stats, data_parameters,
                                                              cluster_sufficient_stats)
    end
    return log_likelihood
end

function compute_cluster_assignment_log_likelihood(cluster::Int64, num_observations::Vector{Int64},
                                                   cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    psi_posterior = compute_stick_breaking_posterior(cluster, num_observations, cluster_parameters.prior)
    return logbeta(psi_posterior) - logbeta(cluster_parameters.prior)
end

function compute_cluster_assignment_log_likelihood(cluster::Int64, cluster_sufficient_stats::MixtureSufficientStatistics, 
                                                   cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    return compute_cluster_assignment_log_likelihood(cluster, cluster_sufficient_stats.num_observations, 
                                                     cluster_parameters)
end

function compute_arrivals_log_likelihood(cluster_sufficient_stats::ClusterSufficientStatistics, 
                                         cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return logbeta(phi_posterior) - logbeta(cluster_parameters.arrival_distribution.prior)
end

function compute_assignment_log_likelihood(cluster_sufficient_stats::MixtureSufficientStatistics, 
                                           cluster_parameters::ParametricArrivalsClusterParameters{T}, 
                                           clusters::Vector{Int64}) where T
    log_likelihood = 0
    for cluster = clusters
        log_likelihood += compute_cluster_assignment_log_likelihood(cluster, cluster_sufficient_stats, 
                                                                    cluster_parameters)
    end
    log_likelihood += compute_arrivals_log_likelihood(cluster_sufficient_stats, cluster_parameters)
    return log_likelihood
end

function compute_assignment_log_likelihood(cluster_sufficient_stats::NtlHmmSufficientStatistics,
                                           cluster_parameters::ClusterParameters, clusters::Vector{Int64})
    log_likelihood = 0
    n = length(cluster_sufficient_stats.clusters)
    for cluster = clusters
        source_clusters = findall(>(0), cluster_sufficient_stats.num_observations[1:n, cluster])
        for source_cluster = source_clusters
            num_observations = cluster_sufficient_stats.num_observations[source_cluster, 1:n]
            log_likelihood += compute_cluster_assignment_log_likelihood(cluster, num_observations, cluster_parameters)
        end
    end
    log_likelihood += compute_arrivals_log_likelihood(cluster_sufficient_stats, cluster_parameters)
    return log_likelihood
end

function compute_joint_log_likelihood(data_sufficient_stats::DataSufficientStatistics, 
                                      cluster_sufficient_stats::ClusterSufficientStatistics, model::Mixture)
    data_parameters = model.data_parameters
    cluster_parameters = model.cluster_parameters
    clusters = get_clusters(cluster_sufficient_stats)
    log_likelihood = compute_data_log_likelihood(data_sufficient_stats, data_parameters, cluster_sufficient_stats, 
                                                 clusters)
    log_likelihood += compute_assignment_log_likelihood(cluster_sufficient_stats, cluster_parameters, clusters)
    return log_likelihood
end

function compute_log_weight!(log_weights::Vector{Float64}, log_likelihoods::Vector{Float64}, 
                             proposal_log_weight::Float64, data_sufficient_stats::DataSufficientStatistics, 
                             cluster_sufficient_stats::ClusterSufficientStatistics,
                             model::Model, particle::Int64)
    previous_log_weight = log_weights[particle]
    previous_log_likelihood = log_likelihoods[particle]
    log_likelihood = compute_joint_log_likelihood(data_sufficient_stats, cluster_sufficient_stats, model)
    log_weights[particle] = previous_log_weight + log_likelihood - previous_log_likelihood - proposal_log_weight
    log_likelihoods[particle] = log_likelihood
end

# TODO: make an HMM version of this function
function update_data_sufficient_statistics!(data_sufficient_stats::GaussianSufficientStatistics, 
                                            cluster::Int64, datum::Vector{Float64},
                                            cluster_sufficient_stats::Union{MixtureSufficientStatistics, ChangepointSufficientStatistics},
                                            model::Union{Mixture, Changepoint};
                                            update_type::String="add")
    cluster_mean = vec(data_sufficient_stats.data_means[:, cluster])
    data_precision_quadratic_sum = data_sufficient_stats.data_precision_quadratic_sums[cluster] 
    n = cluster_sufficient_stats.num_observations[cluster]
    datum_precision_quadratic_sum = transpose(datum)*model.data_parameters.data_precision*datum
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
                                            cluster::Int64, datum::Vector{Int64}, 
                                            ::Union{MixtureSufficientStatistics, ChangepointSufficientStatistics},
                                            ::Union{Mixture, Changepoint}; update_type::String="add")
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

function update_data_sufficient_statistics!(data_sufficient_stats::GaussianSufficientStatistics,
                                            changepoint::Int64, data::Matrix{T}, 
                                            changepoint_sufficient_stats::ChangepointSufficientStatistics, 
                                            model::Changepoint; 
                                            update_type="add") where {T <: Real}
    changepoint_mean = vec(data_sufficient_stats.data_means[:, changepoint])
    data_precision_quadratic_sum = data_sufficient_stats.data_precision_quadratic_sums[changepoint] 

    n = changepoint_sufficient_stats.num_observations[changepoint]
    new_data_size = size(data)[2]
    new_precision_quadratic_sum = sum(transpose(data) * model.data_parameters.data_precision .* transpose(data))
    new_data_sum = sum(data, dims=2)

    if update_type === "new"
        if new_data_size > 0
            changepoint_mean = new_data_sum/new_data_size 
            data_precision_quadratic_sum = new_precision_quadratic_sum

            data_sufficient_stats.data_means[:, changepoint] = changepoint_mean
            data_sufficient_stats.data_precision_quadratic_sums[changepoint] = data_precision_quadratic_sum
        end
    elseif update_type === "add"
        changepoint_mean = (n.*changepoint_mean + new_data_sum)./(n + new_data_size)
        data_precision_quadratic_sum += new_precision_quadratic_sum
        data_sufficient_stats.data_means[:, changepoint] = changepoint_mean
        data_sufficient_stats.data_precision_quadratic_sums[changepoint] = data_precision_quadratic_sum
    elseif update_type === "remove"
        changepoint_mean = (n.*changepoint_mean - new_data_sum)./(n - new_data_size)
        data_precision_quadratic_sum -= new_precision_quadratic_sum
        data_sufficient_stats.data_means[:, changepoint] = changepoint_mean
        data_sufficient_stats.data_precision_quadratic_sums[changepoint] = data_precision_quadratic_sum
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_data_sufficient_statistics!(data_sufficient_stats::MultinomialSufficientStatistics,
                                            changepoint::Int64, data::Matrix{T}, 
                                            ::ChangepointSufficientStatistics, 
                                            ::Changepoint; update_type="add") where {T <: Real}
    counts = sum(data, dims=2)
    if update_type === "new"
        if any(counts .> 0)
            data_sufficient_stats.total_counts[:, changepoint] = counts
        end
    elseif update_type === "add"
        data_sufficient_stats.total_counts[:, changepoint] += counts
    elseif update_type === "remove"
        data_sufficient_stats.total_counts[:, changepoint] -= counts
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_cluster_sufficient_statistics!(cluster_sufficient_stats::MixtureSufficientStatistics, cluster::Int64; 
                                               update_type::String="add")
    if update_type === "add"
        cluster_sufficient_stats.num_observations[cluster] += 1
        if cluster_sufficient_stats.num_observations[cluster] === 1
            cluster_sufficient_stats.clusters[cluster] = true
        end
    elseif update_type === "remove"
        cluster_sufficient_stats.num_observations[cluster] -= 1
        if cluster_sufficient_stats.num_observations[cluster] === 0
            cluster_sufficient_stats.clusters[cluster] = false
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_changepoint_sufficient_statistics!(changepoint_sufficient_stats::ChangepointSufficientStatistics,
                                                   changepoint::Int64; update_type::String="remove")
    if update_type === "remove"
        changepoint_sufficient_stats.num_observations[changepoint] -= 1
        if changepoint_sufficient_stats.num_observations[changepoint] === 0
            changepoint_sufficient_stats.changepoints[changepoint] = false
        end
    elseif update_type === "add"
        changepoint_sufficient_stats.num_observations[changepoint] += 1
        if changepoint_sufficient_stats.num_observations[changepoint] === 1
            changepoint_sufficient_stats.changepoints[changepoint] = true
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_changepoint_sufficient_statistics!(changepoint_sufficient_stats::ChangepointSufficientStatistics,
                                                   changepoint::Int64, changepoint_segment::BitArray;
                                                   update_type::String="add")
    num_observations = count(changepoint_segment)
    if update_type === "new"
        if num_observations > 0
            changepoint_sufficient_stats.num_observations[changepoint] = num_observations
            changepoint_sufficient_stats.changepoints[changepoint] = true
        end
    elseif update_type === "add"
        @assertion changepoint_sufficient_stats.num_observations[changepoint] !== 0
        changepoint_sufficient_stats.num_observations[changepoint] += num_observations
    elseif update_type === "remove"
        changepoint_sufficient_stats.num_observations[changepoint] -= num_observations
        if changepoint_sufficient_stats.num_observations[changepoint] === 0
            changepoint_sufficient_stats.changepoints[changepoint] = false
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_cluster_sufficient_statistics!(cluster_sufficient_stats::HmmSufficientStatistics, cluster::Int64, 
                                               prev_cluster::Int64; update_type::String="add")
    if update_type === "add"
        cluster_sufficient_stats.num_observations[prev_cluster, cluster] += 1
        if sum(cluster_sufficient_stats.num_observations[:, cluster]) === 1
            cluster_sufficient_stats.clusters[cluster] = true
        end
    elseif update_type === "remove"
        cluster_sufficient_stats.num_observations[prev_cluster, cluster] -= 1
        if sum(cluster_sufficient_stats.num_observations[:, cluster]) === 0
            cluster_sufficient_stats.clusters[cluster] = false
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_cluster_stats_new_birth_time!(num_observations::Vector{Int64}, new_time::Int64, old_time::Int64)
    num_observations[new_time] = num_observations[old_time]
    num_observations[old_time] = 0
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::ChangepointSufficientStatistics,
                                              new_time::Int64, old_time::Int64)
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats.num_observations, new_time, old_time)
    cluster_sufficient_stats.changepoints[new_time] = true
    cluster_sufficient_stats.changepoints[old_time] = false
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::MixtureSufficientStatistics,
                                              new_time::Int64, old_time::Int64)
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats.num_observations, new_time, old_time)
    cluster_sufficient_stats.clusters[new_time] = true
    cluster_sufficient_stats.clusters[old_time] = false
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::HmmSufficientStatistics,
                                              new_time::Int64, old_time::Int64)
    clusters = get_clusters(cluster_sufficient_stats)
    for cluster = clusters
        num_observations = cluster_sufficient_stats.num_observation[cluster, :]
        update_cluster_stats_new_birth_time!(num_observations, new_time, old_time)
        cluster_sufficient_stats_array.num_observations[cluster, :] = num_observations
    end
    cluster_sufficient_stats.num_observations[new_time, :] = cluster_sufficient_stats.num_observations[old_time, :]
    cluster_sufficient_stats.num_observations[old_time, :] .= 0
    cluster_sufficient_stats.clusters[new_time] = true
    cluster_sufficient_stats.clusters[old_time] = false
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

function update_cluster_birth_time_remove!(cluster::Int64, iteration::Int64, instances::Matrix{Int64}, 
                                           cluster_sufficient_stats::Union{HmmSufficientStatistics, MixtureSufficientStatistics},
                                           data_sufficient_stats::DataSufficientStatistics, 
                                           data_parameters::DataParameters)
    assigned_to_cluster = instances[:, iteration] .=== cluster
    new_cluster_time = findfirst(assigned_to_cluster)
    instances[assigned_to_cluster, iteration] .= new_cluster_time 
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats, new_cluster_time, cluster)
    update_data_stats_new_birth_time!(data_sufficient_stats, new_cluster_time, cluster, data_parameters)
    return new_cluster_time
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             cluster_sufficient_stats::MixtureSufficientStatistics,
                             data_sufficient_stats::DataSufficientStatistics,
                             model::Mixture) where {T <: Real}
    n = size(instances)[1]
    cluster = instances[observation, iteration]

    instances[observation, iteration] = n+1 
    datum = vec(data[:, observation])
    update_cluster_sufficient_statistics!(cluster_sufficient_stats, cluster, update_type="remove")
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, model, 
                                       update_type="remove")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, model.data_parameters)
    if (cluster === observation) && (cluster_sufficient_stats.num_observations[cluster] > 0)
        cluster = update_cluster_birth_time_remove!(cluster, iteration, instances, cluster_sufficient_stats, 
                                                    data_sufficient_stats, model.data_parameters)
    end
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             changepoint_sufficient_stats::ChangepointSufficientStatistics,
                             data_sufficient_stats::DataSufficientStatistics, model::Model) where {T <: Real}
    n = size(instances)[1]
    changepoint = instances[observation, iteration]

    instances[observation, iteration] = n+1
    new_changepoint = observation + 1
    new_changepoint_segment = (instances[:, iteration] .=== changepoint) .& ((1:n) .> observation)
    new_changepoint_data = data[:, new_changepoint_segment]
    instances[new_changepoint_segment, iteration] .= new_changepoint

    # remove a count from changepoint for the removed observation
    update_changepoint_sufficient_statistics!(changepoint_sufficient_stats, changepoint, update_type="remove")
    # remove counts corresponding to the disconnected segment
    update_changepoint_sufficient_statistics!(changepoint_sufficient_stats, changepoint, new_changepoint_segment, 
                                              update_type="remove")
    datum = data[:, observation]
    # first, remove the observation data from the changepoint
    update_data_sufficient_statistics!(data_sufficient_stats, changepoint, datum, changepoint_sufficient_stats, model, 
                                       update_type="remove")
    # then, remove the data from the now separated latter half of the segment
    update_data_sufficient_statistics!(data_sufficient_stats, changepoint, new_changepoint_data, 
                                       changepoint_sufficient_stats, model, update_type="remove")
    # recompute the posterior parameters for the changepoints
    compute_data_posterior_parameters!(changepoint, changepoint_sufficient_stats, data_sufficient_stats, 
                                       model.data_parameters)
    # add counts for the new changepoint segment
    if new_changepoint <= n
        update_changepoint_sufficient_statistics!(changepoint_sufficient_stats, new_changepoint, new_changepoint_segment, 
                                                  update_type="new")
        # add sufficient stats for the new segment
        update_data_sufficient_statistics!(data_sufficient_stats, new_changepoint, new_changepoint_data, 
                                           changepoint_sufficient_stats, model, update_type="new")
        # recompute the posterior parameters for the changepoints
        compute_data_posterior_parameters!(new_changepoint, changepoint_sufficient_stats, data_sufficient_stats, 
                                           model.data_parameters)
    end
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             cluster_sufficient_stats::HmmSufficientStatistics,
                             data_sufficient_stats::DataSufficientStatistics,
                             data_parameters::DataParameters) where {T <: Real}
    n = size(instances)[1]
    if observation > 1
        prev_cluster = instances[observation-1, iteration]
    else
        prev_cluster = n+1
    end
    cluster = instances[observation, iteration]
    # n+1 indicates that the observation is assigned to no cluster, or the "null" cluster
    null_cluster = n+1
    instances[observation, iteration] = null_cluster
    datum = vec(data[:, observation])
    update_cluster_sufficient_statistics!(cluster_sufficient_stats, cluster, prev_cluster, update_type="remove")
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, data_parameters, 
                                       update_type="remove")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, data_parameters)
    if cluster === observation && sum(cluster_sufficient_stats.num_observations[:, cluster]) > 0
        cluster = update_cluster_birth_time_remove!(cluster, iteration, instances, cluster_sufficient_stats, 
                                                    data_sufficient_stats, data_parameters)
    end
    # Update book keeping for the next observation over
    if observation < n
        next_observation = observation + 1
        next_cluster = instances[next_observation, iteration]
        update_cluster_sufficient_statistics!(cluster_sufficient_stats, next_cluster, cluster, update_type="remove")
        update_cluster_sufficient_statistics!(cluster_sufficient_stats, next_cluster, null_cluster, update_type="add")
    end
end

function update_cluster_birth_time_add!(cluster::Int64, observation::Int64, iteration::Int64, instances::Matrix{Int64}, 
                                        cluster_sufficient_stats::ClusterSufficientStatistics,
                                        data_sufficient_stats::DataSufficientStatistics, 
                                        data_parameters::DataParameters)
    assigned_to_cluster = (instances[:, iteration] .=== cluster)
    old_cluster_time = cluster
    cluster = observation
    instances[assigned_to_cluster, iteration] .= cluster
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats, cluster, old_cluster_time)
    update_data_stats_new_birth_time!(data_sufficient_stats, cluster, old_cluster_time, data_parameters)
    return cluster
end

function add_observation!(observation::Int64, cluster::Int64, instances::Matrix{Int64}, iteration::Int64, 
                          data::Matrix{T}, cluster_sufficient_stats::MixtureSufficientStatistics,
                          data_sufficient_stats::DataSufficientStatistics, 
                          model::Mixture) where {T <: Real}
    if observation < cluster
        cluster = update_cluster_birth_time_add!(cluster, observation, iteration, instances, cluster_sufficient_stats,
                                                 data_sufficient_stats, model.data_parameters)
    end
    instances[observation, iteration] = cluster
    datum = vec(data[:, observation])
    update_cluster_sufficient_statistics!(cluster_sufficient_stats, cluster, update_type="add")
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, model, 
                                       update_type="add")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, model.data_parameters)
end

function add_observation!(observation::Int64, changepoint::Int64, instances::Matrix{Int64}, iteration::Int64,
                          data::Matrix{T}, changepoint_sufficient_stats::ChangepointSufficientStatistics,
                          data_sufficient_stats::DataSufficientStatistics, model::Changepoint) where {T <: Real}
    if observation < changepoint
        @assertion sum(changepoint_sufficient_stats.num_observations[observation:(changepoint-1)]) === 0
        changepoint = update_cluster_birth_time_add!(changepoint, observation, iteration, instances, 
                                                     changepoint_sufficient_stats, data_sufficient_stats, 
                                                     model.data_parameters)
    else
        @assertion sum(changepoint_sufficient_stats.num_observations[(changepoint+1):observation]) === 0
    end
    instances[observation, iteration] = changepoint
    update_changepoint_sufficient_statistics!(changepoint_sufficient_stats, changepoint, update_type="add")
    datum = vec(data[:, observation])
    update_data_sufficient_statistics!(data_sufficient_stats, changepoint, datum, changepoint_sufficient_stats,
                                       model, update_type="add")
    compute_data_posterior_parameters!(changepoint, changepoint_sufficient_stats, data_sufficient_stats, 
                                       model.data_parameters)
end

function add_observation!(observation::Int64, changepoints::Vector{Int64}, instances::Matrix{Int64}, iteration::Int64,
                          data::Matrix{T}, changepoint_sufficient_stats::ChangepointSufficientStatistics,
                          data_sufficient_stats::DataSufficientStatistics, model::Changepoint) where {T <: Real}
    @assertion length(changepoints) === 2

    first_changepoint = minimum(changepoints)
    last_changepoint = maximum(changepoints)

    @assertion sum(changepoint_sufficient_stats.num_observation[first_changepoint+1:last_changepoint-1]) === 0
    @assertion observation in first_changepoint+1:last_changepoint-1

    assigned_to_last_changepoint = (instances[:, iteration] === last_changepoint)
    instances[observation, iteration] = first_changepoint
    instances[assigned_to_last_changepoint, iteration] = first_changepoint
    # add a count for the new observation
    update_changepoint_sufficient_statistics!(changepoint_sufficient_stats, first_changepoint, update_type="add")
    # add counts for the merged changepoint
    update_changepoint_sufficient_statistics!(changepoint_sufficient_stats, first_changepoint, 
                                              assigned_to_last_changepoint, update_type="add")
    datum = vec(data[:, observation])
    last_changepoint_data = data[assigned_to_last_changepoint, observation]
    # update data sufficient statistics for the new observation
    update_data_sufficient_statistics!(data_sufficient_stats, first_changepoint, datum, changepoint_sufficient_stats,
                                       model, update_type="add")
    # update data sufficient statistics for the merged changepoint                                 
    update_data_sufficient_statistics!(data_sufficient_stats, first_changepoint, last_changepoint_data, 
                                       changepoint_sufficient_stats, model, update_type="add")
    # update data sufficient statistics for the first changepoint
    compute_data_posterior_parameters!(first_changepoint, changepoint_sufficient_stats, data_sufficient_stats, 
                                       model.data_parameters)
end

function add_observation!(observation::Int64, cluster::Int64, instances::Matrix{Int64}, iteration::Int64, 
                          data::Matrix{T}, cluster_sufficient_stats::HmmSufficientStatistics,
                          data_sufficient_stats::DataSufficientStatistics, 
                          data_parameters::DataParameters) where {T <: Real}
    if observation < cluster
        cluster = update_cluster_birth_time_add!(cluster, observation, iteration, instances, cluster_sufficient_stats,
                                                 data_sufficient_stats, data_parameters)
    end
    n = size(instances)[1]
    if observation > 1
        prev_cluster = instances[observation-1, iteration]
    else
        prev_cluster = n+1 
    end
    instances[observation, iteration] = cluster
    datum = vec(data[:, observation])
    update_cluster_sufficient_statistics!(cluster_sufficient_stats, cluster, prev_cluster, update_type="add")
    update_data_sufficient_statistics!(data_sufficient_stats, cluster, datum, cluster_sufficient_stats, data_parameters, 
                                       update_type="add")
    compute_data_posterior_parameters!(cluster, cluster_sufficient_stats, data_sufficient_stats, data_parameters)
    # Update book keeping of the next observation over
    if observation < n
        next_observation = observation + 1
        next_cluster = instances[next_observation, iteration]
        null_cluster = n+1
        update_cluster_sufficient_statistics!(cluster_sufficient_stats, next_cluster, null_cluster, update_type="remove")
        update_cluster_sufficient_statistics!(cluster_sufficient_stats, next_cluster, cluster, update_type="add")
    end
end

function compute_arrival_distribution_posterior(cluster_sufficient_stats::ClusterSufficientStatistics,
                                                cluster_parameters::ParametricArrivalsClusterParameters{T}) where {T <: GeometricArrivals}
    n = length(cluster_sufficient_stats.clusters)
    num_clusters = length(get_clusters(cluster_sufficient_stats))
    phi_posterior = deepcopy(cluster_parameters.arrival_distribution.prior)
    phi_posterior[1] += num_clusters - 1
    phi_posterior[2] += n - num_clusters 
    return phi_posterior
end

function compute_arrival_distribution_posterior(changepoint_sufficient_stats::ChangepointSufficientStatistics,
                                                changepoint_parameters::ParametricArrivalsClusterParameters{T}) where {T <: GeometricArrivals}
    n = length(changepoint_sufficient_stats.changepoints)
    num_changepoints = length(get_changepoints(changepoint_sufficient_stats))
    phi_posterior = deepcopy(changepoint_parameters.arrival_distribution.prior)
    phi_posterior[1] += num_changepoints - 1
    phi_posterior[2] += n - num_changepoints 
    return phi_posterior
end

function new_cluster_log_predictive(::Union{MixtureSufficientStatistics, ChangepointSufficientStatistics}, 
                                    cluster_parameters::DpParameters)
    return log(cluster_parameters.alpha)
end

function existing_cluster_log_predictive(cluster_sufficient_stats::ChangepointSufficientStatistics,
                                         cluster_parameters::DpParameters, changepoint::Int64)
    return log(cluster_sufficient_stats.num_observations[changepoint] - 1 + cluster_parameters.beta)
end

function new_cluster_log_predictive(cluster_sufficient_stats::Union{MixtureSufficientStatistics, ChangepointSufficientStatistics},
                                    cluster_parameters::ParametricArrivalsClusterParameters{T}) where {T <: GeometricArrivals} 
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return log(phi_posterior[1]) - log(sum(phi_posterior))
end

function existing_cluster_log_predictive(cluster_sufficient_stats::ChangepointSufficientStatistics,
                                         cluster_parameters::ParametricArrivalsClusterParameters{T}, ::Int64) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return log(phi_posterior[2]) - log(sum(phi_posterior))
end

function existing_cluster_log_predictive(cluster_sufficient_stats::MixtureSufficientStatistics,
                                         cluster_parameters::ParametricArrivalsClusterParameters{T}) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(cluster_sufficient_stats, cluster_parameters)
    return log(phi_posterior[2]) - log(sum(phi_posterior))
end

function new_cluster_log_predictive(cluster_sufficient_stats::MixtureSufficientStatistics,
                                    cluster_parameters::ParametricArrivalsClusterParameters{T}) where {T <: PitmanYorArrivals}
    n = length(cluster_sufficient_stats.clusters)
    num_clusters = count(cluster_sufficient_stats.clusters)
    tau = cluster_parameters.arrival_distribution.tau
    theta = cluster_parameters.arrival_distribution.theta
    return log(theta + num_clusters*tau) - log(n + theta)
end

function existing_cluster_log_predictive(cluster_sufficient_stats::MixtureSufficientStatistics,
                                         cluster_parameters::ParametricArrivalsClusterParameters{T}) where {T <: PitmanYorArrivals}
    n = length(cluster_sufficient_stats.clusters)
    num_clusters = count(cluster_sufficient_stats.clusters)
    tau = cluster_parameters.arrival_distribution.tau
    theta = cluster_parameters.arrival_distribution.theta
    return log(n - num_clusters*tau) - log(n + theta)
end

# TODO: make a version of this for HMMs
function compute_num_complement(cluster::Int64, cluster_num_observations::Vector{Int64}; 
                                missing_observation::Union{Int64, Nothing}=nothing)
    younger_clusters = 1:(cluster-1)
    sum_of_younger_clusters = sum(cluster_num_observations[younger_clusters])
    num_complement = sum(cluster_num_observations[younger_clusters]) - (cluster - 1)
    if missing_observation !== nothing && missing_observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_stick_breaking_posterior(cluster::Int64, num_observations::Vector{Int64}, beta_prior::Vector{T};
                                          missing_observation::Union{Int64, Nothing}=nothing) where {T <: Real}
    posterior = deepcopy(beta_prior)
    posterior[1] += num_observations[cluster] - 1
    posterior[2] += compute_num_complement(cluster, num_observations, missing_observation=missing_observation)
    return posterior
end

function compute_stick_breaking_posterior(cluster::Int64, cluster_sufficient_stats::MixtureSufficientStatistics, 
                                          ntl_parameters::ParametricArrivalsClusterParameters; 
                                          missing_observation::Union{Int64, Nothing}=nothing)
    posterior = compute_stick_breaking_posterior(cluster, cluster_sufficient_stats.num_observations, 
                                                 ntl_parameters.prior, missing_observation=missing_observation)
    return posterior
end

function compute_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, 
                                         ntl_sufficient_stats::MixtureSufficientStatistics, ntl_parameters::NtlParameters)
    # Not strictly the number of observations
    num_clusters = length(clusters)
    log_weights = Vector{Float64}(undef, num_clusters)
    n = maximum(clusters)
    n = maximum([n, observation])
    cluster_log_weights = zeros(Float64, n)
    complement_log_weights = zeros(Float64, n)
    psi_parameters = Array{Int64}(undef, 2, n)
    logbetas = Array{Float64}(undef, n)
    new_num_complement = compute_num_complement(observation, ntl_sufficient_stats.num_observations, 
                                                missing_observation=observation)
    younger_cluster_new_psi = Array{Float64}(undef, 2)
    cluster_new_psi = Array{Float64}(undef, 2)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_stick_breaking_posterior(cluster, ntl_sufficient_stats, ntl_parameters, 
                                                           missing_observation=observation)
            psi_parameters[:, cluster] = cluster_psi
            log_denom = log(sum(cluster_psi))
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
    return log_weights
end

function compute_existing_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, 
                                                  cluster_sufficient_stats::MixtureSufficientStatistics,
                                                  cluster_parameters::ParametricArrivalsClusterParameters)
    log_weights = compute_cluster_log_predictives(observation, clusters, cluster_sufficient_stats, cluster_parameters)
    log_weights .+= existing_cluster_log_predictive(cluster_sufficient_stats, cluster_parameters)
    return log_weights
end

function compute_existing_cluster_log_predictives(::Int64, clusters::Vector{Int64},
                                                  cluster_sufficient_stats::MixtureSufficientStatistics, ::DpParameters)
    log_weights = log.(cluster_sufficient_stats.num_observations[clusters])
    return log_weights
end

function compute_existing_cluster_log_predictives(::Int64, clusters::Vector{Int64},
                                                  cluster_sufficient_stats::MixtureSufficientStatistics, 
                                                  beta_ntl_parameters::BetaNtlParameters)
    n = length(cluster_sufficient_stats.clusters)
    num_clusters = count(cluster_sufficient_stats.clusters)
    log_weights = log.(cluster_sufficient_stats.num_observations[clusters] .- beta_ntl_parameters.alpha)
    log_weights .-= log(n - num_clusters*beta_ntl_parameters.alpha)
    log_weights .+= existing_cluster_log_predictive(cluster_sufficient_stats, beta_ntl_parameters)
    return log_weights
end

function compute_data_posterior_parameters!(cluster::Int64, cluster_sufficient_stats::ClusterSufficientStatistics,
                                            data_sufficient_stats::GaussianSufficientStatistics,
                                            data_parameters::GaussianParameters)
    data_mean = vec(data_sufficient_stats.data_means[:, cluster])
    n = cluster_sufficient_stats.num_observations[cluster]
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

function compute_data_posterior_parameters!(cluster::Int64, 
                                            ::Union{MixtureSufficientStatistics, ChangepointSufficientStatistics},
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
    log_predictive = Array{Float64}(undef, length(clusters))
    datum = vec(data[:, observation])
    dim = length(datum)
    for (index, cluster) = enumerate(clusters)
        log_predictive[index] = data_log_predictive(datum, cluster, data_sufficient_stats, data_parameters)
    end
    return log_predictive
end

get_clusters(cluster_sufficient_stats::ClusterSufficientStatistics) = findall(cluster_sufficient_stats.clusters)

get_changepoints(changepoint_suff_stats::ChangepointSufficientStatistics) = findall(changepoint_suff_stats.changepoints)

function get_changepoints(changepoint_sufficient_stats::ChangepointSufficientStatistics, observation::Int64)
    n = length(changepoint_sufficient_stats.num_observations)
    all_changepoints = findall(changepoint_sufficient_stats.num_observations .> 0)
    changepoints = Int64[]
    if observation > 1
        first_changepoint = all_changepoints[findlast(all_changepoints .< observation)]
        append!(changepoints, first_changepoint)
    end
    if observation < n
        last_changepoint = all_changepoints[findfirst(all_changepoints .> observation)]
        append!(changepoints, last_changepoint)
    end
    return Vector{Int64}(changepoints)
end

function gibbs_move(observation::Int64, data::Matrix{T}, cluster_sufficient_stats::MixtureSufficientStatistics, 
                    data_sufficient_stats::DataSufficientStatistics, 
                    model::Union{Mixture, HiddenMarkovModel}) where {T <: Real}
    data_parameters = model.data_parameters
    cluster_parameters = model.cluster_parameters
    clusters = get_clusters(cluster_sufficient_stats)
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

function gibbs_move(observation::Int64, data::Matrix{T}, changepoint_sufficient_stats::ChangepointSufficientStatistics,
                    data_sufficient_stats::DataSufficientStatistics, model::Changepoint) where {T <: Real}
    data_parameters = model.data_parameters
    changepoint_parameters = model.changepoint_parameters
    n = size(data)[2]

    changepoints = get_changepoints(changepoint_sufficient_stats, observation)
    num_changepoints = length(changepoints)

    choices = Array{Int64}(undef, num_changepoints+1)
    choices[1:num_changepoints] = changepoints
    choices[end] = observation

    weights = Array{Float64}(undef, num_changepoints+1)
    for (index, changepoint) in enumerate(changepoints)
        weights[index] = existing_cluster_log_predictive(changepoint_sufficient_stats, model.changepoint_parameters, 
                                                         changepoint)
    end
    weights[num_changepoints+1] = new_cluster_log_predictive(changepoint_sufficient_stats, model.changepoint_parameters)
    weights += compute_data_log_predictives(observation, choices, data, data_sufficient_stats, data_parameters)

    return gumbel_max(choices, weights)
end

function propose(observation::Int64, ::Int64, data::Matrix{T}, cluster_sufficient_stats::MixtureSufficientStatistics,
                 data_sufficient_stats::DataSufficientStatistics, model::Model) where {T <: Real}
    return gibbs_move(observation, data, cluster_sufficient_stats, data_sufficient_stats, model)    
end

function propose(observation::Int64, prev_cluster::Int64, data::Matrix{T}, 
                 cluster_sufficient_stats::HmmSufficientStatistics, data_sufficient_stats::DataSufficientStatistics, 
                 model::Model) where {T <: Real}
    n = length(data)[1]
    num_observations = cluster_sufficient_stats.num_observations[prev_cluster, :]
    clusters = cluster_sufficient_stats.clusters
    state_cluster_sufficient_stats = prepare_cluster_sufficient_statistics(model.cluster_parameters, num_observations, 
                                                                           clusters)
    return gibbs_move(observation, data, state_cluster_sufficient_stats, data_sufficient_stats, model)
end

end