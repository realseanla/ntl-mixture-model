module NtlFit

using Distributions
using LinearAlgebra
using SpecialFunctions
using Statistics

function fit(data::Matrix, num_instances::Int64, method::String)
    n = size(data)[2]
    dim = size(data)[1]
    instances = Array{Int64}(undef, n, num_instances)
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
    if method === "gibbs"
        gibbs_sample!(instances, data, cluster_num_observations, cluster_means, posterior_means, 
                      posterior_covs)
    else
        message = "$method is not an appropriate fit method."
        throw(ArgumentError(message))
    end
    return instances
end

function compute_log_weight!(log_weights::Vector{Float64}, log_likelihoods::Vector{Float64}, proposal_weight::Float64, 
                             assignment::Vector{Int64}, data::Matrix{Float64}, particle::Int64)
    previous_log_weight = log_weights[particle]
    previous_log_likelihood = log_likelihoods[particle]
    log_likelihood = compute_gaussian_joint_log_likelihood(assignment, data)
    log_weights[particle] = previous_log_weight + log_likelihood - previous_log_likelihood - log(proposal_weight)
end

function smc!(particles::Matrix{Int64}, data::Matrix{Float64}, cluster_num_observations::Vector{Int64}, 
              cluster_means::Matrix{Float64}, posterior_means::Matrix{Float64}, posterior_covs::Matrix{Float64})
    n = size(data)[2]
    particles .= n+1
    num_particles = size(particles)[2]
    log_likelihoods = zeros(Float64, num_particles)
    log_weights = Array{Float64}(undef, num_particles)
    for observation = 1:n
        for particle = 1:num_particles
            assignment = vec(particles[:, particle])
            (cluster, proposal_weight) = gibbs_move(observation, data, assignment, cluster_num_observations, 
                                                    posterior_means, posterior_covs)
            add_observation!(observation, cluster, particles, particle, data, cluster_num_observations, cluster_means,
                             posterior_means, posterior_covs)
            assignment = vec(particles[:, i])
            compute_log_weight!(log_weights, log_likelihoods, proposal_weight, assignment, data, particle)
        end
        weights = exp.(log_weights)
        normalized_weights ./= sum(weights)
        ess = 1/sum(normalized_weights.^2)
        if ess < num_particles/2
            resampled_indices = rand(Categorical(normalized_weights), num_particles)
            particles = particles[:, resampled_indices]
            log_weights = log_weights[resampled_indices]
            log_likelihoods = log_likelihoods[resampled_indices]
        end
    end
end

function gibbs_sample!(instances::Matrix{Int64}, data::Matrix{Float64}, cluster_num_observations::Vector{Int64}, 
                       cluster_means::Matrix{Float64}, posterior_means::Matrix{Float64}, 
                       posterior_covs::Matrix{Float64})
    iterations = size(instances)[2]
    for iteration = 2:iterations
        instances[:, iteration] = instances[:, iteration-1]
        for observation = 1:size(instances)[1]
            remove_observation!(observation, instances, iteration, data, cluster_num_observations, cluster_means, 
                                posterior_means, posterior_covs)
            assignment = vec(instances[:, iteration])
            (cluster, weight) = gibbs_move(observation, data, assignment, cluster_num_observations, posterior_means, 
                                           posterior_covs)
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
    num_complement = sum(cluster_num_observations[younger_clusters]) - (cluster - 1)
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

function gibbs_move(observation::Int64, data::Matrix{Float64}, assignment::Vector{Int64},
                    cluster_num_observations::Vector{Int64}, posterior_means::Matrix{Float64}, 
                    posterior_covs::Matrix{Float64})
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

end