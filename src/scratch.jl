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
