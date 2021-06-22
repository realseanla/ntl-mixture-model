module Evaluate
    using ..Models: Mixture
    using ..Fitter: prepare_sufficient_statistics, update_cluster_sufficient_statistics!, update_data_sufficient_statistics!
    using ..Fitter: compute_data_posterior_parameters!, get_clusters, compute_existing_cluster_log_predictives, new_cluster_log_predictive
    using ..Fitter: compute_data_log_predictives, prepare_auxillary_variables, gibbs_sample!
    using ..Utils: compute_normalized_weights, generate_all_clusterings
    import ..Fitter: compute_joint_log_likelihood

    using Statistics

    function update_sufficient_statistics!(sufficient_stats, observation, cluster, model, data, aux_var; update_type="add")
        update_cluster_sufficient_statistics!(sufficient_stats, cluster, update_type=update_type)
        update_data_sufficient_statistics!(sufficient_stats, cluster, observation, model, data, aux_var, update_type=update_type)
        compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
    end

    function compute_joint_log_likelihood(clustering::Vector{Int64}, data::Matrix{T}, model::Mixture) where {T <: Real}
        sufficient_stats = prepare_sufficient_statistics(model, data)
        aux_variables = prepare_auxillary_variables(model, data)

        for (observation, cluster) in enumerate(clustering) 
            update_sufficient_statistics!(sufficient_stats, observation, cluster, model, data, aux_variables)
        end
        log_likelihood = compute_joint_log_likelihood(sufficient_stats, model, aux_variables)
        return log_likelihood
    end

    function compute_joint_log_likelihood(clustering::Vector{Vector{Int64}}, data::Matrix{T}, model::Mixture) where {T <: Real}
        sufficient_stats = prepare_sufficient_statistics(model, data)
        aux_variables = prepare_auxillary_variables(model, data)

        for cluster in clustering 
            cluster_id = minimum(cluster)
            for observation in cluster
                update_sufficient_statistics!(sufficient_stats, observation, cluster_id, model, data, aux_variables)
            end
        end
        log_likelihood = compute_joint_log_likelihood(sufficient_stats, model, aux_variables)
        return log_likelihood
    end

    function compute_all_clustering_log_likelihoods(n, data::Matrix{T}, model::Mixture) where {T <: Real}
        all_clusterings = generate_all_clusterings(n)
        clustering_log_probabilities = Vector{Float64}(undef, length(all_clusterings))
        for (i, clustering) = enumerate(all_clusterings)
            clustering_log_probabilities[i] = compute_joint_log_likelihood(clustering, data, model)
        end
        return clustering_log_probabilities
    end

    function compute_posterior_probability(clustering::Vector{Int64}, data::Matrix{T}, model::Mixture) where {T <: Real}
        n = length(clustering)

        all_clusterings_log_probs = compute_all_clustering_log_likelihoods(n, data, model)
        denominator = sum(exp.(all_clusterings_log_probs))

        log_numerator = compute_joint_log_likelihood(clustering, data, model)

        log_probability = log_numerator - log(denominator)
        return exp(log_probability) 
    end

    function compute_posterior_probability(clustering::Vector{Vector{Int64}}, data::Matrix{T}, model::Mixture) where {T <: Real}
        n = size(data)[2]
        clustering_vector = Vector{Int64}(undef, n)
        for cluster in clustering 
            cluster_id = minimum(cluster)
            for observation in cluster 
                clustering_vector[observation] = cluster_id
            end
        end
        return compute_posterior_probability(clustering_vector, data, model)
    end


    function compute_mcmc_se(clustering::Vector{Int64}, markov_chain::Matrix{Int64}) 
        chain_length = size(markov_chain)[2]
        batch_length = floor(Int64, sqrt(chain_length))
        num_batches = ceil(Int64, sqrt(chain_length))

        posterior_log_probability = compute_posterior_probability(clustering, markov_chain)

        batch_probabilities = Vector{Float64}(undef, num_batches)
        for batch = 1:num_batches
            start_index = batch_length*(batch-1) + 1
            stop_index = batch_length*batch
            batch_probabilities[batch] = compute_posterior_probability(clustering, markov_chain[:, start_index:stop_index])
        end
        posterior_probability = exp(posterior_log_probability)
        sigma_squared = (batch_length/(num_batches-1))*sum((batch_probabilities .- posterior_log_probability).^2)
        mcmc_se = sqrt(sigma_squared)/sqrt(chain_length)
        return mcmc_se 
    end

    function compute_mcmc_se(clustering::Vector{Vector{Int64}}, markov_chain::Matrix{Int64}) 
        n = size(markov_chain)[1]
        clustering_vector = Vector{Int64}(undef, n)
        for cluster in clustering 
            cluster_id = minimum(cluster)
            for observation in cluster 
                clustering_vector[observation] = cluster_id
            end
        end
        return compute_mcmc_se(clustering_vector, markov_chain)
    end

    function compute_posterior_probability(clustering::Vector{Int64}, markov_chain::Matrix{Int64}) 
        chain_length = size(markov_chain)[2]
        num_iterations_equal = count(vec(mapslices(z->(clustering == vec(z)), markov_chain, dims=1)))
        posterior_probability = num_iterations_equal/chain_length
        return posterior_probability
    end

    function compute_posterior_probability(clustering::Vector{Vector{Int64}}, markov_chain::Matrix{Int64}) 
        n = size(markov_chain)[1]
        clustering_vector = Vector{Int64}(undef, n)
        for cluster in clustering 
            cluster_id = minimum(cluster)
            for observation in cluster 
                clustering_vector[observation] = cluster_id
            end
        end
        return compute_posterior_probability(clustering_vector, markov_chain)
    end

    function compute_posterior_probability_confidence_interval(clustering, markov_chain::Matrix{Int64})
        posterior_probability = compute_posterior_probability(clustering, markov_chain)
        mcmc_se = compute_mcmc_se(clustering, markov_chain) 
        return [posterior_probability - 1.96mcmc_se, posterior_probability + 1.96mcmc_se]
    end

end