module Evaluate
    using ..Fitter: prepare_sufficient_statistics, update_cluster_sufficient_statistics!, update_data_sufficient_statistics!
    using ..Fitter: compute_data_posterior_parameters!, get_clusters, compute_existing_cluster_log_predictives, new_cluster_log_predictive
    using ..Fitter: compute_data_log_predictives, prepare_auxillary_variables
    using ..Utils: compute_normalized_weights

    using Statistics

    function update_sufficient_statistics!(sufficient_stats, datum, cluster, model; update_type="add")
        update_cluster_sufficient_statistics!(sufficient_stats, cluster, update_type=update_type)
        update_data_sufficient_statistics!(sufficient_stats, cluster, datum, model, update_type=update_type)
        compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
    end

    function evaluate(test_datum::Vector{T}, posterior, training_data, model) where {T <: Real}
        data_parameters = model.data_parameters
        cluster_parameters = model.cluster_parameters
        num_iterations = size(posterior)[2]
        num_observations = size(posterior)[1]
        test_observation = num_observations+1
        predictive_probabilities = Array{Float64}(undef, num_iterations)
        aux_variables = prepare_auxillary_variables(model, training_data)
        for iteration = 1:num_iterations
            sufficient_stats = prepare_sufficient_statistics(model, training_data) 
            for observation = 1:num_observations
                datum = training_data[:, observation]
                cluster = posterior[observation, iteration]
                update_sufficient_statistics!(sufficient_stats, datum, cluster, model, update_type="add")
            end
            clusters = get_clusters(sufficient_stats.cluster) 
            num_clusters = length(clusters)
            choices = Array{Int64}(undef, num_clusters+1)
            choices[1:num_clusters] = clusters
            choices[num_clusters+1] = test_observation

            cluster_log_weights = Array{Float64}(undef, num_clusters + 1)
            cluster_log_weights[1:num_clusters] = compute_existing_cluster_log_predictives(test_observation, clusters, 
                                                                                           sufficient_stats, cluster_parameters,
                                                                                           aux_variables)
            cluster_log_weights[num_clusters+1] = new_cluster_log_predictive(sufficient_stats, cluster_parameters, aux_variables,
                                                                             test_observation)
            cluster_log_probabilities = log.(compute_normalized_weights(cluster_log_weights))

            data_log_predictives = compute_data_log_predictives(test_datum, choices, sufficient_stats.data, data_parameters)
            log_predictive_weights = cluster_log_probabilities + data_log_predictives
            predictive_probability = sum(exp.(log_predictive_weights))
            predictive_probabilities[iteration] = predictive_probability
        end
        return log(mean(predictive_probabilities))
    end
end