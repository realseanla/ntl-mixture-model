module Fitter

using ..Models: DataParameters, ClusterParameters, GaussianParameters, NtlParameters, DpParameters
using ..Models: FiniteTopicModelParameters, FiniteTopicModelSufficientStatistics
using ..Models: DataSufficientStatistics, MixtureSufficientStatistics, GaussianSufficientStatistics
using ..Models: MultinomialParameters, MultinomialSufficientStatistics, GeometricArrivals
using ..Models: HmmSufficientStatistics, StationaryHmmSufficientStatistics, NonstationaryHmmSufficientStatistics
using ..Models: Model, Mixture, ClusterSufficientStatistics
using ..Models: ParametricArrivalsClusterParameters, BetaNtlParameters
using ..Models: PitmanYorArrivals, SufficientStatistics 
using ..Models: ArrivalDistribution, AuxillaryVariables, ArrivalsAuxillaryVariables
using ..Models: GaussianAuxillaryVariables, MultinomialAuxillaryVariables
using ..Models: FiniteTopicModelAuxillaryVariables, GeometricAuxillaryVariables
using ..Models: DpAuxillaryVariables
using ..Samplers: Sampler, GibbsSampler, SequentialMonteCarlo, SequentialImportanceSampler
using ..Samplers: MetropolisWithinGibbsSampler, MetropolisHastingsSampler
using ..Utils: logbeta, logmvbeta, log_multinomial_coeff, gumbel_max, isnothing, compute_normalized_weights
using ..Utils: effective_sample_size, hist
using ..Utils: mvt_logpdf, logfactorial
using ..Utils: hasproperty

using Distributions
using LinearAlgebra
using Statistics
using ProgressBars
using ProgressMeter
using SparseArrays
using SpecialFunctions
using DataStructures

asserting() = true # when set to true, this will enable all `@assertion`s

macro assertion(test)
    esc(:(if $(@__MODULE__).asserting()
        @assert($test)
    end))
end

function random_initial_assignment!(auxillary_variables::AuxillaryVariables{A, D}, sufficient_stats::SufficientStatistics,
                                    cluster_assignments::Vector{Int64}, ::Model, data) where
                                    {A, D <: Union{GaussianAuxillaryVariables, MultinomialAuxillaryVariables}}
    nothing
end

function random_initial_assignment!(auxillary_variables::AuxillaryVariables{A, D}, sufficient_stats::SufficientStatistics,
                                    cluster_assignments::Vector{Int64}, model::Model, data) where {A, D <: FiniteTopicModelAuxillaryVariables}
    num_documents = length(cluster_assignments)
    num_tokens = size(data)[1]
    num_topics = model.data_parameters.num_topics
    for document = 1:num_documents
        cluster = cluster_assignments[document]
        for token = 1:num_tokens
            num_instances = data[token, document]
            for instance = 1:num_instances
                new_topic = rand(1:num_topics)
                auxillary_variables.data.token_topic_assignments[document][token][instance] = new_topic
                # update sufficient statistics
                update_token_topic_statistics!(document, token, new_topic, cluster, auxillary_variables, sufficient_stats,
                                               update_type="add")
            end
        end
    end
end

function random_initial_assignment!(instances::Matrix, data::Matrix, sufficient_stats::SufficientStatistics{C, D}, 
                                    model::Mixture, auxillary_variables::AuxillaryVariables{A, F},
                                    sampler::MetropolisWithinGibbsSampler) where {C, D, A, F}
    n = sufficient_stats.n
    iteration = 1
    cluster = 1
    clusters = []
    for observation = 1:n
        coin_flip = reshape(rand(Binomial(1, 0.1), 1), 1)[1]
        if coin_flip === 1 || length(clusters) === 0
            cluster = observation
            append!(clusters, cluster)
        else
            cluster = rand(clusters)
        end
        add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
    random_initial_assignment!(auxillary_variables, sufficient_stats, vec(instances[:, iteration]), model, data)
end

function random_initial_assignment!(instances::Matrix, data::Matrix, sufficient_stats::SufficientStatistics{C, D}, 
                                    model::Mixture, auxillary_variables::AuxillaryVariables{A, F},
                                    sampler::MetropolisHastingsSampler) where {C, D, A, F}
    n = sufficient_stats.n
    iteration = 1
    cluster = 1
    clusters = []
    for observation = 1:n
        coin_flip = reshape(rand(Binomial(1, 0.1), 1), 1)[1]
        if coin_flip === 1 || length(clusters) === 0
            cluster = observation
            append!(clusters, cluster)
        else
            clusters_to_sample = clusters[clusters .< observation]
            clusters_to_sample = clusters_to_sample[maximum([1, end - sampler.proposal_radius]):end]
            cluster = rand(clusters_to_sample)
        end
        add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
    random_initial_assignment!(auxillary_variables, sufficient_stats, vec(instances[:, iteration]), model, data)
end

function deterministic_initial_assignment!(instances, data, sufficient_stats, model::Mixture, auxillary_variables, 
                                           sampler::MetropolisWithinGibbsSampler, chain)
    n = sufficient_stats.n
    iteration = 1
    cluster = 1
    for observation = 1:n
        if sampler.assignment_types[chain] === "all same cluster"
            cluster = 1
        elseif sampler.assignment_types[chain] === "all different clusters"
            cluster = observation
        else
            error("Error in assignment type")
        end
        add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
    random_initial_assignment!(auxillary_variables, sufficient_stats, vec(instances[:, iteration]), model, data)
end

function fit(data::Matrix{T}, model, sampler::MetropolisWithinGibbsSampler) where {T <: Real}
    if hasproperty(model.cluster_parameters, :arrival_distribution)
        sample_arrival_parameter_posterior = model.cluster_parameters.arrival_distribution.sample_parameter_posterior
    else 
        sample_arrival_parameter_posterior = false 
    end
    num_total_iterations = sampler.num_iterations + sampler.num_burn_in
    n = size(data)[2]
    num_chains = length(sampler.assignment_types)
    output_assignments = Array{Int64}(undef, n, num_total_iterations, num_chains)
    output_log_likelihoods = Array{Float64}(undef, num_total_iterations, num_chains)
    if sample_arrival_parameter_posterior
        output_arrival_parameter_posterior = Array{Float64}(undef, num_total_iterations, num_chains) 
    end

    for chain = 1:num_chains
        assignments = Array{Int64}(undef, n, num_total_iterations)
        log_likelihoods = Vector{Float64}(undef, num_total_iterations)
        sufficient_stats = prepare_sufficient_statistics(model, data)
        auxillary_variables = prepare_auxillary_variables(model, data)
        if sample_arrival_parameter_posterior
            arrival_parameter_posterior = Vector{Float64}(undef, num_total_iterations)
        end
        if sampler.assignment_types[chain] === "random"
            random_initial_assignment!(assignments, data, sufficient_stats, model, auxillary_variables, sampler)
        else
            deterministic_initial_assignment!(assignments, data, sufficient_stats, model, auxillary_variables, sampler, chain)
        end
        log_likelihoods[1] = compute_joint_log_likelihood(sufficient_stats, model, auxillary_variables)
        if sample_arrival_parameter_posterior
            arrival_parameter_posterior[1] = sample_arrival_posterior(sufficient_stats, auxillary_variables, model.cluster_parameters)
        end
        @showprogress for iteration = 2:num_total_iterations
            assignments[:, iteration] = assignments[:, iteration-1]
            sample!(assignments, iteration, data, sufficient_stats, model, auxillary_variables, sampler) 
            gibbs_sample!(auxillary_variables, sufficient_stats, model, data, assignments[:, iteration])
            log_likelihoods[iteration] = compute_joint_log_likelihood(sufficient_stats, model, auxillary_variables)
            if sample_arrival_parameter_posterior
                arrival_parameter_posterior[iteration] = sample_arrival_posterior(sufficient_stats, auxillary_variables, 
                                                                                  model.cluster_parameters)
            end
        end
        output_assignments[:, :, chain] = assignments[:, :]
        output_log_likelihoods[:, chain] = log_likelihoods[:]
        if sample_arrival_parameter_posterior 
            output_arrival_parameter_posterior[:, chain] = arrival_parameter_posterior[:]
        end
    end
    output_iterations = Vector((sampler.num_burn_in+1):num_total_iterations)
    output_iterations = output_iterations[output_iterations .% sampler.skip .=== 0]
    mcmc_output = Dict{String, Array}("assignments" => output_assignments[:, output_iterations, :],
                                      "log likelihood" => output_log_likelihoods[output_iterations, :])
    if sample_arrival_parameter_posterior
        mcmc_output["arrival posterior"] = output_arrival_parameter_posterior[output_iterations, :]
    end
    return mcmc_output
end

function sample_arrival_posterior(sufficient_stats, auxillary_variables, cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    posterior_parameters = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    beta_distribution = Beta(posterior_parameters[1], posterior_parameters[2])
    return rand(beta_distribution, 1)[1]
end

function sample_arrival_posterior(sufficient_stats, auxillary_variables, cluster_parameters::DpParameters) 
   return NaN 
end

function sample!(instances, iteration, data, sufficient_stats, model, auxillary_variables, sampler::GibbsSampler)
    n = sufficient_stats.n
    for observation = 1:n
        gibbs_sample!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
end

function propose_cluster(previous_cluster, observation, sufficient_stats, sampler::MetropolisHastingsSampler)
    clusters = findall(sufficient_stats.cluster.clusters[1:observation-1])
    #clusters = get_clusters(sufficient_stats.cluster)
    if !(previous_cluster in clusters)
        push!(clusters, previous_cluster)
    end
    if previous_cluster !== observation 
        push!(clusters, observation)
    end
    sort!(clusters)
    num_clusters = length(clusters)
    center_index = findfirst(clusters .=== previous_cluster)
    start_index = maximum([1, center_index - sampler.proposal_radius])
    stop_index = minimum([num_clusters, center_index + sampler.proposal_radius])
    clusters = clusters[start_index:stop_index]
    if !(observation in clusters)
        push!(clusters, observation)
    end
    cluster = rand(clusters)
    return (cluster, -log(length(clusters)))
end

function sample!(instances, iteration, data, sufficient_stats, model, auxillary_variables, sampler::MetropolisHastingsSampler)
    n = sufficient_stats.n 
    num_accepted = 0
    num_proposals = 0
    for observation = 1:n
        previous_cluster = instances[observation, iteration]
        if (observation > previous_cluster) || (observation === previous_cluster && sufficient_stats.cluster.num_observations[previous_cluster] === 1)
        #if true
            num_proposals += 1
            remove_observation!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)

            previous_cluster_log_weight = data_log_predictive(observation, previous_cluster, sufficient_stats.data, model.data_parameters, 
                                                              data, auxillary_variables)
            if previous_cluster === observation 
                previous_cluster_log_weight += new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                          auxillary_variables, observation)
            else 
                previous_cluster_log_weight += compute_cluster_log_predictive(observation, previous_cluster, sufficient_stats, model.cluster_parameters)
                previous_cluster_log_weight += existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                               auxillary_variables, observation, 
                                                                               previous_cluster) 
            end

            # Propose a new cluster for the current observation
            (proposed_cluster, proposal_log_weight) = propose_cluster(previous_cluster, observation, sufficient_stats, sampler)
            # Compute the joint likelihood of the proposal
            proposed_cluster_log_weight = data_log_predictive(observation, proposed_cluster, sufficient_stats.data, model.data_parameters,
                                                              data, auxillary_variables)
            if proposed_cluster === observation 
                proposed_cluster_log_weight += new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                          auxillary_variables, observation)
            else 
                proposed_cluster_log_weight += compute_cluster_log_predictive(observation, proposed_cluster, sufficient_stats, model.cluster_parameters)
                proposed_cluster_log_weight += existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                               auxillary_variables, observation, 
                                                                               proposed_cluster) 
            end

            (_, previous_cluster_proposal_log_weight) = propose_cluster(proposed_cluster, observation, sufficient_stats, sampler) 

            log_acceptance_ratio = proposed_cluster_log_weight + previous_cluster_proposal_log_weight
            log_acceptance_ratio -= (previous_cluster_log_weight + proposal_log_weight)

            log_uniform_variate = log(rand(Uniform(0, 1), 1)[1])
            if log_uniform_variate < log_acceptance_ratio
                final_cluster = proposed_cluster
                num_accepted += 1
            else
                final_cluster = previous_cluster
            end
            add_observation!(observation, final_cluster, instances, iteration, data, sufficient_stats, model, 
                             auxillary_variables)
        end
    end
    acceptance_rate = num_accepted/num_proposals
    if sampler.adaptive
        if acceptance_rate > 0.3
            sampler.proposal_radius += 1
        elseif acceptance_rate < 0.3
            sampler.proposal_radius = maximum([sampler.proposal_radius - 1, 1])
        end
    end
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       cluster_parameters::NtlParameters{A}) where {A <: GeometricArrivals}
    nothing
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       cluster_parameters::DpParameters)
    nothing
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       data_parameters::Union{GaussianParameters, MultinomialParameters},
                       data::Matrix{T}, cluster_assignments) where {T <: Real}
    nothing
end

function sample_topic(sufficient_stats, data_parameters, cluster, token)
    topic_parameter = data_parameters.topic_parameter
    word_parameter = data_parameters.word_parameter
    num_topics = data_parameters.num_topics
    cluster_topic_frequencies = sufficient_stats.data.cluster_topic_frequencies
    topic_token_frequencies = sufficient_stats.data.topic_token_frequencies
    topics = Vector(1:num_topics) 
    topic_log_weights = Array{Float64}(undef, num_topics)
    num_tokens = size(topic_token_frequencies)[1]
    for topic = topics
        cluster_log_weight = log(cluster_topic_frequencies[topic, cluster] + topic_parameter)
        cluster_log_weight -= log(sum(cluster_topic_frequencies[:, cluster]) + num_topics*topic_parameter)
        token_log_weight = log(topic_token_frequencies[token, topic] + word_parameter) 
        token_log_weight -= log(sum(topic_token_frequencies[:, topic]) + num_tokens*word_parameter)
        topic_log_weights[topic] = cluster_log_weight + token_log_weight
    end
    return gumbel_max(topics, topic_log_weights)
end

function update_token_topic_statistics!(document, token, topic, cluster, auxillary_variables, sufficient_stats;
                                        update_type="add")
    if update_type === "add"
        sufficient_stats.data.cluster_topic_frequencies[topic, cluster] += 1
        sufficient_stats.data.topic_token_frequencies[token, topic] += 1
        auxillary_variables.data.document_topic_frequencies[topic, document] += 1
    elseif update_type === "remove"
        sufficient_stats.data.cluster_topic_frequencies[topic, cluster] -= 1
        sufficient_stats.data.topic_token_frequencies[token, topic] -= 1
        auxillary_variables.data.document_topic_frequencies[topic, document] -= 1
    end
end

function gibbs_sample!(auxillary_variables, sufficient_stats::SufficientStatistics,
                       data_parameters::FiniteTopicModelParameters, data::Matrix{T},
                       cluster_assignments::Vector{Int64}) where {T <: Real}
    num_documents = length(cluster_assignments)
    num_tokens = size(data)[1]
    for document = 1:num_documents
        cluster = cluster_assignments[document]
        for token = 1:num_tokens
            num_instances = data[token, document]
            for instance = 1:num_instances
                previous_topic = auxillary_variables.data.token_topic_assignments[document][token][instance]
                # update sufficient statistics
                update_token_topic_statistics!(document, token, previous_topic, cluster, auxillary_variables, sufficient_stats,
                                               update_type="remove")
                # sample new topic 
                (new_topic, _) = sample_topic(sufficient_stats, data_parameters, cluster, token)
                auxillary_variables.data.token_topic_assignments[document][token][instance] = new_topic
                # update sufficient statistics
                update_token_topic_statistics!(document, token, new_topic, cluster, auxillary_variables, sufficient_stats,
                                               update_type="add")
            end
        end
    end
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, model::Model, data,
                       cluster_assignments::Vector{Int64})
    gibbs_sample!(auxillary_variables, sufficient_stats, model.cluster_parameters)
    gibbs_sample!(auxillary_variables, sufficient_stats, model.data_parameters, data, cluster_assignments)
end

function gibbs_sample!(observation, instances, iteration, data, sufficient_stats, model::Mixture, auxillary_variables)
    remove_observation!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    (cluster, _) = gibbs_proposal(observation, data, sufficient_stats, model, auxillary_variables)
    add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables)
end

function gibbs_proposal(observation::Int64, data::Matrix{T}, sufficient_stats::SufficientStatistics, 
                        model::Mixture, auxillary_variables) where {T<:Real}
    data_parameters = model.data_parameters
    cluster_parameters = model.cluster_parameters
    clusters = get_clusters(sufficient_stats.cluster)
    num_clusters = length(clusters)

    choices = Array{Int64}(undef, num_clusters+1)
    choices[1:num_clusters] = clusters
    choices[num_clusters+1] = observation

    weights = Array{Float64}(undef, num_clusters+1)
    weights[1:num_clusters] = compute_existing_cluster_log_predictives(observation, clusters, 
                                                                       sufficient_stats, cluster_parameters,
                                                                       auxillary_variables)
    weights[num_clusters+1] = new_cluster_log_predictive(sufficient_stats, cluster_parameters, auxillary_variables,
                                                         observation)
    weights += compute_data_log_predictives(observation, choices, data, sufficient_stats.data, data_parameters, auxillary_variables)

    return gumbel_max(choices, weights)
end

function propose(observation, data, particles, particle, sufficient_stats, model::Mixture, auxillary_variables)
    return gibbs_proposal(observation, data, sufficient_stats, model, auxillary_variables)
end

function prepare_arrivals_auxillary_variables(::DpParameters, data::Matrix{T}) where  
                                             {T <: Real}
    return DpAuxillaryVariables()
end

function prepare_arrivals_auxillary_variables(::NtlParameters{A}, data::Matrix{T}) where 
                                             {T <: Real, A <: GeometricArrivals}
    return GeometricAuxillaryVariables()
end

function prepare_data_auxillary_variables(::GaussianParameters, data::Matrix{T}) where {T <: Real}
    return GaussianAuxillaryVariables()
end

function prepare_data_auxillary_variables(::MultinomialParameters, data::Matrix{T}) where {T <: Real}
    return MultinomialAuxillaryVariables()
end

function prepare_data_auxillary_variables(data_parameters::FiniteTopicModelParameters, data::Matrix{T}) where {T <: Real}
    num_topics = data_parameters.num_topics
    num_documents = size(data)[2]
    num_words = size(data)[1]
    #                              Vector(Document => Dict(word => Dict(token => assignment))) 
    word_topic_assignments = Vector{Dict{Int64, Dict{Int64, Int64}}}(undef, num_documents)
    for document = 1:num_documents 
        word_dicts = []
        for word = 1:num_words 
            token_assignments = [(token, 0) for token in 1:data[word, document]]
            token_assignments = Dict{Int64, Int64}(token_assignments)
            push!(word_dicts, (word, token_assignments))
        end
        document_dict = Dict{Int64, Dict{Int64, Int64}}(word_dicts)
        word_topic_assignments[document] = document_dict 
    end
    document_topic_frequencies = zeros(Int64, num_topics, num_documents)
    data_aux_variables = FiniteTopicModelAuxillaryVariables(word_topic_assignments, document_topic_frequencies)
    return data_aux_variables
end

function prepare_auxillary_variables(model, data)
    arrival_aux_variables = prepare_arrivals_auxillary_variables(model.cluster_parameters, data)
    data_aux_variables = prepare_data_auxillary_variables(model.data_parameters, data)
    auxillary_variables = AuxillaryVariables(arrival_aux_variables, data_aux_variables)
    return auxillary_variables
end

function prepare_sufficient_statistics(model, data)
    n = size(data)[2]
    num_assigned = 0
    cluster_sufficient_stats = prepare_cluster_sufficient_statistics(model, n)
    data_sufficient_stats = prepare_data_sufficient_statistics(data, model.data_parameters)
    return SufficientStatistics(n, num_assigned, cluster_sufficient_stats, data_sufficient_stats)
end

function prepare_data_sufficient_statistics(data::Matrix{Float64}, data_parameters::GaussianParameters)
    dim = size(data)[1]
    n = size(data)[2]
    data_means = zeros(Float64, dim, n+1)
    data_precision_quadratic_sums = zeros(Float64, n+1)
    posterior_means = Matrix{Float64}(undef, dim, n+1)
    for i = 1:(n+1)
        posterior_means[:, i] = data_parameters.prior_mean
    end
    posterior_covs = Matrix{Float64}(undef, dim*dim, n+1)
    for i = 1:(n+1)
        posterior_covs[:, i] = vec(data_parameters.prior_covariance)
    end
    return GaussianSufficientStatistics(data_means, data_precision_quadratic_sums, posterior_means, posterior_covs)
end

function prepare_data_sufficient_statistics(data::Matrix{Int64}, data_parameters::MultinomialParameters)
    dim = size(data)[1]
    n = size(data)[2]
    counts = zeros(Int64, dim, n+1)
    posterior_dirichlet_scale = Matrix{Float64}(undef, dim, n+1)
    for i = 1:(n+1)
        posterior_dirichlet_scale[:, i] = vec(data_parameters.prior_dirichlet_scale)
    end
    return MultinomialSufficientStatistics(counts, posterior_dirichlet_scale)
end

function prepare_data_sufficient_statistics(data::Matrix{Int64}, data_parameters::FiniteTopicModelParameters)
    num_topics = data_parameters.num_topics 
    num_tokens = size(data)[1]
    num_documents = size(data)[2] 
    cluster_topic_frequencies = zeros(Int64, num_topics, num_documents)
    cluster_topic_posterior = zeros(Int64, num_topics, num_documents)
    topic_token_frequencies = zeros(Int64, num_tokens, num_topics) 
    topic_token_posterior = zeros(Int64, num_tokens, num_topics) 
    return FiniteTopicModelSufficientStatistics(cluster_topic_frequencies, cluster_topic_posterior, 
                                                topic_token_frequencies, topic_token_posterior)
end

function prepare_cluster_sufficient_statistics(::Mixture, n::Int64)
    cluster_num_observations = Vector{Int64}(zeros(Int64, n))
    cumulative_num_observations = FenwickTree{Int64}(n)
    clusters = BitArray(undef, n)
    clusters .= false
    cluster_sufficient_stats = MixtureSufficientStatistics(cluster_num_observations, cumulative_num_observations, clusters)
    return cluster_sufficient_stats
end

function compute_cluster_data_log_likelihood(cluster::Int64, sufficient_stats::SufficientStatistics, data_parameters::FiniteTopicModelParameters,
                                             auxillary_variables) 
    dirichlet_posterior = vec(sufficient_stats.data.cluster_topic_posterior[:, cluster])
    num_topics = length(dirichlet_posterior)
    dirichlet_prior = data_parameters.topic_parameter*ones(Float64, num_topics)
    counts = convert(Vector{Int64}, round.(dirichlet_posterior - dirichlet_prior))
    log_likelihood = log_multinomial_coeff(counts) + logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
    return log_likelihood
end

function compute_cluster_data_log_likelihood(cluster::Int64, sufficient_stats::SufficientStatistics,
                                             data_parameters::MultinomialParameters, auxillary_variables)
    dirichlet_posterior = sufficient_stats.data.posterior_dirichlet_scale[:, cluster]
    dirichlet_prior = data_parameters.prior_dirichlet_scale
    counts = convert(Vector{Int64}, round.(dirichlet_posterior - dirichlet_prior))
    #log_likelihood = log_multinomial_coeff(counts) + logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
    log_likelihood = logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
    return log_likelihood
end

function compute_cluster_data_log_likelihood(cluster::Int64, sufficient_stats::SufficientStatistics, 
                                             data_parameters::GaussianParameters, auxillary_variables) 
    posterior_mean = vec(sufficient_stats.data.posterior_means[:, cluster])
    dim = length(posterior_mean)
    posterior_covariance = reshape(sufficient_stats.data.posterior_covs[:, cluster], dim, dim)
    prior_mean = data_parameters.prior_mean
    prior_covariance = data_parameters.prior_covariance
    prior_precision = inv(prior_covariance)

    data_precision_quadratic_sum = sufficient_stats.data.data_precision_quadratic_sums[cluster]
    posterior_quadratic = transpose(posterior_mean) * posterior_covariance * posterior_mean 
    prior_quadratic = transpose(prior_mean) * prior_precision * prior_mean

    log_likelihood = -(1/2)*(data_precision_quadratic_sum + prior_quadratic - posterior_quadratic)

    data_log_determinant = log(abs(det(data_parameters.data_covariance)))
    prior_log_determinant = log(abs(det(prior_covariance)))
    posterior_log_determinant = log(abs(det(posterior_covariance)))

    n = sufficient_stats.cluster.num_observations[cluster]
    log_likelihood += log(2*pi)*dim*(n + 1)/2 - (n/2)data_log_determinant - (1/2)prior_log_determinant + (1/2)posterior_log_determinant
    return log_likelihood
end

function compute_auxillary_data_log_likelihood(sufficient_stats, data_parameters::Union{GaussianParameters, MultinomialParameters}, 
                                               auxillary_variables)
    return 0.
end

function compute_auxillary_data_log_likelihood(sufficient_stats, data_parameters::FiniteTopicModelParameters, auxillary_variables)
    log_likelihood = 0.
    num_words = data_parameters.num_words
    dirichlet_prior = data_parameters.word_parameter*ones(Int64, num_words)
    num_topics = data_parameters.num_topics
    for topic = 1:num_topics
        dirichlet_posterior = vec(sufficient_stats.data.topic_token_posterior[:, topic])
        counts = vec(sufficient_stats.data.topic_token_frequencies[:, topic])
        topic_log_likelihood = logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
        log_likelihood += topic_log_likelihood
    end
    return log_likelihood
end

function compute_data_log_likelihood(sufficient_stats::SufficientStatistics, data_parameters::DataParameters, auxillary_variables) 
    clusters = get_clusters(sufficient_stats.cluster)
    log_likelihood = 0
    for cluster = clusters
        cluster_data_log_likelihood = compute_cluster_data_log_likelihood(cluster, sufficient_stats, data_parameters, auxillary_variables)
        log_likelihood += cluster_data_log_likelihood
    end
    aux_log_likelihood = compute_auxillary_data_log_likelihood(sufficient_stats, data_parameters, auxillary_variables)
    log_likelihood += aux_log_likelihood
    return log_likelihood
end

function compute_cluster_assignment_log_likelihood(cluster::Int64, cluster_sufficient_stats::MixtureSufficientStatistics, 
                                                   cluster_parameters::NtlParameters{T}) where {T <: ArrivalDistribution}
    if cluster > 1
        psi_posterior = compute_stick_breaking_posterior(cluster, cluster_sufficient_stats, cluster_parameters.prior)
        return logbeta(psi_posterior) - logbeta(cluster_parameters.prior)
    else
        return 0
    end
end

function compute_arrivals_log_likelihood(sufficient_stats::SufficientStatistics, cluster_parameters::NtlParameters{T}, 
                                         auxillary_variables) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    log_likelihood = logbeta(phi_posterior) - logbeta(cluster_parameters.arrival_distribution.prior)
    @assert (log_likelihood < Inf)
    return log_likelihood 
end

function compute_assignment_log_likelihood(sufficient_stats::SufficientStatistics{C, D},
                                           cluster_parameters::DpParameters,
                                           auxillary_variables) where {C <: MixtureSufficientStatistics, D}
    num_observations = sufficient_stats.cluster.num_observations
    num_observations = num_observations[num_observations .> 0]
    num_clusters = length(num_observations)
    alpha = cluster_parameters.alpha
    log_likelihood = sum(loggamma.(num_observations)) + num_clusters*log(alpha)
    n = sufficient_stats.n
    log_likelihood += loggamma(alpha) - loggamma(n + alpha)
    return log_likelihood
end

function compute_assignment_log_likelihood(sufficient_stats::SufficientStatistics{C, D}, 
                                           cluster_parameters::ParametricArrivalsClusterParameters{T},
                                           auxillary_variables) where 
                                           {T, C <: MixtureSufficientStatistics, D}
    log_likelihood = 0
    clusters = get_clusters(sufficient_stats.cluster)
    for cluster = clusters
        log_likelihood += compute_cluster_assignment_log_likelihood(cluster, sufficient_stats.cluster, 
                                                                    cluster_parameters)
        @assert (log_likelihood < Inf)
    end
    log_likelihood += compute_arrivals_log_likelihood(sufficient_stats, cluster_parameters, auxillary_variables)
    @assert (log_likelihood < Inf)
    return log_likelihood
end

function compute_joint_log_likelihood(sufficient_stats::SufficientStatistics, model, auxillary_variables)
    data_parameters = model.data_parameters
    cluster_parameters = model.cluster_parameters
    data_log_likelihood = compute_data_log_likelihood(sufficient_stats, data_parameters, auxillary_variables)
    @assert (!isnan(data_log_likelihood) & (data_log_likelihood < Inf))
    assignment_log_likelihood = compute_assignment_log_likelihood(sufficient_stats, cluster_parameters, auxillary_variables)
    @assert (!isnan(assignment_log_likelihood) & (assignment_log_likelihood < Inf))
    log_likelihood = data_log_likelihood + assignment_log_likelihood
    return log_likelihood
end

function update_data_sufficient_statistics!(sufficient_stats::SufficientStatistics{C,D}, cluster::Int64, 
                                            observation::Int64, model::Model, data, auxillary_variables;
                                            update_type::String="add") where {C, D<:GaussianSufficientStatistics}
    datum = vec(data[:, observation])
    cluster_mean = vec(sufficient_stats.data.data_means[:, cluster])
    data_precision_quadratic_sum = sufficient_stats.data.data_precision_quadratic_sums[cluster] 
    n = sufficient_stats.cluster.num_observations[cluster]
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
    sufficient_stats.data.data_means[:, cluster] = cluster_mean
    sufficient_stats.data.data_precision_quadratic_sums[cluster] = data_precision_quadratic_sum
end

function update_data_sufficient_statistics!(sufficient_stats::SufficientStatistics{C, D}, cluster::Int64,
                                            observation::Int64, model::Model, data, auxillary_variables;
                                            update_type::String="add") where {C, D <: FiniteTopicModelSufficientStatistics}
    doc_topic_frequencies = vec(auxillary_variables.data.document_topic_frequencies[:, observation])
    if update_type === "add"
        sufficient_stats.data.cluster_topic_frequencies[:, cluster] += doc_topic_frequencies
    elseif update_type === "remove"
        sufficient_stats.data.cluster_topic_frequencies[:, cluster] -= doc_topic_frequencies
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_data_sufficient_statistics!(sufficient_stats::SufficientStatistics{C,D}, cluster::Int64, 
                                            observation::Int64, ::Model, data, auxillary_variables;
                                            update_type::String="add") where {C, D<:MultinomialSufficientStatistics}
    datum = vec(data[:, observation])
    cluster_counts = vec(sufficient_stats.data.total_counts[:, cluster])
    if update_type === "add"
        cluster_counts += datum 
    elseif update_type === "remove"
        cluster_counts -= datum
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
    sufficient_stats.data.total_counts[:, cluster] = cluster_counts
end

function update_cluster_sufficient_statistics!(sufficient_stats::SufficientStatistics{C, D}, cluster::Int64; 
                                               update_type::String="add") where {C <: MixtureSufficientStatistics, D}
    if update_type === "add"
        sufficient_stats.cluster.num_observations[cluster] += 1
        inc!(sufficient_stats.cluster.cumulative_num_observations, cluster, 1)
        if sufficient_stats.cluster.num_observations[cluster] === 1
            sufficient_stats.cluster.clusters[cluster] = true
        end
        sufficient_stats.num_assigned += 1
    elseif update_type === "remove"
        sufficient_stats.cluster.num_observations[cluster] -= 1
        dec!(sufficient_stats.cluster.cumulative_num_observations, cluster, 1)
        if sufficient_stats.cluster.num_observations[cluster] === 0
            sufficient_stats.cluster.clusters[cluster] = false
        end
        sufficient_stats.num_assigned -= 1
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_cluster_stats_new_birth_time!(num_observations::Vector{Int64}, new_time::Int64, old_time::Int64)
    num_observations[new_time] = num_observations[old_time]
    num_observations[old_time] = 0
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::MixtureSufficientStatistics,
                                              new_time::Int64, old_time::Int64)
    num_observations = cluster_sufficient_stats.num_observations[old_time]
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats.num_observations, new_time, old_time)
    cluster_sufficient_stats.clusters[new_time] = true
    cluster_sufficient_stats.clusters[old_time] = false
    dec!(cluster_sufficient_stats.cumulative_num_observations, old_time, num_observations)
    inc!(cluster_sufficient_stats.cumulative_num_observations, new_time, num_observations)
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

function update_data_stats_new_birth_time!(data_sufficient_stats::FiniteTopicModelSufficientStatistics,
                                           new_time::Int64, old_time::Int64, data_parameters::FiniteTopicModelParameters)
    data_sufficient_stats.cluster_topic_frequencies[:, new_time] = data_sufficient_stats.cluster_topic_frequencies[:, old_time]
    data_sufficient_stats.cluster_topic_frequencies[:, old_time] .= 0
    data_sufficient_stats.cluster_topic_posterior[:, new_time] = data_sufficient_stats.cluster_topic_posterior[:, old_time]
    data_sufficient_stats.cluster_topic_posterior[:, old_time] .= 0
end

function update_cluster_birth_time_remove!(cluster::Int64, iteration::Int64, instances::Matrix{Int64}, 
                                           sufficient_stats::SufficientStatistics,
                                           data_parameters::DataParameters)
    assigned_to_cluster = instances[:, iteration] .=== cluster
    new_cluster_time = findfirst(assigned_to_cluster)
    instances[assigned_to_cluster, iteration] .= new_cluster_time 
    update_cluster_stats_new_birth_time!(sufficient_stats.cluster, new_cluster_time, cluster)
    update_data_stats_new_birth_time!(sufficient_stats.data, new_cluster_time, cluster, data_parameters)
    return new_cluster_time
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             sufficient_stats::SufficientStatistics, model::Mixture,
                             auxillary_variables::AuxillaryVariables) where {T <: Real}
    n = sufficient_stats.n
    cluster = instances[observation, iteration]

    instances[observation, iteration] = n+1 
    update_cluster_sufficient_statistics!(sufficient_stats, cluster, update_type="remove")
    update_data_sufficient_statistics!(sufficient_stats, cluster, observation, model, data, auxillary_variables, 
                                       update_type="remove")
    compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
    if (cluster === observation) && (sufficient_stats.cluster.num_observations[cluster] > 0)
        cluster = update_cluster_birth_time_remove!(cluster, iteration, instances, sufficient_stats, 
                                                    model.data_parameters)
    end
end

function update_cluster_birth_time_add!(cluster::Int64, observation::Int64, iteration::Int64, instances::Matrix{Int64}, 
                                        sufficient_stats::SufficientStatistics, data_parameters)
    assigned_to_cluster = (instances[:, iteration] .=== cluster)
    old_cluster_time = cluster
    cluster = observation
    instances[assigned_to_cluster, iteration] .= cluster
    update_cluster_stats_new_birth_time!(sufficient_stats.cluster, cluster, old_cluster_time)
    update_data_stats_new_birth_time!(sufficient_stats.data, cluster, old_cluster_time, data_parameters)
    return cluster
end

function add_observation!(observation::Int64, cluster::Int64, instances::Matrix{Int64}, iteration::Int64, 
                          data::Matrix{T}, sufficient_stats::SufficientStatistics, model::Mixture,
                          auxillary_variables::AuxillaryVariables) where 
                          {T <: Real}
    if observation < cluster
        cluster = update_cluster_birth_time_add!(cluster, observation, iteration, instances, sufficient_stats,
                                                 model.data_parameters)
    end
    instances[observation, iteration] = cluster
    update_cluster_sufficient_statistics!(sufficient_stats, cluster, update_type="add")
    update_data_sufficient_statistics!(sufficient_stats, cluster, observation, model, data, auxillary_variables, 
                                       update_type="add")
    compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
end

function compute_arrival_distribution_posterior(sufficient_stats::SufficientStatistics{C, D},
                                                cluster_parameters::ParametricArrivalsClusterParameters{T}) where 
                                                {T <: GeometricArrivals, C <: Union{MixtureSufficientStatistics, HmmSufficientStatistics}, D}
    num_assigned = sufficient_stats.num_assigned
    num_clusters = length(get_clusters(sufficient_stats.cluster))
    @assert num_assigned >= num_clusters
    phi_posterior = deepcopy(cluster_parameters.arrival_distribution.prior)
    phi_posterior[1] += num_clusters - 1
    phi_posterior[2] += num_assigned - num_clusters 
    return phi_posterior
end

### Mixture cluster log predictives 

function new_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::DpParameters, 
                                    auxillary_variables::AuxillaryVariables, ::Int64) where {C <: MixtureSufficientStatistics, D}
    return log(cluster_parameters.alpha)
end

function new_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{T},
                                    auxillary_variables, observation::Int64) where 
                                    {T <: GeometricArrivals, C <: MixtureSufficientStatistics, D}
    num_complement = compute_num_complement(observation, sufficient_stats.cluster)
    psi_posterior = deepcopy(cluster_parameters.prior)
    psi_posterior[2] += num_complement
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    return log(phi_posterior[1]) - log(sum(phi_posterior)) + logbeta(psi_posterior) - logbeta(cluster_parameters.prior)
end

function existing_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D},
                                         cluster_parameters::ParametricArrivalsClusterParameters{T}, 
                                         auxillary_variables, observation::Int64, cluster::Int64) where 
                                         {T <: GeometricArrivals, C <: MixtureSufficientStatistics, D}
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    return log(phi_posterior[2]) - log(sum(phi_posterior))
end

### End mixture cluster log predictives

function compute_num_complement(cluster::Int64, cluster_suff_stats::MixtureSufficientStatistics;
                                missing_observation::Union{Int64, Nothing}=nothing)
    sum_of_older_clusters = sum(cluster_suff_stats.num_observations[1:(cluster-1)])
    num_complement = sum_of_older_clusters - (cluster - 1)
    if missing_observation !== nothing && missing_observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_stick_breaking_posterior(cluster::Int64, cluster_suff_stats::MixtureSufficientStatistics, beta_prior::Vector{T};
                                          missing_observation::Union{Int64, Nothing}=nothing) where {T <: Real}
    posterior = deepcopy(beta_prior)
    num_obs = cluster_suff_stats.num_observations[cluster]
    @assert num_obs >= 0
    if num_obs > 0
        posterior[1] += cluster_suff_stats.num_observations[cluster] - 1
    end
    posterior[2] += compute_num_complement(cluster, cluster_suff_stats, missing_observation=missing_observation)
    @assert posterior[1] > 0
    @assert posterior[2] > 0
    return posterior
end

function compute_stick_breaking_posterior(cluster::Int64, sufficient_stats::SufficientStatistics, 
                                          ntl_parameters::ParametricArrivalsClusterParameters; 
                                          missing_observation::Union{Int64, Nothing}=nothing)
    posterior = compute_stick_breaking_posterior(cluster, sufficient_stats.cluster, 
                                                 ntl_parameters.prior, missing_observation=missing_observation)
    return posterior
end

function compute_cluster_log_predictive(observation, cluster, sufficient_stats, cluster_parameters::NtlParameters;
                                        psi_parameters::Matrix=Array{Int64}(undef, 2, 0),
                                        logbetas::Vector=Array{Float64}(undef, 0))
    log_predictive = 0
    # Clusters younger than the current cluster
    clusters = get_clusters(sufficient_stats.cluster)

    if cluster < observation
        if cluster > 1
            if size(psi_parameters)[2] === 0
                cluster_psi = compute_stick_breaking_posterior(cluster, sufficient_stats, cluster_parameters,
                                                               missing_observation=observation)
            else 
                cluster_psi = vec(psi_parameters[:, cluster])
            end
            log_predictive += log(cluster_psi[1]) - log(sum(cluster_psi))
        end

        # clusters younger than the current cluster
        younger_clusters_range = Vector{Int64}((cluster+1):(observation-1))
        younger_clusters_mask = clusters .∈ Ref(younger_clusters_range) 
        younger_clusters = clusters[younger_clusters_mask]
        for younger_cluster = younger_clusters
            if size(psi_parameters)[2] === 0 
                younger_cluster_psi = compute_stick_breaking_posterior(younger_cluster, sufficient_stats, cluster_parameters,
                                                                       missing_observation=observation)
            else 
                younger_cluster_psi = vec(psi_parameters[:, younger_cluster])
            end
            log_predictive += log(younger_cluster_psi[2]) - log(sum(younger_cluster_psi))
        end
    else # observation < cluster
        cluster_num_obs = sufficient_stats.cluster.num_observations[cluster]
        cluster_old_psi = compute_stick_breaking_posterior(cluster, sufficient_stats, cluster_parameters,
                                                           missing_observation=observation)
        younger_clusters_log_predictive = 0
        # clusters younger than the observation
        younger_clusters_range = Vector{Int64}((observation+1):(cluster-1))
        younger_clusters_mask = clusters .∈ Ref(younger_clusters_range) 
        younger_clusters = clusters[younger_clusters_mask]

        for younger_cluster = younger_clusters
            if size(psi_parameters)[2] === 0
                younger_cluster_old_psi = compute_stick_breaking_posterior(younger_cluster, sufficient_stats, cluster_parameters,
                                                                           missing_observation=observation)
            else 
                younger_cluster_old_psi = vec(psi_parameters[:, younger_cluster])
            end
            younger_cluster_new_psi = deepcopy(younger_cluster_old_psi)
            younger_cluster_new_psi[2] += cluster_num_obs
            if length(logbetas) === 0
                younger_clusters_log_predictive += logbeta(younger_cluster_new_psi) - logbeta(younger_cluster_old_psi)
            else 
                younger_clusters_log_predictive += logbeta(younger_cluster_new_psi) - logbetas[younger_cluster]
            end
        end

        log_predictive = younger_clusters_log_predictive

        if observation > 1
            new_num_complement = compute_num_complement(observation, sufficient_stats.cluster, missing_observation=observation)
            cluster_new_psi = deepcopy(cluster_old_psi)
            cluster_new_psi[1] += 1
            cluster_new_psi[2] = new_num_complement + cluster_parameters.prior[2]
        else 
            cluster_new_psi = [1, 1]
        end

        if length(logbetas) === 0
            cluster_log_predictive = logbeta(cluster_new_psi) - logbeta(cluster_old_psi)
        else 
            cluster_log_predictive = logbeta(cluster_new_psi) - logbetas[cluster]
        end

        log_predictive += cluster_log_predictive 
    end
    return log_predictive
end

function compute_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, 
                                         sufficient_stats, ntl_parameters::NtlParameters)
    # Not strictly the number of observations
    n = maximum(clusters)
    n = maximum([n, observation])

    clusters = get_clusters(sufficient_stats.cluster)
    num_clusters = length(clusters)
    log_weights = Array{Float64}(undef, num_clusters) 
    psi_parameters = zeros(Float64, 2, n)
    logbetas = zeros(Float64, n)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_stick_breaking_posterior(cluster, sufficient_stats, ntl_parameters, 
                                                           missing_observation=observation)
            psi_parameters[:, cluster] = cluster_psi
            logbetas[cluster] = logbeta(cluster_psi)
        end
    end

    for (i, cluster) = enumerate(clusters)
        log_weights[i] = compute_cluster_log_predictive(observation, cluster, sufficient_stats, ntl_parameters, 
                                                        psi_parameters=psi_parameters, logbetas=logbetas)
    end
    return log_weights
end

function compute_existing_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, 
                                                  sufficient_stats::SufficientStatistics, 
                                                  cluster_parameters::ParametricArrivalsClusterParameters,
                                                  auxillary_variables)
    log_weights = compute_cluster_log_predictives(observation, clusters, sufficient_stats, cluster_parameters)
    for (i, cluster) in enumerate(clusters)
        log_weights[i] += existing_cluster_log_predictive(sufficient_stats, cluster_parameters, auxillary_variables, observation, cluster)
    end
    return log_weights
end

function compute_existing_cluster_log_predictives(::Int64, clusters::Vector{Int64}, sufficient_stats, ::DpParameters,
                                                  auxillary_variables)
    num_observations = deepcopy(sufficient_stats.cluster.num_observations[clusters])
    num_observations[num_observations .< 1] .= 1
    log_weights = log.(num_observations)
    return log_weights
end

function compute_data_posterior_parameters!(cluster::Int64, sufficient_stats::SufficientStatistics,
                                            data_parameters::GaussianParameters)
    data_mean = vec(sufficient_stats.data.data_means[:, cluster])
    n = sufficient_stats.cluster.num_observations[cluster]
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
    sufficient_stats.data.posterior_means[:, cluster] = vec(posterior_mean)
    sufficient_stats.data.posterior_covs[:, cluster] = vec(cov)
end

function compute_data_posterior_parameters!(cluster::Int64, sufficient_stats, data_parameters::MultinomialParameters)
    cluster_counts = sufficient_stats.data.total_counts[:, cluster]
    posterior_parameters = (cluster_counts + data_parameters.prior_dirichlet_scale)
    sufficient_stats.data.posterior_dirichlet_scale[:, cluster] = posterior_parameters
end

function compute_data_posterior_parameters!(cluster::Int64, sufficient_stats, data_parameters::FiniteTopicModelParameters)
    sufficient_stats.data.cluster_topic_posterior[:, :] = sufficient_stats.data.cluster_topic_frequencies .+ data_parameters.topic_parameter
    sufficient_stats.data.topic_token_posterior[:, :] = sufficient_stats.data.topic_token_frequencies .+ data_parameters.word_parameter
end

function data_log_predictive(observation::Int64, cluster::Int64, 
                             data_sufficient_stats::FiniteTopicModelSufficientStatistics, 
                             data_parameters::FiniteTopicModelParameters, data, auxillary_variables)
    document_topic_frequencies = vec(auxillary_variables.data.document_topic_frequencies[:, observation])
    cluster_topic_posterior = vec(data_sufficient_stats.cluster_topic_posterior[:, cluster])
    n = sum(document_topic_frequencies)
    posterior = DirichletMultinomial(n, cluster_topic_posterior)
    return logpdf(posterior, document_topic_frequencies)
end

function data_log_predictive(observation::Int64, cluster::Int64, 
                             data_sufficient_stats::GaussianSufficientStatistics, 
                             data_parameters::GaussianParameters, data, auxillary_variables)
    datum = vec(data[:, observation])
    posterior_mean = vec(data_sufficient_stats.posterior_means[:, cluster])
    dim = length(posterior_mean)
    posterior_cov = reshape(data_sufficient_stats.posterior_covs[:, cluster], dim, dim)
    posterior = MvNormal(posterior_mean, data_parameters.data_covariance + posterior_cov)
    return logpdf(posterior, datum)
end

function data_log_predictive(observation::Int64, cluster::Int64, 
                             data_sufficient_stats::MultinomialSufficientStatistics,
                             ::MultinomialParameters, data, auxillary_variables)
    datum = vec(data[:, observation])
    posterior_scale = vec(data_sufficient_stats.posterior_dirichlet_scale[:, cluster])
    n = sum(datum)
    posterior = DirichletMultinomial(n, posterior_scale)
    log_predictive = logpdf(posterior, datum)
    return log_predictive
end

function compute_data_log_predictives(observation::Int64, clusters::Vector{Int64}, data::Matrix{T}, 
                                      data_sufficient_stats::DataSufficientStatistics, 
                                      data_parameters::DataParameters, auxillary_variables) where {T <: Real}
    log_predictive = Array{Float64}(undef, length(clusters))
    for (index, cluster) = enumerate(clusters)
        log_predictive[index] = data_log_predictive(observation, cluster, data_sufficient_stats, data_parameters, data, auxillary_variables)
    end
    return log_predictive
end

get_clusters(cluster_sufficient_stats::ClusterSufficientStatistics) = findall(cluster_sufficient_stats.clusters)


function update_cluster_stats_new_birth_time!(num_observations::Vector{Int64}, new_time::Int64, old_time::Int64)
    num_observations[new_time] = num_observations[old_time]
    num_observations[old_time] = 0
end

function compute_cluster_log_predictives(observation, cluster::Int64, previous_cluster, cluster_sufficient_stats,
                                         cluster_parameters::NtlParameters)
    return reshape(compute_cluster_log_predictives(observation, [cluster], previous_cluster, cluster_sufficient_stats,
                                                   cluster_parameters), 1)[1]
end


function compute_existing_cluster_log_predictives(observation::Int64, cluster::Int64, cluster_sufficient_stats,
                                                  cluster_parameters)
    result = compute_existing_cluster_log_predictives(observation, [cluster], cluster_sufficient_stats, 
                                                      cluster_parameters)
    return reshape(result, 1)[1]
end

end