module Fitter

using ..Models: DataParameters, ClusterParameters, GaussianParameters, NtlParameters, DpParameters
using ..Models: FiniteTopicModelParameters, FiniteTopicModelSufficientStatistics
using ..Models: GaussianWishartParameters, GaussianWishartSufficientStatistics
using ..Models: DataSufficientStatistics, MixtureSufficientStatistics, GaussianSufficientStatistics
using ..Models: MultinomialParameters, MultinomialSufficientStatistics, GeometricArrivals, PoissonArrivals
using ..Models: HmmSufficientStatistics, StationaryHmmSufficientStatistics, NonstationaryHmmSufficientStatistics
using ..Models: Model, Mixture, HiddenMarkovModel, ClusterSufficientStatistics, Changepoint
using ..Models: ParametricArrivalsClusterParameters, BetaNtlParameters
using ..Models: ChangepointSufficientStatistics, PitmanYorArrivals, SufficientStatistics 
using ..Models: ArrivalDistribution, AuxillaryVariables, ArrivalsAuxillaryVariables
using ..Models: GaussianAuxillaryVariables, GaussianWishartAuxillaryVariables, MultinomialAuxillaryVariables
using ..Models: FiniteTopicModelAuxillaryVariables, PoissonAuxillaryVariables, GeometricAuxillaryVariables
using ..Models: DpAuxillaryVariables
using ..Samplers: Sampler, GibbsSampler, SequentialMonteCarlo, SequentialImportanceSampler
using ..Samplers: MetropolisWithinGibbsSampler, MetropolisHastingsSampler
using ..Utils: logbeta, logmvbeta, log_multinomial_coeff, gumbel_max, isnothing, compute_normalized_weights
using ..Utils: effective_sample_size, hist
using ..Utils: mvt_logpdf, logfactorial

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
                                    {A, D <: Union{GaussianAuxillaryVariables, GaussianWishartAuxillaryVariables, MultinomialAuxillaryVariables}}
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
                                    model::Union{Mixture, HiddenMarkovModel}, auxillary_variables::AuxillaryVariables{A, F},
                                    sampler::GibbsSampler) where {C, D, A, F}
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
                                    model::Union{Mixture, HiddenMarkovModel}, auxillary_variables::AuxillaryVariables{A, F},
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
            clusters_to_sample = clusters[clusters .> observation - sampler.observation_window]
            cluster = rand(clusters_to_sample)
        end
        add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
    random_initial_assignment!(auxillary_variables, sufficient_stats, vec(instances[:, iteration]), model, data)
end

function random_initial_assignment!(instances, data, sufficient_stats, model::Changepoint, auxillary_variables, sampler)
    n = sufficient_stats.n
    observation = 1
    changepoint = 1
    iteration = 1
    add_observation!(observation, changepoint, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    for observation = 2:n
        coin_flip = reshape(rand(Binomial(1, 0.5), 1), 1)[1]
        if coin_flip === 1
            changepoint = observation
        end
        add_observation!(observation, changepoint, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
end

function fit(data::Matrix{T}, model, sampler::MetropolisWithinGibbsSampler) where {T <: Real}
    n = size(data)[2]
    instances = Array{Int64}(undef, n, sampler.num_iterations + sampler.num_burn_in)
    log_likelihoods = Vector{Float64}(undef, sampler.num_iterations + sampler.num_burn_in)
    n = size(data)[2]
    sufficient_stats = prepare_sufficient_statistics(model, data)
    auxillary_variables = prepare_auxillary_variables(model, data)
    # Assign all of the observations to the first cluster
    random_initial_assignment!(instances, data, sufficient_stats, model, auxillary_variables, sampler)
    log_likelihoods[1] = compute_joint_log_likelihood(sufficient_stats, model)
    #for iteration = ProgressBar(2:(sampler.num_iterations + sampler.num_burn_in), printing_delay=0.001)
    @showprogress for iteration = 2:(sampler.num_iterations + sampler.num_burn_in)
        instances[:, iteration] = instances[:, iteration-1]
        for observation = 1:n
            sample!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables, sampler)
        end
        gibbs_sample!(auxillary_variables, sufficient_stats, model, data, instances[:, iteration])
        log_likelihoods[iteration] = compute_joint_log_likelihood(sufficient_stats, model)
    end
    return (instances[:, (sampler.num_burn_in+1):end], log_likelihoods[(sampler.num_burn_in+1):end])
end

function sample!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables, sampler::GibbsSampler)
    gibbs_sample!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)
end

function propose_cluster(previous_cluster, observation, sufficient_stats, sampler::MetropolisHastingsSampler)
    start_index = maximum([1, observation - sampler.observation_window])
    clusters = findall(sufficient_stats.cluster.clusters[start_index:observation-1])
    if observation - sampler.observation_window > 0
        clusters .+= (observation - sampler.observation_window - 1)
    end
    if !(previous_cluster in clusters)
        push!(clusters, previous_cluster)
    end
    if previous_cluster !== observation 
        push!(clusters, observation)
    end
    sort!(clusters)
    num_clusters = length(clusters)
    center_index = findfirst(clusters .=== previous_cluster)
    start_index = maximum([1, center_index - sampler.cluster_radius])
    stop_index = minimum([num_clusters, center_index + sampler.cluster_radius])
    sampled_index = rand(start_index:stop_index)
    return (clusters[sampled_index], -log(stop_index - start_index))
end

function compute_cluster_log_predictive(observation, cluster, sufficient_stats, model::Mixture{C, D}) where {C <: NtlParameters, D}
    log_predictive = 0
    cluster_psi = compute_stick_breaking_posterior(cluster, sufficient_stats.cluster, model.cluster_parameters.prior)
    log_predictive += log(cluster_psi[1]) - log(sum(cluster_psi))
    younger_clusters = (cluster+1):(observation-1)
    for younger_cluster = younger_clusters 
        younger_cluster_psi = compute_stick_breaking_posterior(younger_cluster, sufficient_stats.cluster, model.cluster_parameters.prior)
        log_predictive += log(younger_cluster_psi[2]) - log(sum(younger_cluster_psi))
    end
    return log_predictive
end

function compute_cluster_log_predictive(observation, cluster, sufficient_stats, model::Mixture{C, D}) where {C <: DpParameters, D}
    log_predictive = 0
    cluster_psi = compute_stick_breaking_posterior(cluster, sufficient_stats.cluster, model.cluster_parameters.prior)
    log_predictive += log(cluster_psi[1]) - log(sum(cluster_psi))
    younger_clusters = (cluster+1):(observation-1)
    for younger_cluster = younger_clusters 
        younger_cluster_psi = compute_stick_breaking_posterior(younger_cluster, sufficient_stats.cluster, model.cluster_parameters.prior)
        log_predictive += log(younger_cluster_psi[2]) - log(sum(younger_cluster_psi))
    end
    return log_predictive
end

function sample!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables, sampler::MetropolisHastingsSampler)
    previous_cluster = instances[observation, iteration]
    if (observation > previous_cluster) || (observation === previous_cluster && sufficient_stats.cluster.num_observations[previous_cluster] === 1)
        remove_observation!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)

        previous_cluster_log_weight = data_log_predictive(observation, previous_cluster, sufficient_stats.data, model.data_parameters, 
                                                          data, auxillary_variables)
        previous_cluster_log_weight += compute_cluster_log_predictive(observation, previous_cluster, sufficient_stats, model)
        if previous_cluster === observation 
            previous_cluster_log_weight += new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                      auxillary_variables, previous_cluster)
        else 
            previous_cluster_log_weight += existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                           auxillary_variables, observation, 
                                                                           previous_cluster) 
        end

        # Propose a new cluster for the current observation
        (proposed_cluster, proposal_log_weight) = propose_cluster(previous_cluster, observation, sufficient_stats, sampler)
        # Compute the joint likelihood of the proposal
        proposed_cluster_log_weight = data_log_predictive(observation, proposed_cluster, sufficient_stats.data, model.data_parameters,
                                                          data, auxillary_variables)
        proposed_cluster_log_weight += compute_cluster_log_predictive(observation, proposed_cluster, sufficient_stats, model)
        if proposed_cluster === observation 
            proposed_cluster_log_weight += new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                      auxillary_variables, proposed_cluster)
        else 
            proposed_cluster_log_weight += existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                           auxillary_variables, observation, 
                                                                           proposed_cluster) 
        end

        (_, previous_cluster_proposal_log_weight) = propose_cluster(proposed_cluster, observation, sufficient_stats, sampler) 

        log_acceptance_ratio = proposed_cluster_log_weight - previous_cluster_log_weight
        log_acceptance_ratio += (previous_cluster_proposal_log_weight - proposal_log_weight)

        log_uniform_variate = log(rand(Uniform(0, 1), 1)[1])
        if log_uniform_variate < log_acceptance_ratio
            final_cluster = proposed_cluster
        else
            final_cluster = previous_cluster
        end
        add_observation!(observation, final_cluster, instances, iteration, data, sufficient_stats, model, 
                         auxillary_variables)
    end
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       cluster_parameters::NtlParameters{A}) where {A <: GeometricArrivals}
    nothing
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       cluster_parameters::NtlParameters{A}) where {A <: PoissonArrivals}
    last_cluster = findlast(sufficient_stats.cluster.clusters)
    n = sufficient_stats.n
    lower_bound = n - last_cluster 
    posterior_alpha, posterior_beta = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    negative_binomial = NegativeBinomial(posterior_alpha, 1/(1 + posterior_beta))
    lower_bound_cdf = cdf(negative_binomial, lower_bound)
    uniform_variate = rand(Uniform(lower_bound_cdf, 1), 1)[1]
    log_uniform_variate = log(uniform_variate)
    proposed_interarrival_time = lower_bound
    while logcdf(negative_binomial, proposed_interarrival_time) < log_uniform_variate
        proposed_interarrival_time += 1
    end
    auxillary_variables.arrivals.last_arrival_time = last_cluster + proposed_interarrival_time
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       cluster_parameters::DpParameters)
    nothing
end

function gibbs_sample!(auxillary_variables::AuxillaryVariables, sufficient_stats::SufficientStatistics, 
                       data_parameters::Union{GaussianParameters, MultinomialParameters, GaussianWishartParameters},
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

function gibbs_sample!(observation, instances, iteration, data, sufficient_stats, model::Changepoint, auxillary_variables)
    n = sufficient_stats.n
    if observation in [1,n] || (instances[observation-1, iteration] !== instances[observation+1, iteration])
        remove_observation!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)
        (cluster, _) = gibbs_proposal(observation, data, sufficient_stats, model, auxillary_variables)
        add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    end
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

function gibbs_proposal(observation, data, sufficient_stats, model::Changepoint{C, D}, auxillary_variables) where 
                       {C <: DpParameters, D}
    data_parameters = model.data_parameters
    n = sufficient_stats.n

    changepoints = get_changepoints(sufficient_stats, observation)
    oldest_changepoint = minimum(changepoints)
    newest_changepoint = maximum(changepoints)
    num_changepoints = length(changepoints)

    choices = Array{Int64}(undef, num_changepoints+1)

    choices[1:num_changepoints] = changepoints
    choices[num_changepoints+1] = observation
    weights = Array{Float64}(undef, length(choices))

    if observation > 1 && observation < n
        weights[1:num_changepoints+1] .= new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                    auxillary_variables, oldest_changepoint)
        for (index, changepoint) in enumerate(changepoints) 
            weights[index] += existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                              auxillary_variables, observation, changepoint) 
        end
        weights[num_changepoints+1] += new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, 
                                                                  auxillary_variables, observation)
    elseif observation === 1
        @assertion length(choices) === 2
        weights[1] = new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, auxillary_variables, observation)
        weights[2] = existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, auxillary_variables, 
                                                     observation, newest_changepoint)
    elseif observation === n
        @assertion length(choices) === 2
        weights[1] = new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, auxillary_variables, oldest_changepoint)
        weights[2] = existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, auxillary_variables, 
                                                     observation, oldest_changepoint)
    end

    weights += compute_data_log_predictives(observation, choices, data, sufficient_stats.data, data_parameters, auxillary_variables)

    return gumbel_max(choices, weights)
end

function gibbs_proposal(observation, data, sufficient_stats, model::Changepoint{C, D}, auxillary_variables) where 
                        {C <: NtlParameters, D}
    data_parameters = model.data_parameters

    changepoints = get_changepoints(sufficient_stats, observation)
    num_changepoints = length(changepoints)

    choices = Array{Int64}(undef, num_changepoints+1)

    choices[1:num_changepoints] = changepoints
    choices[num_changepoints+1] = observation
    weights = Array{Float64}(undef, length(choices))

    for (index, changepoint) in enumerate(changepoints) 
        weights[index] = existing_cluster_log_predictive(sufficient_stats, model.cluster_parameters, auxillary_variables, 
                                                         observation, changepoint)
    end
    weights[num_changepoints+1] = new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, auxillary_variables, 
                                                             observation)

    weights += compute_data_log_predictives(observation, choices, data, sufficient_stats.data, data_parameters, auxillary_variables)

    return gumbel_max(choices, weights)
end 

function propose(observation, data, particles, particle, sufficient_stats, model::Mixture, auxillary_variables)
    return gibbs_proposal(observation, data, sufficient_stats, model, auxillary_variables)
end

function fit(data, model, sampler::Union{SequentialMonteCarlo, SequentialImportanceSampler})
    num_particles = sampler.num_particles
    n = size(data)[2]
    particles = Array{Int64}(undef, n, num_particles)
    particles .= n+1
    num_particles = size(particles)[2]
    log_likelihoods = zeros(Float64, num_particles)
    log_weights = zeros(Float64, num_particles)

    sufficient_stats_array = prepare_sufficient_statistics(model, data, num_particles)
    auxillary_variables = prepare_auxillary_variables(model, data)

    # Assign first observation to first cluster
    for particle = 1:num_particles
        sufficient_stats = sufficient_stats_array[particle]
        cluster = 1
        observation = 1
        add_observation!(observation, cluster, particles, particle, data, sufficient_stats, model, auxillary_variables)
        sufficient_stats_array[particle] = sufficient_stats
    end
    for observation = ProgressBar(2:n)
        for particle = 1:num_particles
            sufficient_stats = sufficient_stats_array[particle]

            (cluster, log_proposal_weight) = propose(observation, data, particles, particle, sufficient_stats, model,
                                                     auxillary_variables)

            add_observation!(observation, cluster, particles, particle, data, sufficient_stats, model, auxillary_variables)

            compute_log_weight!(log_weights, log_likelihoods, log_proposal_weight, sufficient_stats, model, particle, sampler)

            sufficient_stats_array[particle] = sufficient_stats
        end
        resample!(particles, log_weights, log_likelihoods, sufficient_stats_array, sampler)
    end
    weights = compute_normalized_weights(log_weights)
    return (particles, weights)
end

function resample!(particles, log_weights, log_likelihoods, sufficient_stats_array, sampler::SequentialMonteCarlo)
    ess = effective_sample_size(log_weights)
    n = size(particles)[1] 
    if ess < sampler.ess_threshold*n
        resampled_indices = gumbel_max(sampler.num_particles, log_weights)

        resampled_particles = particles[:, resampled_indices]
        resampled_sufficient_stats_array = sufficient_stats_array[resampled_indices]
        resampled_log_likelihoods = log_likelihoods[resampled_indices]

        n = size(particles)[1]

        for i = 1:sampler.num_particles
            for j = 1:n
                particles[j, i] = resampled_particles[j, i]
            end
            sufficient_stats_array[i] = deepcopy(resampled_sufficient_stats_array[i])
            log_likelihoods[i] = resampled_log_likelihoods[i]
        end
        log_weights .= 0
    end
end

function resample!(particles, log_weights, log_likelihoods, sufficient_stats_array, sampler::SequentialImportanceSampler)
    nothing
end

function prepare_arrivals_auxillary_variables(::DpParameters, data::Matrix{T}) where  
                                             {T <: Real}
    return DpAuxillaryVariables()
end

function prepare_arrivals_auxillary_variables(::NtlParameters{A}, data::Matrix{T}) where 
                                             {T <: Real, A <: PoissonArrivals}
    last_arrival_time = size(data)[2] + 1 
    poisson_aux_variables = PoissonAuxillaryVariables(last_arrival_time)
    return poisson_aux_variables
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

function prepare_data_auxillary_variables(::GaussianWishartParameters, data::Matrix{T}) where {T <: Real}
    return GaussianWishartAuxillaryVariables()
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
    cluster_sufficient_stats = prepare_cluster_sufficient_statistics(model, n)
    data_sufficient_stats = prepare_data_sufficient_statistics(data, model.data_parameters)
    return SufficientStatistics(n, cluster_sufficient_stats, data_sufficient_stats)
end

function prepare_sufficient_statistics(model, data, num_particles)
    sufficient_statistics_array = Array{SufficientStatistics}(undef, num_particles)
    for i=1:num_particles
        sufficient_statistics_array[i] = prepare_sufficient_statistics(model, data)
    end
    return sufficient_statistics_array
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

function prepare_data_sufficient_statistics(data::Matrix{Float64}, data_parameters::GaussianWishartParameters)
    dim = size(data)[1]
    n = size(data)[2]
    data_means = zeros(Float64, dim, n+1)
    data_outer_product_sums = zeros(Float64, dim*dim, n+1)
    posterior_means = Matrix{Float64}(undef, dim, n+1)
    for i = 1:(n+1)
        posterior_means[:, i] = data_parameters.prior_mean
    end
    posterior_scale_matrices = Matrix{Float64}(undef, dim*dim, n+1)
    for i = 1:(n+1)
        posterior_scale_matrices[:, i] = vec(data_parameters.scale_matrix)
    end
    posterior_scales = Vector{Float64}(undef, n+1)
    posterior_scales .= data_parameters.scale
    posterior_dof = Vector{Float64}(undef, n+1)
    posterior_dof .= data_parameters.dof
    return GaussianWishartSufficientStatistics(data_means, data_outer_product_sums, posterior_means, posterior_scale_matrices, 
                                               posterior_scales, posterior_dof)
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

function prepare_cluster_sufficient_statistics(::Changepoint, n::Int64)
    changepoint_num_observations = Vector{Int64}(zeros(Int64, n))
    changepoints = BitArray(undef, n)
    changepoints .= false
    changepoint_sufficient_stats = ChangepointSufficientStatistics(changepoint_num_observations, changepoints)
    return changepoint_sufficient_stats
end

function prepare_cluster_sufficient_statistics(::Mixture, n::Int64)
    cluster_num_observations = Vector{Int64}(zeros(Int64, n))
    cumulative_num_observations = FenwickTree{Int64}(n)
    clusters = BitArray(undef, n)
    clusters .= false
    cluster_sufficient_stats = MixtureSufficientStatistics(cluster_num_observations, cumulative_num_observations, clusters)
    return cluster_sufficient_stats
end

function compute_cluster_data_log_likelihood(cluster::Int64, sufficient_stats::SufficientStatistics,
                                             data_parameters::MultinomialParameters)
    dirichlet_posterior = sufficient_stats.data.posterior_dirichlet_scale[:, cluster]
    dirichlet_prior = data_parameters.prior_dirichlet_scale
    counts = convert(Vector{Int64}, round.(dirichlet_posterior - dirichlet_prior))
    log_likelihood = log_multinomial_coeff(counts) + logmvbeta(dirichlet_posterior) - logmvbeta(dirichlet_prior)
    return log_likelihood
end

function compute_cluster_data_log_likelihood(cluster::Int64, sufficient_stats::SufficientStatistics, 
                                             data_parameters::GaussianParameters) 
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

function compute_cluster_data_log_likelihood(cluster::Int64, sufficient_stats::SufficientStatistics, data_parameters::FiniteTopicModelParameters) 
    return NaN
end

function compute_data_log_likelihood(sufficient_stats::SufficientStatistics, data_parameters::DataParameters) 
    clusters = get_clusters(sufficient_stats.cluster)
    log_likelihood = 0
    for cluster = clusters
        log_likelihood += compute_cluster_data_log_likelihood(cluster, sufficient_stats, data_parameters)
    end
    return log_likelihood
end

function compute_cluster_assignment_log_likelihood(cluster::Int64, cluster_sufficient_stats::MixtureSufficientStatistics, 
                                                   cluster_parameters::NtlParameters{T}) where {T <: ArrivalDistribution}
    psi_posterior = compute_stick_breaking_posterior(cluster, cluster_sufficient_stats, cluster_parameters.prior)
    return logbeta(psi_posterior) - logbeta(cluster_parameters.prior)
end

function compute_arrivals_log_likelihood(sufficient_stats::SufficientStatistics, 
                                         cluster_parameters::NtlParameters{T}) where {T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    return logbeta(phi_posterior) - logbeta(cluster_parameters.arrival_distribution.prior)
end

function compute_arrivals_log_likelihood(sufficient_stats::SufficientStatistics,
                                         cluster_parameters::NtlParameters{T}) where {T <: PoissonArrivals}
    #alpha = cluster_parameters.arrival_distribution.alpha
    #beta = cluster_parameters.arrival_distribution.beta
    #arrival_times = findall(sufficient_stats.cluster.clusters) 
    #last_cluster = maximum(arrival_times)
    #num_clusters = length(arrival_times)
    #interarrivals = diff(arrival_times)
    #log_constant = sum(-loggamma.(interarrivals))
    #log_constant += alpha*log(beta) - loggamma(alpha)

    #n = sufficient_stats.n

    #log_cdf_values = Vector{Float64}(undef, n-last_cluster)
    #for j = 1:(n-last_cluster)
    #    posterior_alpha = last_cluster + j + alpha       
    #    posterior_beta = num_clusters + beta
    #    log_cdf_values[j] = loggamma(posterior_alpha) - posterior_alpha*log(posterior_beta) - loggamma(j)
    #    log_cdf_values[j] += log_constant
    #end
    #posterior_alpha = last_cluster + alpha
    #posterior_beta = num_clusters - 1 + beta
    #likelihood = exp(loggamma(posterior_alpha) - (last_cluster + alpha)*log(posterior_beta) + log_constant)
    #likelihood -= sum(exp.(log_cdf_values)) 
    #return log(likelihood)
    return NaN
end

function compute_assignment_log_likelihood(sufficient_stats::SufficientStatistics{C, D},
                                           cluster_parameters::DpParameters) where {C <: MixtureSufficientStatistics, D}
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
                                           cluster_parameters::ParametricArrivalsClusterParameters{T}) where 
                                           {T, C <: MixtureSufficientStatistics, D}
    log_likelihood = 0
    clusters = get_clusters(sufficient_stats.cluster)
    for cluster = clusters
        log_likelihood += compute_cluster_assignment_log_likelihood(cluster, sufficient_stats.cluster, 
                                                                    cluster_parameters)
    end
    log_likelihood += compute_arrivals_log_likelihood(sufficient_stats, cluster_parameters)
    return log_likelihood
end

function compute_assignment_log_likelihood(sufficient_stats::SufficientStatistics{C, D},
                                           cluster_parameters::ParametricArrivalsClusterParameters{T}) where
                                           {T, C <: HmmSufficientStatistics, D}
    log_likelihood = 0
    clusters = get_clusters(sufficient_stats.cluster)
    cluster_bit_array = sufficient_stats.cluster.clusters
    for source_cluster = clusters
        source_cluster_num_obs = sufficient_stats.cluster.state_num_observations[source_cluster, :]
        source_cluster_sufficient_stats = MixtureSufficientStatistics(source_cluster_num_obs, cluster_bit_array) 
        target_clusters = get_clusters(sufficient_stats.cluster, source_cluster)
        for target_cluster = target_clusters
            log_likelihood += compute_cluster_assignment_log_likelihood(target_cluster, 
                                                                        source_cluster_sufficient_stats, 
                                                                        cluster_parameters)
        end
    end
    log_likelihood += compute_arrivals_log_likelihood(sufficient_stats, cluster_parameters)
    return log_likelihood
end

function compute_assignment_log_likelihood(sufficient_stats::SufficientStatistics{C, D}, 
                                           cluster_parameters::ParametricArrivalsClusterParameters{T}) where 
                                           {T, C <: ChangepointSufficientStatistics, D}
    log_likelihood = compute_arrivals_log_likelihood(sufficient_stats, cluster_parameters)
    return log_likelihood
end

function compute_assignment_log_likelihood(sufficient_stats::SufficientStatistics{C, D}, 
                                           cluster_parameters::DpParameters) where 
                                           {C <: ChangepointSufficientStatistics, D}
    return NaN
end

function compute_joint_log_likelihood(sufficient_stats::SufficientStatistics, model)
    data_parameters = model.data_parameters
    cluster_parameters = model.cluster_parameters
    log_likelihood = compute_data_log_likelihood(sufficient_stats, data_parameters)
    log_likelihood += compute_assignment_log_likelihood(sufficient_stats, cluster_parameters)
    return log_likelihood
end

function compute_log_weight!(log_weights::Vector{Float64}, log_likelihoods::Vector{Float64}, 
                             proposal_log_weight::Float64, sufficient_stats::SufficientStatistics,
                             model, particle::Int64, ::SequentialImportanceSampler)
    previous_log_weight = log_weights[particle]
    previous_log_likelihood = log_likelihoods[particle]
    log_likelihood = compute_joint_log_likelihood(sufficient_stats, model)
    log_weight = previous_log_weight + log_likelihood - previous_log_likelihood - proposal_log_weight   
    log_weights[particle] = log_weight
    log_likelihoods[particle] = log_likelihood
end

function compute_log_weight!(log_weights::Vector{Float64}, log_likelihoods::Vector{Float64}, 
                             proposal_log_weight::Float64, sufficient_stats::SufficientStatistics,
                             model, particle::Int64, ::SequentialMonteCarlo)
    previous_log_likelihood = log_likelihoods[particle]
    log_likelihood = compute_joint_log_likelihood(sufficient_stats, model)
    log_weight = log_likelihood - previous_log_likelihood - proposal_log_weight   
    log_weights[particle] = log_weight
    log_likelihoods[particle] = log_likelihood
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

function update_data_sufficient_statistics!(sufficient_stats::SufficientStatistics{C,D}, cluster::Int64, 
                                            observation::Int64, model::Model, data, auxillary_variables;
                                            update_type::String="add") where {C, D<:GaussianWishartSufficientStatistics}
    datum = vec(data[:, observation])
    cluster_mean = vec(sufficient_stats.data.data_means[:, cluster])
    dim = length(cluster_mean)
    data_outer_product_sum = reshape(sufficient_stats.data.data_outer_product_sums[:, cluster], dim, dim)
    n = sufficient_stats.cluster.num_observations[cluster]
    datum_outer_product = datum*transpose(datum)
    if update_type == "add"
        cluster_mean = (cluster_mean.*(n-1) + datum)./n
        data_outer_product_sum += datum_outer_product
    elseif update_type == "remove"
        if n > 0
            cluster_mean = (cluster_mean.*(n+1) - datum)./n
            data_outer_product_sum -= datum_outer_product
        else
            cluster_mean .= 0
            data_outer_product_sum .= 0
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
    sufficient_stats.data.data_means[:, cluster] = cluster_mean
    sufficient_stats.data.data_outer_product_sums[:, cluster] = vec(data_outer_product_sum)
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

function update_data_sufficient_statistics!(sufficient_stats::SufficientStatistics{C, D}, clusters::Vector{Int64}) where 
                                           {C, D <: MultinomialSufficientStatistics}
    @assertion length(clusters) === 2
    new_cluster_counts = sum(sufficient_stats.data.total_counts[:, clusters], dims=2)
    n = sufficient_stats.n
    sufficient_stats.data.total_counts[:, n+1] = new_cluster_counts
end

function update_cluster_sufficient_statistics!(sufficient_stats::SufficientStatistics{C, D}, cluster::Int64; 
                                               update_type::String="add") where {C <: MixtureSufficientStatistics, D}
    if update_type === "add"
        sufficient_stats.cluster.num_observations[cluster] += 1
        inc!(sufficient_stats.cluster.cumulative_num_observations, cluster, 1)
        if sufficient_stats.cluster.num_observations[cluster] === 1
            sufficient_stats.cluster.clusters[cluster] = true
        end
    elseif update_type === "remove"
        sufficient_stats.cluster.num_observations[cluster] -= 1
        dec!(sufficient_stats.cluster.cumulative_num_observations, cluster, 1)
        if sufficient_stats.cluster.num_observations[cluster] === 0
            sufficient_stats.cluster.clusters[cluster] = false
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

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::MixtureSufficientStatistics,
                                              new_time::Int64, old_time::Int64)
    num_observations = cluster_sufficient_stats.num_observations[old_time]
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats.num_observations, new_time, old_time)
    cluster_sufficient_stats.clusters[new_time] = true
    cluster_sufficient_stats.clusters[old_time] = false
    dec!(cluster_sufficient_stats.cumulative_num_observations, old_time, num_observations)
    inc!(cluster_sufficient_stats.cumulative_num_observations, new_time, num_observations)
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::ChangepointSufficientStatistics,
                                              new_time::Int64, old_time::Int64)
    update_cluster_stats_new_birth_time!(cluster_sufficient_stats.num_observations, new_time, old_time)
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

function update_data_stats_new_birth_time!(data_sufficient_stats::GaussianWishartSufficientStatistics,
                                           new_time, old_time, data_parameters::GaussianWishartParameters)
    data_sufficient_stats.data_means[:, new_time] = data_sufficient_stats.data_means[:, old_time]
    data_sufficient_stats.data_means[:, old_time] .= 0
    data_sufficient_stats.data_outer_product_sums[:, new_time] = data_sufficient_stats.data_outer_product_sums[:, old_time]
    data_sufficient_stats.data_outer_product_sums[:, old_time] .= 0
    data_sufficient_stats.posterior_means[:, new_time] = data_sufficient_stats.posterior_means[:, old_time]
    data_sufficient_stats.posterior_means[:, old_time] = data_parameters.prior_mean
    data_sufficient_stats.posterior_scale_matrix[:, new_time] = data_sufficient_stats.posterior_scale_matrix[:, old_time]
    data_sufficient_stats.posterior_scale_matrix[:, old_time] = vec(data_parameters.scale_matrix)
    data_sufficient_stats.posterior_scale[new_time] = data_sufficient_stats.posterior_scale[old_time]
    data_sufficient_stats.posterior_scale[old_time] = data_parameters.scale
    data_sufficient_stats.posterior_dof[new_time] = data_sufficient_stats.posterior_dof[old_time]
    data_sufficient_stats.posterior_dof[old_time] = data_parameters.dof
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
                             sufficient_stats::SufficientStatistics, model::Union{Mixture, Changepoint},
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
                          data::Matrix{T}, sufficient_stats::SufficientStatistics, model::Union{Mixture, Changepoint},
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
                                                {T <: PoissonArrivals, C, D}
    last_cluster = findlast(sufficient_stats.cluster.clusters)
    num_clusters = count(sufficient_stats.cluster.clusters)
    posterior_alpha = cluster_parameters.arrival_distribution.alpha + last_cluster - 1
    posterior_beta = cluster_parameters.arrival_distribution.beta + num_clusters
    return (posterior_alpha, posterior_beta)
end

function compute_arrival_distribution_posterior(sufficient_stats::SufficientStatistics{C, D},
                                                cluster_parameters::ParametricArrivalsClusterParameters{T}) where 
                                                {T <: GeometricArrivals, C <: Union{MixtureSufficientStatistics, HmmSufficientStatistics}, D}
    n = sufficient_stats.n
    num_clusters = length(get_clusters(sufficient_stats.cluster))
    phi_posterior = deepcopy(cluster_parameters.arrival_distribution.prior)
    phi_posterior[1] += num_clusters - 1
    phi_posterior[2] += n - num_clusters 
    return phi_posterior
end

function compute_arrival_distribution_posterior(sufficient_stats::SufficientStatistics{C, D},
                                                cluster_parameters::ParametricArrivalsClusterParameters{T}) where 
                                                {T <: GeometricArrivals, C <: ChangepointSufficientStatistics, D}
    n = sufficient_stats.n
    num_changepoints = length(get_changepoints(sufficient_stats.cluster))
    phi_posterior = deepcopy(cluster_parameters.arrival_distribution.prior)
    phi_posterior[1] += num_changepoints - 1
    phi_posterior[2] += n - num_changepoints 
    return phi_posterior
end

### Changepoint log predictives

function new_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::DpParameters, 
                                    auxillary_variables, changepoint) where {C <: ChangepointSufficientStatistics, D}
    num_obs = maximum([sufficient_stats.cluster.num_observations[changepoint] - 1, 0])
    denominator = num_obs + cluster_parameters.alpha + cluster_parameters.beta
    return log(cluster_parameters.alpha) - log(denominator)
end

function existing_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::DpParameters, 
                                         auxillary_variables, observation, changepoint) where 
                                         {C <: ChangepointSufficientStatistics, D}
    num_obs = maximum([sufficient_stats.cluster.num_observations[changepoint] - 1, 0])
    numerator = num_obs + cluster_parameters.beta
    return log(numerator) - log(numerator + cluster_parameters.alpha)
end

function new_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{T},
                                    auxillary_variables, observation) where 
                                    {C <: ChangepointSufficientStatistics, D, T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    return log(phi_posterior[1]) - log(sum(phi_posterior))
end

function existing_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{T},
                                         auxillary_variables, observation, changepoint) where 
                                         {C <: ChangepointSufficientStatistics, D, T <: GeometricArrivals}
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    return log(phi_posterior[2]) - log(sum(phi_posterior))
end

function new_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{A}, 
                                    auxillary_variables, observation::Int64) where 
                                    {A <: PoissonArrivals, C <: ChangepointSufficientStatistics, D}
    n = sufficient_stats.n
    last_arrival_time = auxillary_variables.arrivals.last_arrival_time
    num_clusters = count(sufficient_stats.cluster.clusters)

    next_changepoint = observation + 1
    if next_changepoint === findlast(sufficient_stats.cluster.clusters) || (next_changepoint === n + 1) 
        changepoint_after_next = last_arrival_time
    else
        changepoint_after_next = findfirst(sufficient_stats.cluster.clusters .& (1:n .> next_changepoint))
    end
    next_interarrival = changepoint_after_next - next_changepoint

    if observation === 1
        previous_changepoint = 0
    else
        previous_changepoint = findlast(sufficient_stats.cluster.clusters .& (1:n .< observation))
    end
    previous_interarrival = observation - previous_changepoint

    posterior_alpha = last_arrival_time - 1 + cluster_parameters.arrival_distribution.alpha
    posterior_beta = num_clusters + 2 + cluster_parameters.arrival_distribution.beta 
    log_predictive = -posterior_alpha*log(posterior_beta)
    log_predictive += -(logfactorial(next_interarrival - 1) + logfactorial(previous_interarrival - 1))

    return log_predictive
end

function existing_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{A}, 
                                         auxillary_variables, observation::Int64, changepoint::Int64) where 
                                         {A <: PoissonArrivals, C <: ChangepointSufficientStatistics, D}
    n = sufficient_stats.n
    last_arrival_time = auxillary_variables.arrivals.last_arrival_time
    num_clusters = count(sufficient_stats.cluster.clusters)
    posterior_alpha = last_arrival_time - 1 + cluster_parameters.arrival_distribution.alpha 
    posterior_beta = num_clusters + 1 + cluster_parameters.arrival_distribution.beta 

    log_predictive = -posterior_alpha*log(posterior_beta)

    if changepoint - 1 === observation
        if changepoint === findlast(sufficient_stats.cluster.clusters)
            changepoint_after_next = last_arrival_time
        else
            changepoint_after_next = findfirst(sufficient_stats.cluster.clusters .& (1:n .> changepoint))
        end
        next_interarrival = changepoint_after_next - observation

        if observation === 1
            previous_changepoint = 0
        else
            previous_changepoint = findlast(sufficient_stats.cluster.clusters .& (1:n .< observation))
        end
        prev_interarrival = observation - previous_changepoint
    else
        next_changepoint = observation + 1
        if next_changepoint === findlast(sufficient_stats.cluster.clusters) || (next_changepoint === n + 1)
            changepoint_after_next = last_arrival_time
        else
            changepoint_after_next = findfirst(sufficient_stats.cluster.clusters .& (1:n .> next_changepoint))
        end
        next_interarrival = changepoint_after_next - next_changepoint

        prev_interarrival = next_changepoint - changepoint
    end
    log_predictive += -(logfactorial(next_interarrival - 1) + logfactorial(prev_interarrival - 1))

    return log_predictive
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
    return log(phi_posterior[1]) - log(sum(phi_posterior)) + logbeta(psi_posterior)
end

function existing_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D},
                                         cluster_parameters::ParametricArrivalsClusterParameters{T}, 
                                         auxillary_variables, observation::Int64, cluster::Int64) where 
                                         {T <: GeometricArrivals, C <: MixtureSufficientStatistics, D}
    phi_posterior = compute_arrival_distribution_posterior(sufficient_stats, cluster_parameters)
    return log(phi_posterior[2]) - log(sum(phi_posterior))
end

function new_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{A}, 
                                    auxillary_variables, observation::Int64) where 
                                    {A <: PoissonArrivals, C <: MixtureSufficientStatistics, D}
    n = sufficient_stats.n
    last_arrival_time = auxillary_variables.arrivals.last_arrival_time
    num_clusters = count(sufficient_stats.cluster.clusters)

    if observation === 1
        prev_arrival_time = observation
    else
        prev_arrival_time = findlast(sufficient_stats.cluster.clusters .& (observation .>= 1:n)) 
    end
    if observation > findlast(sufficient_stats.cluster.clusters)
        next_arrival_time = last_arrival_time
    else
        next_arrival_time = findfirst(sufficient_stats.cluster.clusters .& (observation .<= 1:n))
    end

    posterior_alpha = last_arrival_time - 1 + cluster_parameters.arrival_distribution.alpha
    posterior_beta = num_clusters + 2 + cluster_parameters.arrival_distribution.beta 
    log_predictive = -posterior_alpha*log(posterior_beta)

    if observation === 1 
        log_predictive += -logfactorial(next_arrival_time - observation - 1)
    else 
        log_predictive += logfactorial(next_arrival_time - prev_arrival_time - 1)
        log_predictive += -(logfactorial(observation - prev_arrival_time - 1) + logfactorial(next_arrival_time - observation - 1))
    end
    return log_predictive
end

function existing_cluster_log_predictive(sufficient_stats::SufficientStatistics{C, D}, cluster_parameters::NtlParameters{A}, 
                                         auxillary_variables, observation::Int64, cluster::Int64) where 
                                         {A <: PoissonArrivals, C <: MixtureSufficientStatistics, D}
    n = sufficient_stats.n
    last_arrival_time = auxillary_variables.arrivals.last_arrival_time
    num_clusters = count(sufficient_stats.cluster.clusters)
    posterior_alpha = last_arrival_time - 1 + cluster_parameters.arrival_distribution.alpha 
    posterior_beta = num_clusters + 1 + cluster_parameters.arrival_distribution.beta 

    log_predictive = -posterior_alpha*log(posterior_beta)

    if observation < cluster 
        if observation === 1
            prior_to_observation = 0
        else
            prior_to_observation = findlast(sufficient_stats.cluster.clusters .& (observation .>= 1:n)) 
        end
        if observation > findlast(sufficient_stats.cluster.clusters)
            after_observation = last_arrival_time
        else
            after_observation = findfirst(sufficient_stats.cluster.clusters .& (observation .<= 1:n))
        end
        log_predictive += logfactorial(after_observation - prior_to_observation - 1) 
        log_predictive += -(logfactorial(after_observation - observation - 1) + logfactorial(observation - prior_to_observation - 1))

        if cluster === 2
            prior_to_cluster = 2  
        else
            prior_to_cluster = findlast(sufficient_stats.cluster.clusters .& (cluster .> 1:n))
        end
        if cluster === findlast(sufficient_stats.cluster.clusters)
            after_cluster = last_arrival_time
        else
            after_cluster = findfirst(sufficient_stats.cluster.clusters .& (cluster .< 1:n))
        end

        log_predictive += logfactorial(after_cluster - prior_to_cluster - 1)
        if observation === 1
            prior_to_cluster = 0
            cluster = 1
        end
        log_predictive += -(logfactorial(after_cluster - cluster - 1) + logfactorial(cluster - prior_to_cluster - 1))
    end
    return log_predictive
end

### End mixture cluster log predictives

function compute_num_complement(cluster::Int64, cluster_suff_stats::MixtureSufficientStatistics;
                                missing_observation::Union{Int64, Nothing}=nothing)
    sum_of_older_clusters = prefixsum(cluster_suff_stats.cumulative_num_observations, cluster-1)
    num_complement = sum_of_older_clusters - (cluster - 1)
    if missing_observation !== nothing && missing_observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_stick_breaking_posterior(cluster::Int64, cluster_suff_stats::MixtureSufficientStatistics, beta_prior::Vector{T};
                                          missing_observation::Union{Int64, Nothing}=nothing) where {T <: Real}
    posterior = deepcopy(beta_prior)
    posterior[1] += cluster_suff_stats.num_observations[cluster] - 1
    posterior[2] += compute_num_complement(cluster, cluster_suff_stats, missing_observation=missing_observation)
    posterior[posterior .<= 0] .= 1
    return posterior
end

function compute_stick_breaking_posterior(cluster::Int64, sufficient_stats::SufficientStatistics, 
                                          ntl_parameters::ParametricArrivalsClusterParameters; 
                                          missing_observation::Union{Int64, Nothing}=nothing)
    posterior = compute_stick_breaking_posterior(cluster, sufficient_stats.cluster, 
                                                 ntl_parameters.prior, missing_observation=missing_observation)
    return posterior
end

function compute_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, 
                                         sufficient_stats, ntl_parameters::NtlParameters)
    # Not strictly the number of observations
    ntl_sufficient_stats = sufficient_stats.cluster
    num_clusters = length(clusters)
    log_weights = Vector{Float64}(undef, num_clusters)
    n = maximum(clusters)
    n = maximum([n, observation])
    cluster_log_weights = zeros(Float64, n)
    complement_log_weights = zeros(Float64, n)
    complement_log_weights_tree = FenwickTree{Float64}(n)
    psi_parameters = Array{Int64}(undef, 2, n)
    logbetas = Array{Float64}(undef, n)
    new_num_complement = compute_num_complement(observation, ntl_sufficient_stats, missing_observation=observation)
    younger_cluster_new_psi = Array{Float64}(undef, 2)
    cluster_new_psi = Array{Float64}(undef, 2)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_stick_breaking_posterior(cluster, sufficient_stats, ntl_parameters, 
                                                           missing_observation=observation)
            psi_parameters[:, cluster] = cluster_psi
            log_denom = log(sum(cluster_psi))
            cluster_log_weights[cluster] = log(cluster_psi[1]) - log_denom
            complement_log_weights[cluster] = log(cluster_psi[2]) - log_denom
            inc!(complement_log_weights_tree, cluster, complement_log_weights[cluster])
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
            log_weight += prefixsum(complement_log_weights_tree, observation-1) - prefixsum(complement_log_weights_tree, cluster)
            #log_weight += sum(complement_log_weights[younger_clusters])
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

function compute_existing_cluster_log_predictives(observation::Int64, clusters::Vector{Int64},
                                                  sufficient_stats::SufficientStatistics, 
                                                  beta_ntl_parameters::BetaNtlParameters,
                                                  auxillary_variables)
    n = sufficient_stats.n
    num_clusters = count(sufficient_stats.cluster.clusters[1:observation])
    num_observations = deepcopy(sufficient_stats.cluster.num_observations[clusters])
    num_observations[num_observations .< 1] .= 1
    log_weights = log.(num_observations .- beta_ntl_parameters.alpha)
    log_weights .-= log(n - num_clusters*beta_ntl_parameters.alpha)
    log_weights .+= existing_cluster_log_predictive(sufficient_stats, beta_ntl_parameters, observation, auxillary_variables)
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

function compute_data_posterior_parameters!(cluster, sufficient_stats, data_parameters::GaussianWishartParameters)
    data_mean = vec(sufficient_stats.data.data_means[:, cluster])
    dim = length(data_mean)
    data_outer_product_sum = reshape(sufficient_stats.data.data_outer_product_sums[:, cluster], dim, dim)
    n = sufficient_stats.cluster.num_observations[cluster]
    prior_mean = data_parameters.prior_mean
    scale_matrix = data_parameters.scale_matrix
    scale = data_parameters.scale
    dof = data_parameters.dof

    posterior_scale = scale + n
    posterior_mean = (scale*prior_mean + n*data_mean)/posterior_scale
    C = data_outer_product_sum - n*data_mean*transpose(data_mean)
    inverse_scale = inv(scale_matrix)
    posterior_inverse_scale_matrix = inverse_scale + C + (scale*n)*(data_mean - prior_mean)*transpose(data_mean - prior_mean)/(posterior_scale)
    posterior_scale_matrix = inv(posterior_inverse_scale_matrix)
    posterior_dof = dof + n
    sufficient_stats.data.posterior_means[:, cluster] = vec(posterior_mean)
    sufficient_stats.data.posterior_scale_matrix[:, cluster] = vec(posterior_scale_matrix)
    sufficient_stats.data.posterior_scale[cluster] = posterior_scale
    sufficient_stats.data.posterior_dof[cluster] = posterior_dof
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

function data_log_predictive(observation::Int64, cluster::Int64, data_sufficient_stats::GaussianWishartSufficientStatistics, 
                             data_parameters::GaussianWishartParameters, data, auxillary_variables)
    datum = vec(data[:, observation])
    posterior_mean = vec(data_sufficient_stats.posterior_means[:, cluster])
    dim = length(posterior_mean)
    posterior_scale_matrix = reshape(data_sufficient_stats.posterior_scale_matrix[:, cluster], dim, dim)
    posterior_scale = data_sufficient_stats.posterior_scale[cluster]
    posterior_dof = data_sufficient_stats.posterior_dof[cluster]
    mat = (posterior_scale + 1)*posterior_scale_matrix/(posterior_scale*(posterior_dof - dim + 1))
    return mvt_logpdf(posterior_mean, mat, posterior_dof - dim + 1, datum)
end

function data_log_predictive(observation::Int64, cluster::Int64, 
                             data_sufficient_stats::MultinomialSufficientStatistics,
                             ::MultinomialParameters, data, auxillary_variables)
    datum = vec(data[:, observation])
    posterior_scale = vec(data_sufficient_stats.posterior_dirichlet_scale[:, cluster])
    n = sum(datum)
    posterior = DirichletMultinomial(n, posterior_scale)
    return logpdf(posterior, datum)
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

get_clusters(cluster_sufficient_stats::HmmSufficientStatistics) = findall(cluster_sufficient_stats.clusters)

function get_clusters(cluster_sufficient_stats::HmmSufficientStatistics, cluster) 
    return findall(cluster_sufficient_stats.state_num_observations[cluster, :] .> 0)
end

get_changepoints(changepoint_suff_stats::ChangepointSufficientStatistics) = findall(changepoint_suff_stats.clusters)

function get_changepoints(sufficient_stats::SufficientStatistics, observation::Int64)
    n = sufficient_stats.n
    all_changepoints = findall(sufficient_stats.cluster.num_observations .> 0)
    changepoints = Int64[]
    if observation > 1
        first_changepoint = all_changepoints[findlast(all_changepoints .< observation)]
        append!(changepoints, first_changepoint)
    end
    if observation < n
        last_changepoint_index = findfirst(all_changepoints .> observation)
        if !isnothing(last_changepoint_index)
            last_changepoint = all_changepoints[last_changepoint_index]
            append!(changepoints, last_changepoint)
        end
    end
    return Vector{Int64}(changepoints)
end

### HIDDEN MARKOV MODEL

function gibbs_sample!(observation, instances, iteration, data, sufficient_stats, model::HiddenMarkovModel, auxillary_variables)
    n = size(data)[2]
    remove_observation!(observation, instances, iteration, data, sufficient_stats, model, auxillary_variables)
    previous_state = (observation > 1) ? instances[observation-1, iteration] : n+1
    next_state = (observation < n) ? instances[observation+1, iteration] : n+1
    (cluster, _) = gibbs_proposal(observation, data, sufficient_stats, model, previous_state, next_state, auxillary_variables)
    add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, auxillary_variables, 
                     update_next_cluster=true)
end

function gibbs_proposal(observation, data, sufficient_stats, model, previous_state, next_state, auxillary_variables)
    clusters = get_clusters(sufficient_stats.cluster)
    num_clusters = length(clusters)

    choices = Array{Int64}(undef, num_clusters+1)
    choices[1:num_clusters] = clusters
    choices[num_clusters+1] = observation

    weights = zeros(Float64, num_clusters+1)
    # Add contribution from transitioning from previous state
    n = size(data)[2]
    if previous_state !== n+1
        weights[1:num_clusters] = compute_existing_cluster_log_predictives(observation, clusters, previous_state, 
                                                                           sufficient_stats, model.cluster_parameters,
                                                                           auxillary_variables)
        weights[num_clusters+1] = new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, observation, auxillary_variables)
    end
    @assertion !any(isnan.(weights))
    # Add contribution from transitioning into next state 
    if next_state !== n+1
        for (index, choice) = enumerate(choices)
            # TODO: make this dispatch on the boolean value
            if next_state === observation+1
                weights[index] += new_cluster_log_predictive(sufficient_stats, model.cluster_parameters, observation, auxillary_variables)
            else
                weights[index] += compute_existing_cluster_log_predictives(observation, next_state, choice, 
                                                                           sufficient_stats, model.cluster_parameters,
                                                                           auxillary_variables)
            end
        end
    end
    @assertion !any(isnan.(weights))
    # Add contribution from data
    weights += compute_data_log_predictives(observation, choices, data, sufficient_stats.data, model.data_parameters, auxillary_variables)
    @assertion !any(isnan.(weights))

    return gumbel_max(choices, weights)
end

function propose(observation, data, particles, particle, sufficient_stats, model::HiddenMarkovModel, auxillary_variables)
    previous_state = particles[observation-1, particle]
    next_state = size(data)[2] + 1
    return gibbs_proposal(observation, data, sufficient_stats, model, previous_state, next_state, auxillary_variables)
end

function prepare_cluster_sufficient_statistics(::HiddenMarkovModel{C, D}, n) where {C <: BetaNtlParameters, D}
    num_observations = Vector{Int64}(zeros(Int64, n))
    # The null state is represented as n+1
    state_num_observations = Matrix{Int64}(zeros(Int64, n+1, n))
    clusters = BitArray(undef, n)
    clusters .= false
    return StationaryHmmSufficientStatistics(num_observations, state_num_observations, clusters)
end

function prepare_cluster_sufficient_statistics(::HiddenMarkovModel{C, D}, n) where {C <: NtlParameters, D}
    num_observations = Vector{Int64}(zeros(Int64, n))
    state_num_observations = Matrix{Int64}(zeros(Int64, n+1, n))
    state_assignments = Vector{SparseMatrixCSC}(undef, n+1)
    # TODO: this might need to be modified to account for null state n+1
    for i = 1:(n+1)
        state_assignments[i] = spzeros(Int64, n+1, n+1)
    end
    clusters = BitArray(undef, n)
    clusters .= false
    return NonstationaryHmmSufficientStatistics(num_observations, state_num_observations, state_assignments, clusters)
end

function update_cluster_sufficient_statistics!(sufficient_stats::SufficientStatistics{C, D}, cluster, source_cluster, 
                                               ::Int64; update_type::String="add") where 
                                               {C <: StationaryHmmSufficientStatistics, D}
    if update_type === "add"
        sufficient_stats.cluster.state_num_observations[source_cluster, cluster] += 1
        sufficient_stats.cluster.num_observations[cluster] += 1
        if sufficient_stats.cluster.num_observations[cluster] === 1
            sufficient_stats.cluster.clusters[cluster] = true
        end
    elseif update_type === "remove"
        sufficient_stats.cluster.state_num_observations[source_cluster, cluster] -= 1
        sufficient_stats.cluster.num_observations[cluster] -= 1
        if sufficient_stats.cluster.num_observations[cluster] === 0
            sufficient_stats.cluster.clusters[cluster] = false
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
end

function update_cluster_sufficient_statistics!(sufficient_stats::SufficientStatistics{C, D}, cluster, source_cluster, 
                                               observation::Int64; update_type::String="add") where 
                                               {C <: NonstationaryHmmSufficientStatistics, D}
    if update_type === "add"
        sufficient_stats.cluster.state_num_observations[source_cluster, cluster] += 1
        sufficient_stats.cluster.state_assignments[source_cluster][observation, cluster] += 1
        sufficient_stats.cluster.num_observations[cluster] += 1
        if sufficient_stats.cluster.num_observations[cluster] === 1
            sufficient_stats.cluster.clusters[cluster] = true
        end
    elseif update_type === "remove"
        sufficient_stats.cluster.state_num_observations[source_cluster, cluster] -= 1
        sufficient_stats.cluster.state_assignments[source_cluster][observation, cluster] -= 1
        sufficient_stats.cluster.num_observations[cluster] -= 1
        if sufficient_stats.cluster.num_observations[cluster] === 0
            sufficient_stats.cluster.clusters[cluster] = false
        end
    else
        message = "$update_type is not a supported update type."
        throw(ArgumentError(message))
    end
    @assertion sufficient_stats.cluster.state_num_observations[source_cluster, cluster] >= 0
    @assertion sufficient_stats.cluster.num_observations[cluster] >= 0
    @assertion sufficient_stats.cluster.state_assignments[source_cluster][observation, cluster] >= 0
end

function update_cluster_stats_new_birth_time!(num_observations::Vector{Int64}, new_time::Int64, old_time::Int64)
    num_observations[new_time] = num_observations[old_time]
    num_observations[old_time] = 0
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::StationaryHmmSufficientStatistics, new_time, old_time)
    @assertion any(cluster_sufficient_stats.state_num_observations[:, old_time] .> 0)
    @assertion all(cluster_sufficient_stats.state_num_observations[new_time, :] .=== 0)
    @assertion all(cluster_sufficient_stats.state_num_observations[:, new_time] .=== 0)
    @assertion !cluster_sufficient_stats.clusters[new_time]
    @assertion cluster_sufficient_stats.clusters[old_time]

    num_obs = deepcopy(cluster_sufficient_stats.state_num_observations[old_time, :])
    cluster_sufficient_stats.state_num_observations[new_time, :] = num_obs 
    cluster_sufficient_stats.state_num_observations[old_time, :] .= 0
    cluster_sufficient_stats.clusters[new_time] = true
    cluster_sufficient_stats.clusters[old_time] = false
    # TODO: fix this cheap hack
    n = length(cluster_sufficient_stats.clusters)
    clusters = get_clusters(cluster_sufficient_stats)
    append!(clusters, n+1)
    for cluster = clusters
        cluster_sufficient_stats.state_num_observations[cluster, new_time] = cluster_sufficient_stats.state_num_observations[cluster, old_time]
        cluster_sufficient_stats.state_num_observations[cluster, old_time] = 0
    end
    
    cluster_sufficient_stats.num_observations[new_time] = cluster_sufficient_stats.num_observations[old_time]
    cluster_sufficient_stats.num_observations[old_time] = 0

    @assertion all(cluster_sufficient_stats.state_num_observations[old_time, :] .=== 0)
    @assertion all(cluster_sufficient_stats.state_num_observations[:, old_time] .=== 0)
    @assertion any(cluster_sufficient_stats.state_num_observations[:, new_time] .> 0)
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::NonstationaryHmmSufficientStatistics, new_time, old_time)
    num_obs = deepcopy(cluster_sufficient_stats.state_num_observations[old_time, :])
    cluster_sufficient_stats.state_num_observations[new_time, :] = num_obs 
    cluster_sufficient_stats.state_num_observations[old_time, :] .= 0
    # Modify the state assignments
    cluster_sufficient_stats.state_assignments[new_time][:, :] = deepcopy(cluster_sufficient_stats.state_assignments[old_time][:, :])
    cluster_sufficient_stats.state_assignments[old_time][:, :] .= 0

    cluster_sufficient_stats.clusters[new_time] = true
    cluster_sufficient_stats.clusters[old_time] = false
    # TODO: fix this cheap hack
    n = length(cluster_sufficient_stats.clusters)
    clusters = get_clusters(cluster_sufficient_stats)
    append!(clusters, n+1)
    # Modify the state num observations
    for cluster = clusters
        cluster_sufficient_stats.state_num_observations[cluster, new_time] = cluster_sufficient_stats.state_num_observations[cluster, old_time]
        cluster_sufficient_stats.state_num_observations[cluster, old_time] = 0
        old_time_state_assignments = deepcopy(cluster_sufficient_stats.state_assignments[cluster][:, old_time])
        cluster_sufficient_stats.state_assignments[cluster][:, new_time] = old_time_state_assignments 
        cluster_sufficient_stats.state_assignments[cluster][:, old_time] .= 0
        @assertion all(cluster_sufficient_stats.state_assignments[cluster][:, old_time] .=== 0) 
    end
    cluster_sufficient_stats.num_observations[new_time] = cluster_sufficient_stats.num_observations[old_time]
    cluster_sufficient_stats.num_observations[old_time] = 0
    @assertion all(cluster_sufficient_stats.state_num_observations[old_time, :] .=== 0)
    @assertion all(cluster_sufficient_stats.state_num_observations[:, old_time] .=== 0)
    @assertion any(cluster_sufficient_stats.state_num_observations[:, new_time] .> 0)
    @assertion any(cluster_sufficient_stats.state_assignments[new_time][:, :] .> 0)
    @assertion all(cluster_sufficient_stats.state_assignments[old_time][:, :] .=== 0)
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             sufficient_stats::SufficientStatistics, model::HiddenMarkovModel, 
                             auxillary_variables::AuxillaryVariables) where {T <: Real}
    n = size(instances)[1]
    cluster = instances[observation, iteration]
    instances[observation, iteration] = n+1 

    previous_cluster = observation > 1 ? instances[observation-1, iteration] : n+1
    update_cluster_sufficient_statistics!(sufficient_stats, cluster, previous_cluster, observation, 
                                          update_type="remove")
    @assertion all(sufficient_stats.cluster.state_num_observations .>= 0)

    if (cluster === observation) && sufficient_stats.cluster.num_observations[cluster] > 0
        cluster = update_cluster_birth_time_remove!(cluster, iteration, instances, sufficient_stats, 
                                                    model.data_parameters)
    end
    # Update cluster book keeping for the next cluster over
    if observation < n
        next_cluster = instances[observation+1, iteration]
        update_cluster_sufficient_statistics!(sufficient_stats, next_cluster, cluster, observation+1, update_type="remove")
        update_cluster_sufficient_statistics!(sufficient_stats, next_cluster, n+1, observation+1, update_type="add")
    end
    update_data_sufficient_statistics!(sufficient_stats, cluster, observation, model, data, auxillary_variables, update_type="remove")
    compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
end

function add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model::HiddenMarkovModel,
                          auxillary_variables::AuxillaryVariables; update_next_cluster=false)
    if observation < cluster
        cluster = update_cluster_birth_time_add!(cluster, observation, iteration, instances, sufficient_stats, 
                                                 model.data_parameters)
    end
    instances[observation, iteration] = cluster
    n = size(instances)[1]
    previous_cluster = observation > 1 ? instances[observation-1, iteration] : n+1
        # Update cluster book keeping for the current observation
    update_cluster_sufficient_statistics!(sufficient_stats, cluster, previous_cluster, observation, update_type="add")
    # Update cluster book keeping for the next observation over
    if update_next_cluster && observation < n 
        next_cluster = instances[observation+1, iteration]
        update_cluster_sufficient_statistics!(sufficient_stats, next_cluster, n+1, observation+1, update_type="remove")
        update_cluster_sufficient_statistics!(sufficient_stats, next_cluster, cluster, observation+1, update_type="add")
    end
    datum = vec(data[:, observation])
    update_data_sufficient_statistics!(sufficient_stats, cluster, observation, model, data, auxillary_variables, update_type="add")
    compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
    @assertion all(sufficient_stats.cluster.state_num_observations .>= 0)
end

function compute_num_complement(cluster, previous_state, hmm_sufficient_stats::NonstationaryHmmSufficientStatistics)
    num_complement = 0
    state_assignments = hmm_sufficient_stats.state_assignments[previous_state]
    for older_cluster=1:(cluster-1)
        num_complement += sum(state_assignments[cluster:end-1, older_cluster]) 
    end
    return num_complement
end

function compute_stick_breaking_posterior(cluster, prev_cluster, hmm_sufficient_stats::NonstationaryHmmSufficientStatistics,
                                          ntl_parameters)
    posterior = deepcopy(ntl_parameters.prior)
    posterior[1] += hmm_sufficient_stats.state_num_observations[prev_cluster, cluster] - 1.
    if posterior[1] < 0.
        posterior[1] += 1.
    end
    posterior[2] += compute_num_complement(cluster, prev_cluster, hmm_sufficient_stats)
    posterior[posterior .=== 0.] .= 1.
    return posterior
end

# Note: this only computes the cluster log predictice for the older clusters, younger cluster log predictives
# are too computationally inefficient to compute
function compute_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, previous_cluster::Int64, 
                                         cluster_sufficient_stats, cluster_parameters::NtlParameters)
    @assertion all(clusters .< observation)
    # Not strictly the number of observations
    ntl_sufficient_stats = cluster_sufficient_stats.cluster
    num_clusters = length(clusters)
    log_weights = Vector{Float64}(undef, num_clusters)
    n = maximum(clusters)
    n = maximum([n, observation])
    cluster_log_weights = zeros(Float64, n)
    complement_log_weights = zeros(Float64, n)

    for cluster = clusters
        if cluster > 1
            cluster_psi = compute_stick_breaking_posterior(cluster, previous_cluster, ntl_sufficient_stats, cluster_parameters) 
            log_denom = log(sum(cluster_psi))
            cluster_log_weights[cluster] = log(cluster_psi[1]) - log_denom
            complement_log_weights[cluster] = log(cluster_psi[2]) - log_denom
        end
    end
    for (i, cluster) = enumerate(clusters)
        log_weight = 0.
        if cluster > 1
            log_weight += cluster_log_weights[cluster]
        end
        # Clusters younger than the current cluster
        younger_clusters = (cluster+1):(observation-1)
        log_weight += sum(complement_log_weights[younger_clusters])
        log_weights[i] = log_weight
    end
    return log_weights
end

function compute_cluster_log_predictives(observation, cluster::Int64, previous_cluster, cluster_sufficient_stats,
                                         cluster_parameters::NtlParameters)
    return reshape(compute_cluster_log_predictives(observation, [cluster], previous_cluster, cluster_sufficient_stats,
                                                   cluster_parameters), 1)[1]
end

function compute_existing_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, previous_cluster::Int64,
                                                  cluster_sufficient_stats, cluster_parameters::NtlParameters,
                                                  auxillary_variables)
    log_weights = compute_cluster_log_predictives(observation, clusters, previous_cluster, cluster_sufficient_stats, 
                                                  cluster_parameters)
    log_weights .+= existing_cluster_log_predictive(cluster_sufficient_stats, cluster_parameters, observation, auxillary_variables)
    return log_weights
end

# HMM
function compute_existing_cluster_log_predictives(observation::Int64, cluster::Int64, previous_cluster::Int64,
                                                  cluster_sufficient_stats, cluster_parameters, auxillary_variables)
    return reshape(compute_existing_cluster_log_predictives(observation, [cluster], previous_cluster, cluster_sufficient_stats,
                                                            cluster_parameters), 1)[1]
end

function compute_existing_cluster_log_predictives(observation::Int64, cluster::Int64, cluster_sufficient_stats,
                                                  cluster_parameters)
    result = compute_existing_cluster_log_predictives(observation, [cluster], cluster_sufficient_stats, 
                                                      cluster_parameters)
    return reshape(result, 1)[1]
end

function compute_existing_cluster_log_predictives(observation::Int64, clusters::Vector{Int64}, previous_cluster::Int64,
                                                  sufficient_stats, cluster_parameters::BetaNtlParameters{A}) where 
                                                  {A <: PitmanYorArrivals}
    previous_state_num_obs = sufficient_stats.cluster.state_num_observations[previous_cluster, :]
    previous_state_cluster_sufficient_stats = MixtureSufficientStatistics(previous_state_num_obs, 
                                                                          sufficient_stats.cluster.clusters)
    previous_state_sufficient_stats = SufficientStatistics(sufficient_stats.n, previous_state_cluster_sufficient_stats,
                                                           sufficient_stats.data) 
    return compute_existing_cluster_log_predictives(observation, clusters, previous_state_sufficient_stats, 
                                                    cluster_parameters)
end

get_clusters(cluster_sufficient_stats::HmmSufficientStatistics) = findall(cluster_sufficient_stats.clusters)

end