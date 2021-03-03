function gibbs_sample!(observation, instances, iteration, data, sufficient_stats, model::HiddenMarkovModel)
    n = size(data)[2]
    remove_observation!(observation, instances, iteration, data, sufficient_stats, model)
    previous_state = (observation > 1) ? instances[observation-1, iteration] : n+1
    next_state = (observation < n) ? instances[observation+1, iteration] : n+1
    (cluster, weight) = gibbs_proposal(observation, data, sufficient_stats, model, previous_state, next_state)
    add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model, 
                     update_next_cluster=true)
end

function gibbs_proposal(observation, data, sufficient_stats, model, previous_state, next_state)
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
                                                                           sufficient_stats, model.cluster_parameters)
        weights[num_clusters+1] = new_cluster_log_predictive(sufficient_stats.cluster, model.cluster_parameters)
    end
    @assert !any(isnan.(weights))
    # Add contribution from transitioning into next state 
    if next_state !== n+1
        for (index, choice) = enumerate(choices)
            # TODO: make this dispatch on the boolean value
            if next_state === observation+1
                weights[index] += new_cluster_log_predictive(sufficient_stats.cluster, model.cluster_parameters)
            else
                weights[index] += compute_existing_cluster_log_predictives(observation, next_state, choice, 
                                                                           sufficient_stats, model.cluster_parameters)
            end
        end
    end
    @assert !any(isnan.(weights))
    # Add contribution from data
    weights += compute_data_log_predictives(observation, choices, data, sufficient_stats.data, model.data_parameters)
    @assert !any(isnan.(weights))

    return gumbel_max(choices, weights)
end

function propose(observation, data, particles, particle, sufficient_stats, model::HiddenMarkovModel)
    previous_state = particles[observation-1, particle]
    next_state = size(data)[2] + 1
    return gibbs_proposal(observation, data, sufficient_stats, model, previous_state, next_state)
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
    @assert sufficient_stats.cluster.state_num_observations[source_cluster, cluster] >= 0
    @assert sufficient_stats.cluster.num_observations[cluster] >= 0
    @assert sufficient_stats.cluster.state_assignments[source_cluster][observation, cluster] >= 0
end

function update_cluster_stats_new_birth_time!(num_observations::Vector{Int64}, new_time::Int64, old_time::Int64)
    num_observations[new_time] = num_observations[old_time]
    num_observations[old_time] = 0
end

function update_cluster_stats_new_birth_time!(cluster_sufficient_stats::StationaryHmmSufficientStatistics, new_time, old_time)
    @assert any(cluster_sufficient_stats.state_num_observations[:, old_time] .> 0)
    @assert all(cluster_sufficient_stats.state_num_observations[new_time, :] .=== 0)
    @assert all(cluster_sufficient_stats.state_num_observations[:, new_time] .=== 0)
    @assert !cluster_sufficient_stats.clusters[new_time]
    @assert cluster_sufficient_stats.clusters[old_time]

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

    @assert all(cluster_sufficient_stats.state_num_observations[old_time, :] .=== 0)
    @assert all(cluster_sufficient_stats.state_num_observations[:, old_time] .=== 0)
    @assert any(cluster_sufficient_stats.state_num_observations[:, new_time] .> 0)
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
        @assert all(cluster_sufficient_stats.state_assignments[cluster][:, old_time] .=== 0) 
    end
    cluster_sufficient_stats.num_observations[new_time] = cluster_sufficient_stats.num_observations[old_time]
    cluster_sufficient_stats.num_observations[old_time] = 0
    @assert all(cluster_sufficient_stats.state_num_observations[old_time, :] .=== 0)
    @assert all(cluster_sufficient_stats.state_num_observations[:, old_time] .=== 0)
    @assert any(cluster_sufficient_stats.state_num_observations[:, new_time] .> 0)
    @assert any(cluster_sufficient_stats.state_assignments[new_time][:, :] .> 0)
    @assert all(cluster_sufficient_stats.state_assignments[old_time][:, :] .=== 0)
end

function remove_observation!(observation::Int64, instances::Matrix{Int64}, iteration::Int64, data::Matrix{T}, 
                             sufficient_stats::SufficientStatistics, model::HiddenMarkovModel) where {T <: Real}
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
    datum = vec(data[:, observation])
    update_data_sufficient_statistics!(sufficient_stats, cluster, datum, model, update_type="remove")
    compute_data_posterior_parameters!(cluster, sufficient_stats, model.data_parameters)
end

function add_observation!(observation, cluster, instances, iteration, data, sufficient_stats, model::HiddenMarkovModel;
                          update_next_cluster=false)
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
    update_data_sufficient_statistics!(sufficient_stats, cluster, datum, model, update_type="add")
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
                                                  cluster_sufficient_stats, cluster_parameters::NtlParameters)
    log_weights = compute_cluster_log_predictives(observation, clusters, previous_cluster, cluster_sufficient_stats, 
                                                  cluster_parameters)
    log_weights .+= existing_cluster_log_predictive(cluster_sufficient_stats.cluster, cluster_parameters)
    return log_weights
end

# HMM
function compute_existing_cluster_log_predictives(observation::Int64, cluster::Int64, previous_cluster::Int64,
                                                  cluster_sufficient_stats, cluster_parameters)
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
                                                  cluster_sufficient_stats, 
                                                  cluster_parameters::BetaNtlParameters{A}) where 
                                                  {A <: PitmanYorArrivals}
    previous_cluster_num_obs = cluster_sufficient_stats.cluster.state_num_observations[previous_cluster, :]
    previous_cluster_sufficient_stats = MixtureSufficientStatistics(previous_cluster_num_obs, 
                                                                    cluster_sufficient_stats.cluster.clusters)
    return compute_existing_cluster_log_predictives(observation, clusters, previous_cluster_sufficient_stats, 
                                                    cluster_parameters)
end

get_clusters(cluster_sufficient_stats::HmmSufficientStatistics) = findall(cluster_sufficient_stats.clusters)

end