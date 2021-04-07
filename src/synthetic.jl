module Generate

using ..Models: ClusterParameters, DataParameters, GaussianParameters, GeometricArrivals, NtlParameters
using ..Models: GaussianWishartParameters
using ..Models: ArrivalDistribution, MultinomialParameters
using ..Models: Model, Mixture, Changepoint, HiddenMarkovModel
using Distributions
using LinearAlgebra
using DataFrames

struct LocationScale
    location::Vector{Float64}
    scale::Matrix{Float64}
end

function generate_log_pmfs(psi_variates::Vector{Float64})
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

    for i = 1:num_clusters
        log_pmfs[i:num_clusters, 1:(i-1)] .+= log_comple_psi_variates[i]
    end

    return log_pmfs
end

function generate_log_pmfs(psi_variates::Matrix{Float64})
    num_states = size(psi_variates)[1]
    log_transition_matrices = Array{Float64}(undef, num_states, num_states, num_states)
    for state = 1:num_states
        log_transition_matrices[state, :, :] = generate_log_pmfs(vec(psi_variates[state, :]))
    end
    return log_transition_matrices
end

function sample_clusters(arrivals::Array{Int64}, model::Mixture)
    cluster_parameters = model.cluster_parameters
    n = length(arrivals)
    cluster_assignments = Array{Int64}(undef, n)
    num_clusters = sum(arrivals) 
    new_cluster_indices = arrivals .=== 1
    arrival_times = (1:n)[new_cluster_indices]
    psi_prior = cluster_parameters.prior
    psi_variates  = rand(Beta(psi_prior[1], psi_prior[2]), num_clusters)
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

function sample_clusters(arrivals::Array{Int64}, ::Changepoint)
    n = length(arrivals)
    changepoint_assignments = Array{Int64}(undef, n)
    num_changepoints = sum(arrivals) 
    changepoint_indices = arrivals .=== 1
    arrival_times = (1:n)[changepoint_indices]

    for index = 1:num_changepoints
        arrival_time = arrival_times[index]
        if index === num_changepoints
            next_arrival_time = n + 1
        else
            next_arrival_time = arrival_times[index+1]
        end
        changepoint_assignments[arrival_time:(next_arrival_time-1)] .= index
    end

    return changepoint_assignments 
end

function sample_clusters(arrivals::Array{Int64}, model::HiddenMarkovModel)
    cluster_parameters = model.cluster_parameters
    n = length(arrivals)
    cluster_assignments = Array{Int64}(undef, n)
    num_clusters = sum(arrivals) 
    new_cluster_indices = arrivals .=== 1
    arrival_times = (1:n)[new_cluster_indices]
    psi_prior = cluster_parameters.prior

    psi_variates = rand(Beta(psi_prior[1], psi_prior[2]), num_clusters*num_clusters)
    psi_variates_matrix = reshape(psi_variates, num_clusters, num_clusters) 
    psi_variates_matrix[:, 1] .= 1 

    log_pmfs = generate_log_pmfs(psi_variates_matrix)
    pmfs = exp.(log_pmfs)

    cluster_assignments[new_cluster_indices] = 1:num_clusters
    for observation = 2:n
        if !(observation in arrival_times)
            previous_assignment = cluster_assignments[observation-1]
            newest_state = maximum(cluster_assignments[1:observation-1])
            pmf = vec(pmfs[previous_assignment, newest_state, :])
            assignment = rand(Categorical(pmf), 1)[1]
            cluster_assignments[observation] = assignment
        end
    end
    return cluster_assignments 
end

function sample_cluster_parameters(num_clusters, data_parameters::GaussianParameters)
    dist = MvNormal(data_parameters.prior_mean, data_parameters.prior_covariance)    
    return rand(dist, num_clusters)
end

function sample_cluster_parameters(num_clusters, data_parameters::GaussianWishartParameters)
    location_scale_parameters = Vector{LocationScale}(undef, num_clusters)
    dim = data_parameters.dim
    for i = 1:num_clusters
        covariance_matrix = rand(InverseWishart(data_parameters.dof, data_parameters.scale_matrix), 1)[1]
        covariance_matrix = reshape(covariance_matrix, dim, dim)
        scale = data_parameters.scale
        data_mean = vec(rand(MvNormal(data_parameters.prior_mean, (1/scale)covariance_matrix), 1))
        location_scale = LocationScale(data_mean, covariance_matrix)
        location_scale_parameters[i] = location_scale
    end
    return location_scale_parameters
end

function sample_cluster_parameters(num_clusters, data_parameters::MultinomialParameters)
    dist = Dirichlet(data_parameters.prior_dirichlet_scale)
    return rand(dist, num_clusters)
end

function sample_data(cluster, num_assigned, location_scale_parameters::Vector{LocationScale}, ::GaussianWishartParameters)
    location_scale = location_scale_parameters[cluster]
    location = location_scale.location
    scale = location_scale.scale
    normal_distribution = MvNormal(location, scale)
    cluster_data = rand(normal_distribution, num_assigned)
    return cluster_data
end

function sample_data(cluster::Int64, num_assigned::Int64, cluster_means::Matrix{Float64}, 
                     data_parameters::GaussianParameters)
    cluster_mean = vec(cluster_means[:, cluster])
    normal_distribution = MvNormal(cluster_mean, data_parameters.data_covariance)
    cluster_data = rand(normal_distribution, num_assigned)
    return cluster_data
end

function sample_data(cluster::Int64, num_assigned::Int64, cluster_probabilities::Matrix{Float64},
                     data_parameters::MultinomialParameters)
    cluster_probability = vec(cluster_probabilities[:, cluster])
    multinomial_distribution = Multinomial(data_parameters.n, cluster_probability) 
    cluster_data = rand(multinomial_distribution, num_assigned)
    return cluster_data
end

function generate_arrival_times(n::Int64, arrival_distribution::GeometricArrivals)
    arrival_prior = arrival_distribution.prior
    phi = reshape(rand(Beta(arrival_prior[1], arrival_prior[2]), 1), 1)[1] 
    arrivals = rand(Binomial(1, phi), n-1)
    prepend!(arrivals, [1])
    return arrivals
end

function prepare_data_matrix(::Type{T}, dim, n) where {T <: MultinomialParameters}
    return Array{Int64}(undef, dim, n)
end

function prepare_data_matrix(::Type{T}, dim, n) where {T <: GaussianParameters}
    return Array{Float64}(undef, dim, n)
end

function prepare_data_matrix(::Type{T}, dim, n) where {T <: GaussianWishartParameters}
    return Array{Float64}(undef, dim, n)
end

get_cluster_parameters(model::Union{Mixture, HiddenMarkovModel}) = model.cluster_parameters
get_cluster_parameters(model::Changepoint) = model.cluster_parameters

function generate(model::Model; n::Int64=100)
    data_parameters = model.data_parameters
    dim = data_parameters.dim

    cluster_prior_parameters = get_cluster_parameters(model)
    arrivals = generate_arrival_times(n, cluster_prior_parameters.arrival_distribution)
    num_clusters = sum(arrivals)
    assignments = sample_clusters(arrivals, model)
    data = prepare_data_matrix(typeof(data_parameters), dim, n)

    cluster_parameters = sample_cluster_parameters(num_clusters, data_parameters)

    for cluster = 1:num_clusters
        assigned_to_cluster = (assignments .=== cluster) 
        num_assigned = count(assigned_to_cluster)
        cluster_data = sample_data(cluster, num_assigned, cluster_parameters, data_parameters)
        data[:, assigned_to_cluster] = cluster_data
    end

    assignments = DataFrame(reshape(assignments, length(assignments), 1))
    assignments = select(assignments, "x1" => "cluster")
    data = DataFrame(transpose(data))

    data = hcat(assignments, data)
    data = convert(Matrix, data)
    return data
end

end