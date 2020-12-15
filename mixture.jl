using DataFrames
using Distributions
using LinearAlgebra

# Mixture Generator

function generate_log_pmfs(psi_variates)
    num_clusters = size(psi_variates)[1] 
    complement_psi_variates = 1 .- psi_variates
    log_psi_variates = log.(psi_variates)
    log_comple_psi_variates = log.(complement_psi_variates)

    log_pmfs = -ones(num_clusters, num_clusters)./0
    log_pmfs_diag = diagind(log_pmfs)
    log_pmfs[log_pmfs_diag] = log_psi_variates

    for j = 1:num_clusters
        log_pmfs[(j+1):num_clusters, j] .= log_psi_variates[j]
    end

    for i = 2:num_clusters
        log_pmfs[i:num_clusters, 1:(i-1)] .+= log_comple_psi_variates[i]
    end

    return log_pmfs
end

function sample_clusters(arrivals::Array{Int64})
    n = size(arrivals)[1]
    cluster_assignments = Array{Int64}(undef, n)
    num_clusters = sum(arrivals) 
    new_cluster_indices = arrivals .=== 1
    arrival_times = (1:n)[new_cluster_indices]
    psi_variates  = rand(Beta(1, 1), num_clusters)
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
        num_obs = size(observation_range)[1]
        pmf = vec(pmfs[i, :])
        observation_assignments = rand(Categorical(pmf), num_obs)
        cluster_assignments[observation_range] = observation_assignments
    end

    return cluster_assignments 
end

function generate_mixture(n::Int64, dim::Int64, variance::Float64=0.1)
    phi_distribution = Beta(1, 1)
    phi = reshape(rand(phi_distribution, 1), 1)[1] 
    arrivals = rand(Binomial(1, phi), n-1)
    prepend!(arrivals, [1])
    num_clusters = sum(arrivals)

    assignments = sample_clusters(arrivals)

    prior_mean = vec(zeros(dim, 1))
    prior_cov = I
    cluster_means = rand(MvNormal(prior_mean, prior_cov), num_clusters)
    print(cluster_means)

    data_covariance = variance*I 
    data = Array{Float64}(undef, n, dim)

    for cluster = 1:num_clusters
        assigned_to_cluster = (assignments .=== cluster) 
        num_assigned = count(assigned_to_cluster)
        cluster_mean = vec(cluster_means[:, cluster])
        normal_distribution = MvNormal(cluster_mean, data_covariance)
        cluster_data = transpose(rand(normal_distribution, num_assigned))
        data[assigned_to_cluster, :] = cluster_data
    end

    return assignments, data
end