module NtlMixture

using DataFrames
using Distributions
using LinearAlgebra
using Plots
using SpecialFunctions
using Statistics

# Mixture Generator

function generate_log_pmfs(psi_variates::Array{Float64})
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

    assignments = DataFrame(reshape(assignments, length(assignments), 1))
    assignments = select(assignments, "x1" => "cluster")
    data = DataFrame(data)

    mixture = hcat(assignments, data)
    return mixture
end

function plot_assignments(assignments)
    plot(1:size(assignments)[1], assignments, seriestype = :scatter)
end

## Mixture Fitter

function fit_mixture(data::Matrix, iterations::Int64=100)
    instances = zeros(Int64, size(data)[1], iterations)
    instances[:, 1] .= 1
    for i = 2:iterations
        instances[:, i] = gibbs_sample(data, vec(instances[:, i-1]))
    end
    instances = DataFrame(instances)
    return instances
end

function fix_mixture(data::DataFrame, iterations::Int64=100)
    return fit_mixture(Matrix(data), iterations)
end

function gibbs_sample(data, assignments)
    new_assignment = deepcopy(assignments)
    for observation = 1:size(assignments)[1]
        sample_assignment!(observation, data, new_assignment)
    end
    return new_assignment
end

function gumbel_max(objects, weights)
    gumbel_values = rand(Gumbel(0, 1), size(weights)[1])
    return objects[argmax(gumbel_values + weights)]
end

function remove_observation!(observation::Int64, assignment::Vector{Int64})
    cluster = assignment[observation]
    assignment[observation] = length(assignment) + 1 
    assigned_to_cluster = assignment .=== cluster
    if cluster === observation && count(assigned_to_cluster) > 1
        new_cluster_time = findfirst(assigned_to_cluster)
        assignment[assigned_to_cluster] .= new_cluster_time 
    end
end

function add_observation!(observation::Int64, cluster::Int64, assignment::Vector{Int64})
    if observation < cluster
        assigned_to_cluster = assignment .=== cluster
        cluster = observation
        assignment[assigned_to_cluster] .= cluster
    end
    assignment[observation] = cluster
end

function geometric_new_log_weight(n, num_clusters)
    phi_prior_a = 1
    phi_prior_b = 1
    return log(num_clusters - 1 + phi_prior_a) - log(n - 1 + phi_prior_a + phi_prior_b)
end

function geometric_existing_log_weight(n, num_clusters)
    phi_prior_a = 1
    phi_prior_b = 1
    return log(n - num_clusters + phi_prior_a) - log(n - 1 + phi_prior_a + phi_prior_b)
end

function compute_num_observations(cluster, assignment)
    assigned_to_cluster = assignment .=== cluster
    return size(assignment[assigned_to_cluster])[1]
end

function compute_num_complement(cluster, assignment, observation)
    younger_clusters = 1:(cluster-1)
    num_complement = -(cluster - 1)
    for younger_cluster = younger_clusters
        num_complement += compute_num_observations(younger_cluster, assignment)
    end
    if observation < cluster
        num_complement += 1
    end
    return num_complement
end

function compute_psi_posterior(cluster, assignment, observation)
    psi_posterior = Array{Int64}(undef, 2)
    psi_prior_a = 1
    psi_prior_b = 1 
    psi_posterior[1] = compute_num_observations(cluster, assignment) - 1 + psi_prior_a
    psi_posterior[2] = compute_num_complement(cluster, assignment, observation) + psi_prior_b
    return psi_posterior
end

function compute_ntl_predictive(observation::Int64, cluster::Int64, assignment::Vector{Int64})
    if cluster < observation
        log_weight = 0
        if cluster > 1
            cluster_psi = compute_psi_posterior(cluster, assignment, observation)
            log_weight += log(cluster_psi[1]) - log(cluster_psi[1] + cluster_psi[2])
        end
        # Clusters younger than the current cluster
        younger_clusters = (cluster+1):(observation-1)
        for younger_cluster = younger_clusters
            younger_cluster_psi = compute_psi_posterior(younger_cluster, assignment, observation)
            log_weight += log(younger_cluster_psi[2]) - log(younger_cluster_psi[1] + younger_cluster_psi[2])
        end
    else # observation < cluster
        cluster_num_obs = compute_num_observations(cluster, assignment)
        cluster_old_psi = compute_psi_posterior(cluster, assignment, observation)
        # Clusters younger than the observation
        younger_clusters = (observation+1):(cluster-1)
        youngest_cluster = minimum(assignment)

        younger_clusters_log_weight = 0
        for younger_cluster = younger_clusters
            younger_cluster_old_psi = compute_psi_posterior(younger_cluster, assignment, observation)
            younger_cluster_new_psi = deepcopy(younger_cluster_old_psi)
            if observation === 1 && younger_cluster === youngest_cluster
                younger_cluster_new_psi[1] = cluster_num_obs + 1 
                younger_cluster_old_psi = cluster_old_psi
            else
                younger_cluster_new_psi[1] += cluster_num_obs
            end
            new_a = younger_cluster_new_psi[1]
            new_b = younger_cluster_new_psi[2]
            old_a = younger_cluster_old_psi[1]
            old_b = younger_cluster_old_psi[2]
            younger_clusters_log_weight += logbeta(new_a, new_b) - logbeta(old_a, old_b)
        end
        if observation > youngest_cluster
            cluster_new_psi = deepcopy(cluster_old_psi)
            cluster_new_psi[1] += 1
            cluster_new_psi[2] = compute_num_complement(observation, assignment, observation) + 1
            new = cluster_new_psi
            old = cluster_old_psi
            cluster_log_weight = logbeta(new[1], new[2]) - logbeta(old[1], old[2])
            log_weight = cluster_log_weight + younger_clusters_log_weight
        else
            log_weight = younger_clusters_log_weight
        end
    end
    return log_weight
end

function compute_existing_cluster_weights(observation, assignment)
    clusters = get_clusters(assignment)
    num_clusters = length(clusters)
    log_weights = Array{Float64}(undef, num_clusters)
    for i = 1:num_clusters
        cluster = clusters[i]
        log_weights[i] = compute_ntl_predictive(observation, cluster, assignment)
    end
    log_weights .+= geometric_existing_log_weight(size(assignment)[1], num_clusters)
    return log_weights
end

function compute_normal_posterior_parameters(data)
    n = size(data)[1] 
    dim = size(data)[2]
    prior_mean = zeros(dim)
    if n > 0
        cluster_mean = vec(Statistics.mean(data, dims=1))
        data_precision = inv(0.1I)
        cov = inv(I + n*data_precision)
        posterior_mean = cov*(I*prior_mean + n*data_precision*cluster_mean) 
        return posterior_mean, cov
    else
        return prior_mean, I 
    end
end

function compute_cluster_data_log_predictive(datum, cluster, data, assignment)
    assigned_to_cluster = assignment .=== cluster
    cluster_data = data[assigned_to_cluster, :]
    posterior_mean, cov = compute_normal_posterior_parameters(cluster_data)
    posterior = MvNormal(posterior_mean, cov)
    value = logpdf(posterior, datum)
    return value
end

function compute_data_log_predictive(observation, clusters, data, assignment)
    num_clusters = length(clusters)
    log_predictive = Array{Float64}(undef, num_clusters)
    datum = vec(data[observation, :])
    for i = 1:num_clusters
        cluster = clusters[i]
        log_predictive[i] = compute_cluster_data_log_predictive(datum, cluster, data, assignment)
    end
    return log_predictive
end

function get_clusters(assignment)
    assigned = assignment .<= length(assignment)
    return sort(unique(assignment[assigned]))
end

function sample_assignment!(observation::Int64, data, assignment)
    remove_observation!(observation, assignment)
    clusters = get_clusters(assignment)
    n = size(data)[1]
    num_clusters = size(clusters)[1]

    choices = Array{Int64}(undef, num_clusters+1)
    choices[1:num_clusters] = clusters
    choices[num_clusters+1] = observation

    weights = Array{Float64}(undef, num_clusters+1)
    weights[1:num_clusters] = compute_existing_cluster_weights(observation, assignment)
    weights[num_clusters+1] = geometric_new_log_weight(n, num_clusters)
    weights += compute_data_log_predictive(observation, choices, data, assignment)

    cluster = gumbel_max(choices, weights)
    add_observation!(observation, cluster, assignment)
end

end