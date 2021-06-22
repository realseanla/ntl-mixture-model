module Utils

using Distributions
using SpecialFunctions
import SpecialFunctions: logbeta, logfactorial 
using LinearAlgebra
using Clustering
using RCall
using ProgressMeter
using Combinatorics

function one_hot_encode(assignments::Vector{Int64})
    n = length(assignments)
    youngest_cluster = maximum(assignments)
    encoding_matrix = zeros(Int64, youngest_cluster, n)
    for observation = 1:n
        cluster = assignments[observation]
        encoding_matrix[cluster, observation] = 1
    end
    return encoding_matrix
end

function compute_co_occurrence_matrix(markov_chain::Matrix{Int64})
    num_instances = size(markov_chain)[2]
    weights = ones(Float64, num_instances)
    return compute_co_occurrence_matrix(markov_chain, weights)
end

function compute_co_occurrence_matrix(markov_chain::Matrix{Int64}, weights::Vector{Float64})
    n = size(markov_chain)[1]
    co_occurrence_matrix = zeros(Int64, n, n)
    num_instances = size(markov_chain)[2]
    @showprogress for i = 1:num_instances
        assignment = markov_chain[:, i]
        ohe_assignment = one_hot_encode(assignment)
        instance_co_occurrence_matrix = transpose(ohe_assignment) * ohe_assignment
        weight = weights[i]
        co_occurrence_matrix += weight*instance_co_occurrence_matrix
    end
    max_value = maximum(co_occurrence_matrix)
    co_occurrence_matrix /= max_value
    return co_occurrence_matrix
end

eachcol(A::AbstractVecOrMat) = (view(A, :, i) for i in axes(A, 2))


logbeta(parameters::Vector{T}) where {T<:Real} = logbeta(parameters[1], parameters[2])

function logmvbeta(parameters::Vector{T}) where {T <: Real}
    return sum(loggamma.(parameters)) - loggamma(sum(parameters))
end

function log_multinomial_coeff(counts::Vector{Int64})
    return logfactorial(sum(counts)) - sum(logfactorial.(counts)) 
end

logfactorial(x::Float64) = logfactorial(Int(x))

function gumbel_max(objects::Vector{Int64}, log_weights::Vector{Float64})
    gumbel_values = rand(Gumbel(0, 1), length(log_weights))
    index = argmax(gumbel_values + log_weights)
    return (objects[index], log_weights[index])
end

function gumbel_max(num_draws::Int64, log_weights::Vector{Float64})
    indices = Vector{Int64}(1:length(log_weights))
    resampled_indices = Vector{Int64}(undef, num_draws) 
    for i = 1:num_draws
        (sampled_index, log_weight) = gumbel_max(indices, log_weights) 
        resampled_indices[i] = sampled_index
    end
    return resampled_indices
end

isnothing(::Any) = false
isnothing(::Nothing) = true 

function compute_normalized_weights(log_weights)
    num_weights = length(log_weights)
    max_value = maximum(log_weights)
    shifted_log_weights = log_weights .- max_value
    weights = exp.(shifted_log_weights)
    normalized_weights = weights ./ sum(weights)
    @assert !any(isnan.(normalized_weights))
    return normalized_weights
end

function effective_sample_size(log_weights)
    normalized_weights = compute_normalized_weights(log_weights)
    ess = 1/sum(normalized_weights.^2)
    @assert !isnan(ess)
    return ess
end

function map_estimate(instances, log_likelihoods; num_burn_in=0)
    @assert size(instances)[2] === length(log_likelihoods)
    burned_in_log_likelihoods = log_likelihoods[num_burn_in+1:end]
    max_value = maximum(burned_in_log_likelihoods)
    burned_in_instances = instances[:, num_burn_in+1:end]
    map_index = findfirst(burned_in_log_likelihoods .=== max_value)
    return burned_in_instances[:, map_index]
end

function num_clusters(instances)
    return vec(mapslices(a -> length(unique(a)), instances; dims=[1]))
end

function mean_num_clusters(instances, weights)
    number_of_clusters = num_clusters(instances)
    return sum(weights .* number_of_clusters)/sum(weights)
end

function mean_num_clusters(instances::Matrix)
    return mean(num_clusters(instances))
end

function mean_num_clusters(instances::Vector)
    return mean_num_clusters(instances, [1])
end

function hist(v::AbstractVector, edg::AbstractVector)
    n = length(edg)-1
    h = zeros(Int, n)
    for x in v
        i = searchsortedfirst(edg, x)-1
        if 1 <= i <= n
            h[i] += 1
        end
    end
    return h
end

function mvt_logpdf(location, scale, dof, datum)
    log_det = logdet(scale)
    dim = length(datum)
    log_likelihood = -(dof + dim)log(1 + transpose(datum - location)*inv(scale)*(datum - location)/dof)/2
    log_likelihood += loggamma((dof + dim)/2) - loggamma(dof/2) - dim*log(dof*pi)/2 - log_det/2 
    return log_likelihood
end

function adjusted_rand_index(c1::Vector{Int64}, c2::Vector{Int64})
    return Clustering.randindex(c1,c2)[1]
end

function ari_over_markov_chain(true_clustering::Vector{Int64}, markov_chain::Matrix{Int64})
    posterior_ari = vec(mapslices(z->adjusted_rand_index(true_clustering, z), markov_chain, dims=1))
    return posterior_ari
end

function minbinder(posterior_similarity_matrix::Matrix{Float64}, iterations::Matrix{Int64})
    draws = Matrix(transpose(iterations))
    R"library('mcclust')"
    R"library('mcclust.ext')"
    @rput posterior_similarity_matrix
    @rput draws
    R"binder_clustering <- minbinder.ext(posterior_similarity_matrix, draws, method='draws')"
    clustering = vec(rcopy(R"binder_clustering$cl"))
    R"rm(list = ls())"
    return clustering
end

function minbinder(markov_chain::Matrix{Int64}; num_burn_in=0)
    psm = compute_co_occurrence_matrix(markov_chain[:, (num_burn_in+1):end])
    return minbinder(psm, markov_chain)
end

function minVI(posterior_similarity_matrix, iterations)
    draws = Matrix(transpose(iterations))
    R"library('mcclust')"
    R"library('mcclust.ext')"
    @rput posterior_similarity_matrix
    @rput draws
    R"vi_clustering <- minVI(posterior_similarity_matrix, draws, method='draws')"
    clustering = vec(rcopy(R"vi_clustering$cl"))
    R"rm(list = ls())"
    return clustering
end

function minVI(markov_chain::Matrix{Int64}; num_burn_in=0)
    psm = compute_co_occurrence_matrix(markov_chain[:, (num_burn_in+1):end])
    return minVI(psm, markov_chain[:, (num_burn_in+1):end])
end

hasproperty(x, s::Symbol) = s in fieldnames(typeof(x))

function generate_all_clusterings(n::Int64)
    clusterings = Vector{Vector{Vector{Int64}}}(undef, 0)
    for i=1:n 
        clusterings = vcat(clusterings, collect(partitions(1:n, i)))
    end
    return clusterings
end

end