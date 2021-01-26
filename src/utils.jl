module Utils

using Distributions
using SpecialFunctions
import SpecialFunctions: logbeta 
using ProgressBars

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
    n = size(markov_chain)[1]
    co_occurrence_matrix = zeros(Int64, n, n)
    num_instances = size(markov_chain)[2]
    for i = ProgressBar(1:num_instances)
        assignment = markov_chain[:, i]
        ohe_assignment = one_hot_encode(assignment)
        co_occurrence_matrix += transpose(ohe_assignment) * ohe_assignment
    end
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
    scaled_weights = exp.(log_weights .- max_value .+ log(num_weights))
    normalized_weights = scaled_weights ./ sum(scaled_weights)
    return normalized_weights
end

end