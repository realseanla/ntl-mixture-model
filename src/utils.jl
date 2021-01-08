module Utils

using SpecialFunctions
import SpecialFunctions: logbeta 

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
    for i = 1:num_instances
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

end