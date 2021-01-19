module Plot

using ..Utils: compute_co_occurrence_matrix
using Plots

function plot_assignments(assignments::Vector{Int64})
    plot(1:length(assignments), assignments, seriestype = :scatter, xlabel="Observation", ylabel="Cluster", 
         legend=false)
end

function plot_assignments(assignments::Vector{Float64})
    assignments = Vector{Int64}(assignments)
    plot_assignments(assignments)
end

function plot_co_occurrence_matrix(markov_chain::Matrix{Int64})
    co_occurrence_matrix = compute_co_occurrence_matrix(markov_chain)
    gr()
    heatmap(1:size(co_occurrence_matrix,1),
        1:size(co_occurrence_matrix,2), co_occurrence_matrix,
        c=cgrad([:white, :blue]),
        xlabel="Observations", ylabel="Observations",
        title="Co-occurrence Matrix")
end

function plot_co_occurrence_matrix(assignment::Vector{Int64})
    markov_chain = reshape(assignment, length(assignment), 1)
    plot_co_occurrence_matrix(markov_chain)
end

function plot_co_occurrence_matrix(assignment::Vector{Float64})
    assignment = Vector{Int64}(assignment)
    plot_co_occurrence_matrix(assignment)
end

end