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

function plot_co_occurrence_matrix(markov_chain::Matrix{Int64}, weights::Vector{Float64})
    co_occurrence_matrix = compute_co_occurrence_matrix(markov_chain, weights)
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

function plot_arrival_posterior_probabilities(markov_chain, true_clustering)
    num_iterations = size(markov_chain)[2]
    num_observations = size(markov_chain)[1]
    arrival_counts = zeros(Float64, num_observations)
    observations = Array{Int64}(1:num_observations)
    for i = 1:num_iterations
        arrival_counts += (markov_chain[:, i] .=== observations)
    end
    arrival_posterior_probabilities = arrival_counts / num_iterations

    num_clusters = maximum(true_clustering)
    clusters = Array{Float64}(1:num_clusters)
    true_arrivals = []
    for cluster = clusters
        arrival = findfirst(true_clustering .=== cluster)
        append!(true_arrivals, arrival)
    end

    plot(1:num_observations, arrival_posterior_probabilities, seriestype=:scatter,
         xlabel="Observation", ylabel="Probability", legend=false, )
    vline!(true_arrivals)
end

function plot_log_likelihoods(log_likelihoods; burn_in=0) 
    plot(1:size(log_likelihoods)[1], log_likelihoods, seriestype=:line,
         xlabel="Iteration", ylabel="Log likelihood", legend=false)
end

function plot_num_clusters(markov_chain) 
    num_unique_clusters_vector = mapslices(u->length(unique(u)), markov_chain, dims=1)
    plot(1:length(num_unique_clusters_vector), vec(num_unique_clusters_vector), seriestype=:line,
         xlabel="Iteration", ylabel="Number of clusters", legend=false)
end

function plot_trace(values; ylabel="trace")
    plot(1:length(values), values, seriestype=:line, xlabel="Iteration", ylabel=ylabel, legend=false)
end

end