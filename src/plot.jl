module Plot

using ..Utils: compute_co_occurrence_matrix, ari_over_markov_chain, generate_all_clusterings
using ..Evaluate: compute_posterior_probability, compute_posterior_probability_confidence_interval
using Plots
using Statistics
using Clustering

function plot_assignments(assignments::Vector{Int64}; title="Cluster assignments for observations")
    plot(
        1:length(assignments), 
        assignments, 
        seriestype = :scatter, 
        xlabel="Observation", 
        ylabel="Cluster", 
        title=title,
        legend=false
    )
end

function plot_assignments(assignments::Vector{Float64}; title="Cluster assignments for observations")
    assignments = Vector{Int64}(assignments)
    plot_assignments(assignments, title=title)
end

function plot_co_occurrence_matrix(markov_chain::Matrix{Int64}; num_burn_in=0, title="Co-occurrence matrix")
    co_occurrence_matrix = compute_co_occurrence_matrix(markov_chain[:, (num_burn_in+1):end])
    gr()
    heatmap(1:size(co_occurrence_matrix,1),
           1:size(co_occurrence_matrix,2), 
           co_occurrence_matrix,
           c=cgrad([:white, :blue]),
           xlabel="Observations", 
           ylabel="Observations",
           title=title
        )
end

function plot_co_occurrence_matrix(markov_chains::Array{Int64}; num_burn_in=0, plot_size=500)
    num_chains = size(markov_chains)[3]
    subplots = []
    for chain = 1:num_chains
        markov_chain = markov_chains[:, :, chain]
        co_occurrence_matrix = compute_co_occurrence_matrix(markov_chain, num_burn_in=num_burn_in)
        gr()
        subplot = heatmap(1:size(co_occurrence_matrix,1),
           1:size(co_occurrence_matrix,2), co_occurrence_matrix,
           c=cgrad([:white, :blue]),
           xlabel="Observations", ylabel="Observations",
           title="Chain $chain: Co-occurrence Matrix")
        push!(subplots, subplot)
    end
    plot(subplots..., layout = (num_chains, 1), legend=false, size = (plot_size*(4/3), plot_size * num_chains*(3/4)))
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

function plot_arrival_posterior_probabilities(markov_chain::Matrix{Int64}, true_clustering; 
                                              num_burn_in=0, title="Arrival time posterior probabilities")
    num_iterations = size(markov_chain)[2]
    num_observations = size(markov_chain)[1]
    arrival_counts = zeros(Float64, num_observations)
    observations = Array{Int64}(1:num_observations)
    for i = (num_burn_in+1):num_iterations
        arrival_counts += (markov_chain[:, i] .=== observations)
    end
    arrival_posterior_probabilities = arrival_counts / (num_iterations - num_burn_in)

    num_clusters = maximum(true_clustering)
    clusters = Array{Int64}(1:num_clusters)
    true_arrivals = []
    for cluster = clusters
        arrival = findfirst(true_clustering .=== cluster)
        append!(true_arrivals, arrival)
    end

    colours = Vector{String}(undef, num_observations)

    for observation = 1:num_observations
        posterior_probability = arrival_posterior_probabilities[observation]
        if posterior_probability >= 0.5 
            if observation in true_arrivals # true positive
                colours[observation] = "green"
            else # false positive  
                colours[observation] = "red"
            end
        else 
            if observation in true_arrivals # false negative 
                colours[observation] = "red"
            else # true negative 
                colours[observation] = "green"
            end
        end
    end
    vline(
        true_arrivals,
        linecolor="black",
        linestyle=:dash,
        legend=nothing,
        ylims=[0., 1.]
    )
    plot!(
        1:num_observations, 
        arrival_posterior_probabilities, 
        color=colours,
        seriestype=:scatter,
        xlabel="Observation", 
        ylabel="Probability", 
        legend=false, 
        title=title     
    )
end

function plot_log_likelihoods(log_likelihoods::Vector{Float64}; title="Log likelihoods over iterations")
    plot(
        1:size(log_likelihoods)[1], 
        log_likelihoods, 
        seriestype=:line,
        xlabel="Iteration", 
        ylabel="Log likelihood", 
        legend=false,
        title=title
    )
end

function plot_log_likelihoods(log_likelihoods::Matrix{Float64}; 
                              assignment_types=[], 
                              legend_location=:topright, 
                              title="Log likelihood over iterations")
    num_chains = size(log_likelihoods)[2]
    num_iterations = size(log_likelihoods)[1]
    if length(assignment_types) > 0
        @assert length(assignment_types) == num_chains
    end
    assignment_types = ["Initial assignment: $assignment_type" for assignment_type in assignment_types]
    plot(
        1:num_iterations, 
        log_likelihoods, 
        seriestype=:line, 
        title=title,
        xlabel="Iteration", 
        ylabel="Log likelihood",
        legend=legend_location,
        label=reshape(assignment_types, 1, length(assignment_types))
    )
end

function plot_num_clusters(markov_chain::Matrix{Int64}; true_number=0, title="Number of clusters") 
    num_unique_clusters_vector = mapslices(u->length(unique(u)), markov_chain, dims=1)
    plot(
        1:length(num_unique_clusters_vector), 
        vec(num_unique_clusters_vector), 
        seriestype=:line,
        xlabel="Iteration", 
        ylabel="Number of clusters", 
        title=title,
        legend=false
    )
    if true_number > 0 
        hline!([true_number], label="True number of clusters = $true_number")  
    end
end

function plot_num_clusters(markov_chain::Array{Int64}; 
                           true_number=0, 
                           assignment_types=[], 
                           legend_location=:topright,
                           title="Number of clusters over iterations") 
    num_iterations = size(markov_chain)[2]
    num_chains = size(markov_chain)[3]
    if length(assignment_types) > 0
        @assert length(assignment_types) === num_chains
    end
    num_clusters = Matrix{Int64}(undef, num_iterations, num_chains)
    for chain = 1:num_chains
        num_unique_clusters_vector = vec(mapslices(u->length(unique(u)), markov_chain[:, :, chain], dims=1))
        num_clusters[:, chain] = num_unique_clusters_vector
    end
    assignment_types = ["Initial assignment: $assignment_type" for assignment_type in assignment_types]
    plot(1:num_iterations, 
        num_clusters, 
        seriestype=:line, 
        title=title,
        xlabel="Iteration", 
        ylabel="Number of clusters",
        legend=legend_location,
        label=reshape(assignment_types, 1, length(assignment_types))
        )
    if true_number > 0 
        hline!([true_number], label="True number of clusters = $true_number")  
    end
end

function plot_trace(values; 
                    ylabel="Trace", 
                    assignment_types=[],
                    title="$ylabel over iterations")
    num_chains = size(values)[end]
    num_iterations = size(values)[1]
    if length(assignment_types) > 0
        @assert length(assignment_types) == num_chains
    end
    assignment_types = ["Initial assignment: $assignment_type" for assignment_type in assignment_types]
    plot(
        1:num_iterations, 
        values, 
        seriestype=:line, 
        title="$ylabel over iterations",
        xlabel="Iteration", 
        ylabel=ylabel,
        label=reshape(assignment_types, 1, length(assignment_types))
    )
end

function plot_ari_posterior_distribution(true_clustering::Vector{Int64}, 
                                         assignment_posterior::Matrix{Int64}; 
                                         num_burn_in=0,
                                         title="ARI posterior distribution")
    ari_posterior = ari_over_markov_chain(true_clustering, assignment_posterior[:, (num_burn_in+1):end])
    mean_ari = round(mean(ari_posterior), digits=3)
    plot(
        ari_posterior, 
        seriestype=:histogram, 
        legend=true, 
        label=""
    )
    vline!(
        [mean_ari], 
        label="Mean ARI = $mean_ari",
        xlabel="ARI",
        ylabel="Frequency",
        title=title
    )
end

function plot_ari_posterior_distribution(true_clustering::Vector{Int64}, 
                                         assignment_posterior::Array{Int64}; 
                                         num_burn_in=0,
                                         title="ARI posterior distribution")
    num_chains = size(assignment_posterior)[3]
    subplots = []
    for chain = 1:num_chains
        ari_posterior = ari_over_markov_chain(true_clustering, assignment_posterior[:, (num_burn_in+1):end, chain])
        mean_ari = round(mean(ari_posterior), digits=3)
        subplot = plot(ari_posterior, seriestype=:histogram, legend=true, label="", title="Chain $chain", xlabel="ARI")
        vline!([mean_ari], label="Mean ARI = $mean_ari")
        push!(subplots, subplot)
    end
    plot(subplots..., layout=(num_chains, 1), legend=true)
end

function plot_cluster_sizes_histogram(clustering::Vector{Int64}; title="Histogram of cluster sizes")
    num_clusters = length(unique(clustering))
    cluster_sizes = Vector{Int64}(undef, num_clusters)
    for cluster = 1:num_clusters 
        cluster_sizes[cluster] = count(clustering .=== cluster)
    end
    mean_cluster_size = round(mean(cluster_sizes), digits=2)
    plot(
        cluster_sizes, 
        seriestype=:histogram, 
        legend=true, 
        label="", 
        bins=10,
        title=title
    )
    xlabel!("Size of cluster")
    vline!([mean_cluster_size], label="Mean cluster size = $mean_cluster_size")
end

function plot_clustering_posterior_probability_validation(markov_chain::Matrix{Int64}, 
                                                          data, 
                                                          model; 
                                                          title="95% Confidence Intervals for True Posterior Probabilities")
    gr()
    n = size(markov_chain)[1]
    all_clusterings = generate_all_clusterings(n)
    num_clusterings = size(all_clusterings)[1]
    true_posterior_probabilities = Vector{Float64}(undef, num_clusterings)
    mcmc_means = Vector{Float64}(undef, num_clusterings)
    mcmc_conf_length = Vector{Float64}(undef, num_clusterings)
    for (i, clustering) in enumerate(all_clusterings) 
        true_clustering_posterior_probability = compute_posterior_probability(clustering, data, model)
        true_posterior_probabilities[i] = true_clustering_posterior_probability 
        (prob_mean, conf_length) = compute_posterior_probability_confidence_interval(clustering, markov_chain)
        mcmc_means[i] = prob_mean
        mcmc_conf_length[i] = conf_length
    end
    
    colours = Vector{String}(undef, num_clusterings)
    for i = 1:num_clusterings
        true_posterior_prob = true_posterior_probabilities[i]
        mcmc_mean = mcmc_means[i]
        conf_length = mcmc_conf_length[i]
        if (mcmc_mean - conf_length <= true_posterior_prob) && (true_posterior_prob <= mcmc_mean + conf_length)
            colours[i] = "blue"
        else
            colours[i] = "red"
        end
    end
    
    # Create custom error bars
    # Top caps
    p = plot(
          mcmc_means + mcmc_conf_length, 
          seriestype=:scatter,
          markershape=:hline,
          markercolor=:black,
          markersize=10,
          legend=nothing, 
          )
    # Bottom caps
    plot!(p,
          mcmc_means - mcmc_conf_length, 
          seriestype=:scatter,
          markershape=:hline,
          markercolor=:black,
          markersize=10,
          legend=nothing, 
          )
    # Vertical bars 
    for i=1:num_clusterings 
        mcmc_mean = mcmc_means[i]
        conf_length = mcmc_conf_length[i]
        bar_end_points = [
            mcmc_mean - conf_length,
            mcmc_mean + conf_length
        ]
        plot!(
            p,
            [i, i],
            bar_end_points,
            seriestype=:line,
            legend=nothing,
            linecolor=:black
            )
    end
    plot!(
        p,
        true_posterior_probabilities, 
        seriestype=:scatter,
        color=colours, 
        legend=nothing,
        xlabel="Clustering",
        ylabel="Posterior Probability",
        title=title,
        )
    p
end

function plot_kmeans_elbow(data; max_num_clusters=10, title="Elbow plot")
    totalcosts = Vector{Float64}(undef, max_num_clusters)
    for k = 1:max_num_clusters
        kmeans_result = kmeans(data, k)
        totalcosts[k] = kmeans_result.totalcost
    end
    plot(
        1:max_num_clusters, 
        totalcosts, 
        label="", 
        title=title,
        xlabel="Number of clusters (k)",
        ylabel="Sum of squared errors"
    )
end

end