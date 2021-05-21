module Models

export Model, Mixture, HiddenMarkovModel
export ArrivalDistribution, GeometricArrivals, PoissonArrivals
export MixtureParameters, NtlParameters, DpParameters
export DataParameters, GaussianParameters, MultinomialParameters 
export GaussianWishartParameters
export ClusterSufficientStatistics, DataSufficientStatistics
export NtlSufficientStatistics, GaussianSufficientStatistics, MultinomialParameters
export HmmSufficientStatistics, MixtureSufficientStatistics
export StationaryHmmSufficientStatistics, NonstationaryHmmSufficientStatistics
export ChangepointSufficientStatistics, Changepoint
export BetaNtlParameters, BetaNtlSufficientStatistics, ParametricArrivalsClusterParameters
export PitmanYorArrivals
export SufficientStatistics
export AuxillaryVariables, ArrivalsAuxillaryVariables, DataAuxillaryVariables
export GeometricAuxillaryVariables, PoissonAuxillaryVariables, DpAuxillaryVariables
export GaussianAuxillaryVariables, GaussianWishartAuxillaryVariables, MultinomialAuxillaryVariables
export FiniteTopicModelAuxillaryVariables

using SparseArrays
using DataStructures

abstract type DataParameters end

struct GaussianParameters <: DataParameters
    data_covariance::Matrix{Float64}
    data_precision::Matrix{Float64}
    prior_mean::Vector{Float64}
    prior_covariance::Matrix{Float64}
    dim::Int64
    function GaussianParameters(data_covariance::Matrix{Float64},
                                prior_mean::Vector{Float64},
                                prior_covariance::Matrix{Float64})
        if size(data_covariance) !== size(prior_covariance)
            error("Dimension of data and prior covariance must match")
        end
        dim = length(prior_mean)
        if dim !== size(prior_covariance)[1] || dim !== size(prior_covariance)[2] 
            error("Dimension of mean prior must match each dimension of prior covariance")
        end
        data_precision = inv(data_covariance)
        return new(data_covariance, data_precision, prior_mean, prior_covariance, dim)
    end
end

struct GaussianWishartParameters <: DataParameters
    prior_mean::Vector{Float64}
    scale_matrix::Matrix{Float64}
    scale::Float64
    dof::Float64
    dim::Int64
    function GaussianWishartParameters(prior_mean, scale_matrix, scale, dof)
        dim = length(prior_mean) 
        return new(prior_mean, scale_matrix, scale, dof, dim)
    end
end

struct MultinomialParameters <: DataParameters
    n::Int64
    dim::Int64
    prior_dirichlet_scale::Vector
    function MultinomialParameters(n::Int64, prior_dirichlet_scale::Vector)
        dim = length(prior_dirichlet_scale)
        if dim < 2
            error("Dirichlet distribution must have dimension at least 2")
        end
        if any(prior_dirichlet_scale .<= 0)
            error("Each value of Dirichlet scale parameter must be greater than 0")
        end
        if n <= 0
            error("n must be greater than 0")
        end
        return new(n, dim, prior_dirichlet_scale)
    end
    function MultinomialParameters(prior_dirichlet_scale::Vector)
        return MultinomialParameters(1, prior_dirichlet_scale)
    end
end

struct FiniteTopicModelParameters <: DataParameters
    num_topics::Int64
    num_words::Int64
    length::Int64
    topic_parameter::Float64
    word_parameter::Float64
    function FiniteTopicModelParameters(;num_topics::Int64=10, num_words::Int64=100, length::Int64=100, 
                                        topic_parameter::Float64=0.1, word_parameter::Float64=0.1)
        return new(num_topics, num_words, length, topic_parameter, word_parameter)
    end
end

abstract type ArrivalDistribution end

struct GeometricArrivals <: ArrivalDistribution
    prior::Vector{Float64}
end

struct PoissonArrivals <: ArrivalDistribution 
    alpha::Float64
    beta::Float64
end

struct PitmanYorArrivals <: ArrivalDistribution
    tau::Float64
    theta::Float64
    function PitmanYorArrivals(;tau::Float64=0., theta::Float64=1.)
        if tau < 0 || tau > 1
            error("tau must be in (0,1)")
        end
        if theta <= -tau
            error("theta must be greater than -tau")
        end
        return new(tau, theta)
    end
end

abstract type ParametricArrivalsClusterParameters{T<:ArrivalDistribution} end

struct NtlParameters{T<:ArrivalDistribution} <: ParametricArrivalsClusterParameters{T}
    prior::Vector{Float64}
    arrival_distribution::T
end

struct BetaNtlParameters{T<:ArrivalDistribution} <: ParametricArrivalsClusterParameters{T}
    alpha::Float64
    arrival_distribution::T
    #function BetaNtlParameters{T}(arrival_distribution::T; alpha=0.) where {T<:ArrivalDistribution}
    #    return new(alpha, arrival_distribution)
    #end
end

abstract type ClusterParameters end

struct DpParameters <: ClusterParameters
    beta::Float64
    alpha::Float64
    function DpParameters(beta, alpha)
        return new(beta, alpha)
    end
    function DpParameters(alpha)
        return new(0, alpha)
    end
end

abstract type ClusterSufficientStatistics end

struct MixtureSufficientStatistics <: ClusterSufficientStatistics
    num_observations::Vector{Int64}
    cumulative_num_observations::FenwickTree{Int64}
    clusters::BitArray
end


abstract type HmmSufficientStatistics <: ClusterSufficientStatistics end

# Stationary HMM
mutable struct StationaryHmmSufficientStatistics <: HmmSufficientStatistics
    num_observations::Vector{Int64}
    state_num_observations::Matrix{Int64}
    clusters::BitArray
end

# Nonstationary HMM
mutable struct NonstationaryHmmSufficientStatistics <: HmmSufficientStatistics
    num_observations::Vector{Int64}
    state_num_observations::Matrix{Int64}
    state_assignments::Vector{SparseMatrixCSC}
    clusters::BitArray
end

struct ChangepointSufficientStatistics <: ClusterSufficientStatistics
    num_observations::Vector{Int64}
    clusters::BitArray 
end

abstract type DataSufficientStatistics end

struct FiniteTopicModelSufficientStatistics <: DataSufficientStatistics
    cluster_topic_frequencies::Matrix{Int64}
    cluster_topic_posterior::Matrix{Float64}
    topic_token_frequencies::Matrix{Int64}
    topic_token_posterior::Matrix{Float64}
end

struct GaussianSufficientStatistics <: DataSufficientStatistics
    data_means::Matrix{Float64}
    data_precision_quadratic_sums::Vector{Float64}
    posterior_means::Matrix{Float64}
    posterior_covs::Matrix{Float64}
    function GaussianSufficientStatistics(data_means::Matrix{Float64}, 
                                          data_precision_quadratic_sum::Vector{Float64},
                                          posterior_means::Matrix{Float64}, 
                                          posterior_covs::Matrix{Float64})
        # Make sure dimensions match
        if size(data_means)[1] !== size(posterior_means)[1]
            error("Dimension of data means and posterior means do not match")
        end
        dim = size(posterior_means)[1]
        if size(posterior_covs)[1] !== dim^2
            error("Dimension of covariances must be square of dimension of means")
        end
        return new(data_means, data_precision_quadratic_sum, posterior_means, posterior_covs)
    end
end

struct GaussianWishartSufficientStatistics <: DataSufficientStatistics
    data_means::Matrix{Float64}
    data_outer_product_sums::Matrix{Float64}
    posterior_means::Matrix{Float64}
    posterior_scale_matrix::Matrix{Float64}
    posterior_scale::Vector{Float64}
    posterior_dof::Vector{Float64}
end

struct MultinomialSufficientStatistics <: DataSufficientStatistics
    total_counts::Matrix{Int64}
    posterior_dirichlet_scale::Matrix{Float64}
    function MultinomialSufficientStatistics(total_counts::Matrix{Int64}, posterior_dirichlet_scale::Matrix{Float64})
        if any(total_counts .< 0)
            error("Each count in vector must be at least 0")
        end
        if any(posterior_dirichlet_scale .<= 0)
            error("Each value in dirichlet scale posterior must be greater than 0")
        end
        return new(total_counts, posterior_dirichlet_scale)
    end
end

struct SufficientStatistics{C<:ClusterSufficientStatistics, D<:DataSufficientStatistics}
    n::Int
    cluster::C
    data::D
end

abstract type Model end

struct Mixture{C<:Union{ClusterParameters, ParametricArrivalsClusterParameters, DpParameters}, D<:DataParameters} <: Model
    cluster_parameters::C
    data_parameters::D
end

struct HiddenMarkovModel{C<:Union{ClusterParameters, ParametricArrivalsClusterParameters}, D<:DataParameters} <: Model
    cluster_parameters::C
    data_parameters::D
end

struct Changepoint{C<:Union{ClusterParameters, ParametricArrivalsClusterParameters}, D<:DataParameters} <: Model
    cluster_parameters::C
    data_parameters::D
end

abstract type ArrivalsAuxillaryVariables end

struct GeometricAuxillaryVariables <: ArrivalsAuxillaryVariables end

mutable struct PoissonAuxillaryVariables <: ArrivalsAuxillaryVariables
    last_arrival_time::Int64
end

struct DpAuxillaryVariables <: ArrivalsAuxillaryVariables end

abstract type DataAuxillaryVariables end

struct GaussianAuxillaryVariables <: DataAuxillaryVariables end

struct MultinomialAuxillaryVariables <: DataAuxillaryVariables end

struct GaussianWishartAuxillaryVariables <: DataAuxillaryVariables end 

struct FiniteTopicModelAuxillaryVariables <: DataAuxillaryVariables 
    token_topic_assignments::Vector{Dict{Int64, Dict{Int64, Int64}}}
    document_topic_frequencies::Matrix 
end

struct AuxillaryVariables{A <: ArrivalsAuxillaryVariables, D <: DataAuxillaryVariables}
    arrivals::A
    data::D
end

end