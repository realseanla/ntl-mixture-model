module Models

export ModelParameters
export ArrivalDistribution, GeometricArrivals
export ClusterParameters, NtlParameters, DpParameters
export DataParameters, GaussianParameters, MultinomialParameters 
export SufficientStatistics, ClusterSufficientStatistics, DataSufficientStatistics
export NtlSufficientStatistics, GaussianSufficientStatistics, MultinomialParameters

abstract type ModelParameters end

abstract type DataParameters <: ModelParameters end

struct GaussianParameters <: DataParameters
    dim::Int64
    data_covariance::Matrix{Float64}
    prior_mean::Vector{Float64}
    prior_covariance::Matrix{Float64}
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
        return new(dim, data_covariance, prior_mean, prior_covariance)
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

abstract type ArrivalDistribution <: ModelParameters end

struct GeometricArrivals <: ArrivalDistribution
    prior::Vector{Float64}
end

abstract type ClusterParameters <: ModelParameters end

struct NtlParameters{T<:ArrivalDistribution} <: ClusterParameters
    prior::Vector{Float64}
    arrival_distribution::T
end

struct DpParameters <: ClusterParameters
    alpha::Float64
end

abstract type SufficientStatistics end

abstract type ClusterSufficientStatistics <: SufficientStatistics end

mutable struct NtlSufficientStatistics <: ClusterSufficientStatistics
    num_observations::Vector
    num_clusters::UInt64
end

abstract type DataSufficientStatistics <: SufficientStatistics end

struct GaussianSufficientStatistics <: DataSufficientStatistics
    data_means::Matrix{Float64}
    posterior_means::Matrix{Float64}
    posterior_covs::Matrix{Float64}
    function GaussianSufficientStatistics(data_means::Matrix{Float64}, 
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
        return new(data_means, posterior_means, posterior_covs)
    end
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

end