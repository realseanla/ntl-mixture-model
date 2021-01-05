module Models

export ModelParameters
export ArrivalDistribution, GeometricArrivals
export ClusterParameters, NtlParameters, DpParameters
export DataParameters, GaussianParameters 
export SufficientStatistics, ClusterSufficientStatistics, DataSufficientStatistics
export NtlSufficientStatistics, GaussianSufficientStatistics

abstract type ModelParameters end

abstract type DataParameters <: ModelParameters end

struct GaussianParameters <: DataParameters
    data_covariance::Matrix
    prior_mean::Vector
    prior_covariance::Matrix
    function GaussianParameters(data_covariance::Matrix,
                                prior_mean::Vector,
                                prior_covariance::Matrix)
        if size(data_covariance) !== size(prior_covariance)
            error("Dimension of data and prior covariance must match")
        end
        dim = length(prior_mean)
        if dim !== size(prior_covariance)[1] || dim !== size(prior_covariance)[2] 
            error("Dimension of mean prior must match each dimension of prior covariance")
        end
        return new(data_covariance, prior_mean, prior_covariance)
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

end