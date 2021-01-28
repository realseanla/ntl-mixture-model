module Samplers

export GibbsSampler, SequentialMonteCarlo

abstract type Sampler end

struct GibbsSampler <: Sampler
    num_iterations::Int64
    burn_in::Int64
    function GibbsSampler(;num_iterations::Int64=1000, burn_in::Int64=0)
        if num_iterations < 1
            error("Number of iterations should be positive.")
        end
        if burn_in >= num_iterations
            error("Burn-in should be less than number of iterations.")
        end
        return new(num_iterations, burn_in)
    end
    function GibbsSampler(;num_iterations::Int64=1000)
        return new(num_iterations, 0)
    end
end

struct SequentialMonteCarlo <: Sampler
    num_particles::Int64
    ess_threshold::Float64
    resample_at_end::Bool
    function SequentialMonteCarlo(;num_particles::Int64=1000, ess_threshold::T=0.5, resample_at_end::Bool=false) where 
                                  {T <: Real}
        if num_particles < 1
            error("Number of particles should be positive.")
        end
        if ess_threshold < 0
            error("ESS Threshold should be non-negative.")
        end
        if typeof(ess_threshold) <: AbstractFloat && ess_threshold > 1
            error("ESS Threshold should be either a non-negative integer or a fraction")
        end
        if ess_threshold < 1
            ess_threshold = num_particles*ess_threshold
        end
        return new(num_particles, ess_threshold, resample_at_end)
    end
end

end