module Samplers

export GibbsSampler, SequentialMonteCarlo, SequentialImportanceSampler, MetropolisHastingsSampler

abstract type Sampler end

struct GibbsSampler <: Sampler
    num_iterations::Int64
    num_burn_in::Int64
    function GibbsSampler(;num_iterations::Int64=1000, num_burn_in::Int64=0)
        if num_iterations < 1
            error("Number of iterations should be positive.")
        end
        if num_burn_in >= num_iterations
            error("Burn-in should be less than number of iterations.")
        end
        return new(num_iterations, num_burn_in)
    end
    function GibbsSampler(;num_iterations::Int64=1000)
        return new(num_iterations, 0)
    end
end

struct SequentialMonteCarlo <: Sampler
    num_particles::Int64
    ess_threshold::Float64
    function SequentialMonteCarlo(;num_particles::Int64=1000, ess_threshold::T=0.5) where 
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
        return new(num_particles, ess_threshold)
    end
end

struct SequentialImportanceSampler <: Sampler
    num_particles::Int64
    function SequentialImportanceSampler(;num_particles=1000)
        return new(num_particles)
    end
end

struct MetropolisHastingsSampler <: Sampler 
    num_iterations::Int64 
    num_burn_in::Int64
    radius::Int64 
end

end