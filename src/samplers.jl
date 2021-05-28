module Samplers

export GibbsSampler, SequentialMonteCarlo, SequentialImportanceSampler, MetropolisHastingsSampler
export MetropolisHastingsSampler, MetropolisWithinGibbsSampler

abstract type Sampler end

abstract type MetropolisWithinGibbsSampler <: Sampler  end

struct GibbsSampler <: MetropolisWithinGibbsSampler
    num_iterations::Int64
    num_burn_in::Int64
    skip::Int64
    random_assignment::Bool
    assignment_type::String
    function GibbsSampler(;num_iterations::Int64=1000, num_burn_in::Int64=0, skip::Int64=1, 
                          random_assignment=true, assignment_type="same")
        if num_iterations < 1
            error("Number of iterations should be positive.")
        end
        if !random_assignment && !(assignment_type in ["same", "different"])
            error("Initial assignment can only be one of 'same' or 'different'")
        end
        return new(num_iterations, num_burn_in, skip, random_assignment, assignment_type)
    end
end

mutable struct MetropolisHastingsSampler <: MetropolisWithinGibbsSampler 
    num_iterations::Int64 
    num_burn_in::Int64
    proposal_radius::Int64 
    skip::Int64
    adaptive::Bool
    random_assignment::Bool
    assignment_type::String
    function MetropolisHastingsSampler(;num_iterations=1000, num_burn_in=1000, proposal_radius=5, skip=1, 
                                       adaptive=false, random_assignment=true, assignment_type="same")
        if !random_assignment && !(assignment_type in ["same", "different"])
            error("Initial assignment can only be one of 'same' or 'different'")
        end
        return new(num_iterations, num_burn_in, proposal_radius, skip, adaptive, random_assignment, assignment_type)
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

end