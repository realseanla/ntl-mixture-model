module Tests

include("../ntl.jl")

import Random
using LinearAlgebra

function changepoint_gibbs_test(;n=10, iterations=10) 
    Random.seed!(1)
    dirichlet_scale = ones(Float64, 10)
    data_parameters = Ntl.Models.MultinomialParameters(20, dirichlet_scale)

    psi_prior = Vector{Float64}([1, 1])
    phi_prior = Vector{Float64}([1, 1])
    geometric_arrival = Ntl.Models.GeometricArrivals(phi_prior)
    ntl_cluster_parameters = Ntl.Models.NtlParameters(psi_prior, geometric_arrival)

    changepoint_model = Ntl.Models.Changepoint(ntl_cluster_parameters, data_parameters)

    changepoint = Ntl.Generate.generate(changepoint_model, n=n)
    data = Matrix(transpose(changepoint[:, 2:end]))

    gibbs_sampler = Ntl.Samplers.GibbsSampler(iterations)
    results = Ntl.Fitter.fit(data, changepoint_model, gibbs_sampler)
end

function changepoint_smc_test(;num_particles=10, ess_threshold=0.5)
    Random.seed!(1)
    data_covariance = Matrix{Float64}(0.1I, 2, 2)
    prior_covariance = Matrix{Float64}(I, 2, 2)
    prior_mean = Vector{Float64}(zeros(2))
    data_parameters = Ntl.Models.GaussianParameters(data_covariance, prior_mean, prior_covariance)
    
    psi_prior = Vector{Float64}([1, 1])
    phi_prior = Vector{Float64}([1, 1])
    geometric_arrival = Ntl.Models.GeometricArrivals(phi_prior)
    ntl_cluster_parameters = Ntl.Models.NtlParameters(psi_prior, geometric_arrival)
    
    changepoint_model = Ntl.Models.Changepoint(ntl_cluster_parameters, data_parameters)
    
    changepoint = Ntl.Generate.generate(changepoint_model, n=100)
    data = Matrix(transpose(changepoint[:, 2:end]))
    smc = Ntl.Samplers.SequentialMonteCarlo(num_particles=num_particles, ess_threshold=ess_threshold)
    results = Ntl.Fitter.fit(data, changepoint_model, smc)
end

function dp_hmm_test(;n=10, iterations=10)
    Random.seed!(1)
    
    data_covariance = Matrix{Float64}(0.1I, 2, 2)
    prior_covariance = Matrix{Float64}(I, 2, 2)
    prior_mean = Vector{Float64}(zeros(2))
    data_parameters = Ntl.Models.GaussianParameters(data_covariance, prior_mean, prior_covariance)
    
    psi_prior = Vector{Float64}([1, 1])
    phi_prior = Vector{Float64}([1, 1])
    geometric_arrival = Ntl.Models.GeometricArrivals(phi_prior)
    ntl_cluster_parameters = Ntl.Models.NtlParameters(psi_prior, geometric_arrival)
    
    mixture_model = Ntl.Models.Mixture(ntl_cluster_parameters, data_parameters)
    mixture = Ntl.Generate.generate(mixture_model, n=n)
    data = Matrix(transpose(mixture[:, 2:end]))

    dp_arrivals = Ntl.Models.PitmanYorArrivals()
    dp_parameters = Ntl.Models.BetaNtlParameters(0., dp_arrivals)
    gibbs_sampler = Ntl.Samplers.GibbsSampler(num_iterations=iterations)
    hmm_model = Ntl.Models.HiddenMarkovModel(dp_parameters, data_parameters)
    markov_chain = Ntl.Fitter.fit(data, hmm_model, gibbs_sampler)
end

function ntl_hmm_test(;n=10, iterations=10)
    Random.seed!(1)
    
    data_covariance = Matrix{Float64}(0.1I, 2, 2)
    prior_covariance = Matrix{Float64}(I, 2, 2)
    prior_mean = Vector{Float64}(zeros(2))
    data_parameters = Ntl.Models.GaussianParameters(data_covariance, prior_mean, prior_covariance)
    
    psi_prior = Vector{Float64}([1, 1])
    phi_prior = Vector{Float64}([1, 1])
    geometric_arrival = Ntl.Models.GeometricArrivals(phi_prior)
    ntl_cluster_parameters = Ntl.Models.NtlParameters(psi_prior, geometric_arrival)
    
    mixture_model = Ntl.Models.Mixture(ntl_cluster_parameters, data_parameters)
    mixture = Ntl.Generate.generate(mixture_model, n=n)
    data = Matrix(transpose(mixture[:, 2:end]))

    gibbs_sampler = Ntl.Samplers.GibbsSampler(num_iterations=iterations)
    hmm_model = Ntl.Models.HiddenMarkovModel(ntl_cluster_parameters, data_parameters)
    markov_chain = Ntl.Fitter.fit(data, hmm_model, gibbs_sampler)
end

end