module Tests

include("../ntl.jl")

import Random
using LinearAlgebra

function changepoint_test(;n=10, iterations=10) 
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

end