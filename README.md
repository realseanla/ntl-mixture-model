# The Neutral-to-the-Left Mixture Model
A Julia implementation of the NTL mixture model.

## Installation
Make sure the following Julia packages have been installed using `Pkg`:
- [`Statistics`](https://docs.julialang.org/en/v1/stdlib/Statistics/)
- [`Distributions`](https://juliastats.org/Distributions.jl/stable/)
- [`LinearAlgebra`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
- [`ProgressBars`](https://github.com/cloud-oak/ProgressBars.jl)
- [`ProgressMeter`](https://github.com/timholy/ProgressMeter.jl)
- [`SparseArrays`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
- [`SpecialFunctions`](https://specialfunctions.juliamath.org/stable/)
- [`DataStructures`](https://juliacollections.github.io/DataStructures.jl/latest/)
- [`Clustering`](https://juliastats.org/Clustering.jl/stable/)
- [`DataFrames`](https://dataframes.juliadata.org/stable/)
- [`RCall`](https://juliainterop.github.io/RCall.jl/stable/)
- [`Combinatorics`](https://github.com/JuliaMath/Combinatorics.jl)

This package also uses the following R package:
- [`mcclust`](https://cran.r-project.org/web/packages/mcclust/index.html)
- [`mcclust.ext`](https://github.com/sarawade/mcclust.ext)

## Usage

1. Specify the data parameters (i.e. the within-cluster distribution of the data X).

This implementation of the NTL mixture model accepts two types of data parameters:

Multivariate Gaussian data.
```
Ntl.Models.GaussianParameters(data_covariance::Matrix{Float64}, prior_mean::Vector{Float64}, prior_covariance::Matrix{Float64})
```
where `data_covariance` is the covariance of the observations arising from the same cluster, `prior_mean` is the mean of the Multivariate Gaussian prior on the data mean, and `prior_covariance` is the covariance of the Multivariate Gaussian prior on the data mean.
The dimensions of `data_covariance`, `prior_mean`, and `prior_covariance` should be compatible -- if the dimension of `prior_mean` is `d`, then the dimensions of the matrices `data_covariance` and `prior_covariance` should be `d x d`.

Multinomial data.
```
Ntl.Models.MultinomialParameters(count::Int64, dirichlet_scale::Vector{Float64})
```
where `count` is the number of counts per observation (this is only used when simulating from the NTL mixture model prior with Multinomial data), and `dirichlet_scale` is the parameter vector for the Dirichlet distribution prior on the probability parameter for each Multinomial distribution at eaach cluster.

2. Specify the cluster parameters.

This implementation accepts two types of cluster parameters:

Dirichlet Process cluster parameters.
```
Ntl.Models.DpParameters(alpha::Float64, sample_parameter_posterior::Bool=false)
```
where `alpha` is the value of the `alpha` parameter of the Dirichlet Process, and `sample_parameter_posterior` is a flag indicating whether to sample the posterior of the `alpha` parameter at each iteration of the Metropolis-within-Gibbs sampler.
