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

This package also uses the following R packages:
- [`mcclust`](https://cran.r-project.org/web/packages/mcclust/index.html)
- [`mcclust.ext`](https://github.com/sarawade/mcclust.ext)

## Usage

### 1. Specify the data parameters (i.e. the within-cluster distribution of the data X).

This implementation of the NTL mixture model accepts two types of data parameters.

Multivariate Gaussian data:
```
data_parameters = Ntl.Models.GaussianParameters(data_covariance, prior_mean, prior_covariance)
```
where `data_covariance::Matrix{Float64}` is the covariance of the observations arising from the same cluster, `prior_mean::Vector{Float64}` is the mean of the Multivariate Gaussian prior on the data mean, and `prior_covariance::Matrix{Float64}` is the covariance of the Multivariate Gaussian prior on the data mean.
The dimensions of `data_covariance`, `prior_mean`, and `prior_covariance` should be compatible -- if the dimension of `prior_mean` is `d`, then the dimensions of the matrices `data_covariance` and `prior_covariance` should be `d x d`.

Multinomial data:
```
data_parameters = Ntl.Models.MultinomialParameters(count, dirichlet_scale::Vector{Float64})
```
where `count::Int64` is the number of counts per observation (this is only used when simulating from the NTL mixture model prior with Multinomial data), and `dirichlet_scale::Vector{Float64}` is the parameter vector for the Dirichlet distribution prior on the probability parameter for each Multinomial distribution at eaach cluster.

### 2. Specify the cluster parameters.

This implementation allows for two types of cluster parameters.

Dirichlet Process cluster parameters:
```
cluster_parameters = Ntl.Models.DpParameters(alpha, sample_parameter_posterior=false)
```
where `alpha::Float64` is the value of the `alpha` parameter of the Dirichlet Process, and `sample_parameter_posterior::Bool` is a flag indicating whether to sample the posterior of the `alpha` parameter at each iteration of the Metropolis-within-Gibbs sampler.

NTL mixture model cluster parameters:
```
cluster_parameters = Ntl.Models.NtlParameters(psi_prior, arrivals_distribution::Ntl.Models.ArrivalDistribution), sample_parameter_posterior)
```
where `psi_prior::Vector{Float64}` is a vector of parameters for the Beta distribution prior on the stick breaking weights, `arrival_distribution<:Ntl.Models.ArrivalDistribution` is the chosen distribution for the arrival times of the clusters, and `sample_parameter_posterior::Bool` is a flag indicating whether to sample the posterior distribution of the parameters of the stick breaking weights.

Currently, the only supported `arrivals_distribution` are iid interarrivals distributed according to a Geometric distribution:
```
arrivals_distribution = Ntl.Models.GeometricArrivals(prior, sample_parameter_posterior)
```
where `prior::Vector{Float64}` is the parameters of the Beta distribution prior on the rate parameter of the Geometric distribution, and `sample_parameter_posterior::Bool` is a flag indicating whether to sample the posterior distribution of the rate parameter of the Geometric distribution.

### 3. Specify the mixture model.
```
mixture_model = Ntl.Models.Mixture(cluster_parameters, data_parameters)
```

### 4. (Optional) Simulate data from the prior distribution of the specified mixture model.

This only works when `cluster_parameters` has type `Ntl.Models.NtlParameters`.

```
simulated_data = Ntl.Generate.generate(mixture_model, n)
```
where `n::Int64` is the number of observations to generate from the prior.

The true clustering is in the first column:
```
true_clustering = simulated_data[:, 1]
```
The observations are the rest of the columns. You should transpose the matrix so that the data is in column-major order.
```
data = Matrix(transpose(Matrix(simulated_data[:, 2:end])))
```

### 5. Specify the Metropolis-within-Gibbs sampler to sample from the mixture model posterior.

This implementation supports two types of samplers: a Collapsed Gibbs sampler, and a Metropolis-Hastings sampler.

Collapsed Gibbs Sampler:
```
sampler = Ntl.Samplers.GibbsSampler(num_iterations, num_burn_in, assignment_types)
```
where `num_iterations::Int64` is the number of iterations to output, `num_burn_in::Int64` is the number of iterations to throw away in the beginning as burn-in (not recommended), and `assignment_types::Vector{String}` which specifies both the number of chains to run, as well as their initial cluster assignments.
For example, we could have
```
assignment_types = ["random", "all same cluster", "all different clusters"]
```
which corresponds to us running three chains, where the first chain randomly assigns observations to clusters, the second chain assigns all observations to one cluster, and the third chain assigns each observation to its own cluster.

The Metropolis-Hastings sampler:
```
sampler = Ntl.Samplers.MetropolisHastingsSampler(num_iterations, num_burn_in, proposal_radius, skip=1, adaptive=false, assignment_types)
```
where `num_iterations`, `num_burn_in`, and `assignment_types` are as before for the Collapsed Gibbs sampler.
`proposal_radius::Int64` is the width of the discrete Uniform distribution proposal (must be at least 1).
`skip::Int64` corresponds to how much thinning of the Markov Chain we do, where if `n = skip`, then we take every `n`th iteration for our final calculation (not recommended).
`adaptive::bool` indicates whether to use an adaptive proposal scheme for the Metropolis-Hastings sampler, where the width of the discrete uniform proposal increases if the acceptance rate is above 0.3, and discreases if it is below 0.3 (also not recommended).

The Metropolis-Hastings sampler is only compatible with `Ntl.ModelsNtlParameters` cluster parameters, at the moment.

### 6. Sample from the posterior.
Fit the mixture model by doing:
```
mcmc_output = Ntl.Fitter.fit(data, mixture_model, sampler)
```
`mcmc_output` is a dictionary containing the following fields:
- `"log likelihood"` contains the log likelihoods of each iteration, in the form of a Matrix with dimensions `num_iterations x num_chains`. The first chain corresponds to the initial cluster assignment in `assignment_types`, the second chain to the second cluster assignment in `assignment_types`, and so on.
- `"assignments"` correspond to the cluster assignments for each iteration and for each chain of the Markov Chain, in the form of an Array with dimensions `num_observations x num_iterations x num_chains`. The order of the chains corresponds to the order of the `assignment_types`, similarly to the `log likelihood` matrix.
- If `Ntl.Models.NtlParameters` cluster parameters are used, and the arrival distribution parameter has the `sample_parameter_posterior=true`, then there will be a field called `"arrival posterior"`, which contain draws from the arrival distribution parameter posterior for each iteration, in the form of an array with dimensions `parameter_dim x num_iterations x num_chains`, where `parameter_dim` is the dimension of the arrival distribution parameter.
- If `Ntl.Models.DpParameters` cluster parameters are used, and the flag `sample_parameter_posterior=true`, then there will be a field called `alpha` which corresponds to draws from the posterior distribution of the `alpha` parameter for each iteration, in the form of an array with dimensions `num_iterations x num_chains`.

