<h1 align="center">DynamicPPL.jl</h1>
<p align="center"><i>A domain-specific language and backend for probabilistic programming, used by <a href="https://github.com/TuringLang/Turing.jl">Turing.jl</a>.</i></p>
<p align="center">
<a href="https://turinglang.github.io/DynamicPPL.jl/stable"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Docs" /></a>
<a href="https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3ACI+branch%3Amain"><img src="https://github.com/TuringLang/DynamicPPL.jl/workflows/CI/badge.svg?branch=main" alt="CI" /></a>
<a href="https://codecov.io/gh/TuringLang/DynamicPPL.jl"><img src="https://codecov.io/gh/TuringLang/DynamicPPL.jl/branch/main/graph/badge.svg" alt="Code Coverage" /></a>
</p>

## Get started

Install Julia (see [the official Julia website](https://julialang.org/install/)), then run:

```julia
julia> using Pkg; Pkg.add("DynamicPPL");
```

DynamicPPL is the part of Turing.jl that deals with defining, running, and manipulating probabilistic models.
You can define models using the `@model` macro:

```julia
julia> using DynamicPPL, Distributions, LinearAlgebra, Random

julia> @model function linear_regression(x)
           α ~ Normal(0, 1)
           β ~ Normal(0, 1)
           σ² ~ truncated(Cauchy(0, 3); lower=0)
           y ~ MvNormal(α .+ β .* x, σ² * I)
           return y
       end
linear_regression (generic function with 2 methods)

julia> model = linear_regression([1.0, 2.0, 3.0]); # Create a model instance

julia> model = model | VarNamedTuple(y=[4.0, 5.0, 6.0]);  # Condition on observed data

julia> params = rand(Xoshiro(5), model)  # Sample from the prior
VarNamedTuple
├─ α => 0.017205046232868317
├─ β => 1.2623658511338067
└─ σ² => 4.384838499382732

julia> logjoint(model, params)  # Compute the log joint probability of the sampled parameters
-12.384715307629321
```

Most users will want to use DynamicPPL through [Turing.jl](https://github.com/TuringLang/Turing.jl), which provides inference algorithms that build on top of DynamicPPL models.
You can find tutorials and general information about Turing.jl at [**https://turinglang.org**](https://turinglang.org).

## For inference algorithm developers

DynamicPPL is intentionally designed to be extensible, and allows you to define custom behaviour for how inputs are supplied to models, and outputs collected from models.
If you are developing inference algorithms or other tools that work with probabilistic models, DynamicPPL provides several interfaces you can use and extend:

  - [**Initialisation strategies**](https://turinglang.github.io/DynamicPPL.jl/stable/init/) control how parameter values are generated (e.g. from the prior, or from a fixed set of parameters).
  - [**Transform strategies**](https://turinglang.github.io/DynamicPPL.jl/stable/transforms/) control whether parameters, and log-densities, are interpreted as being in transformed or untransformed space.
  - [**Accumulators**](https://turinglang.github.io/DynamicPPL.jl/stable/accs/overview/) collect information during model execution, such as log-densities and raw parameter values. You can define your own accumulators to gather custom information.
  - [**`LogDensityFunction`**](https://turinglang.github.io/DynamicPPL.jl/stable/ldf/overview/) wraps a model for use with [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl), providing efficient log-density and gradient evaluation for use with samplers.

Full API documentation and guides can be found at [**https://turinglang.github.io/DynamicPPL.jl/stable**](https://turinglang.github.io/DynamicPPL.jl/stable).

## Contributing

If you find any bugs or unintuitive behaviour, please do [open an issue](https://github.com/TuringLang/DynamicPPL.jl/issues)!
We are also very happy to receive pull requests.
Non-breaking changes (as defined by [SemVer](https://semver.org)) should target the `main` branch; breaking changes should target the `breaking` branch.

You can find us in the `#turing` channel on [Julia Slack](https://julialang.org/slack/) or on [Julia Discourse](https://discourse.julialang.org).

## Citing

DynamicPPL.jl can be cited with the following (although if you used Turing.jl generally, you should probably use [the Turing citations](https://github.com/TuringLang/Turing.jl?tab=readme-ov-file#citing-turingjl) instead):

[**DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models**](https://arxiv.org/abs/2002.02702)
Hong Ge, Kai Xu, Zoubin Ghahramani
arXiv preprint, 2020.

<details>
<summary>Expand for BibTeX</summary>
```bibtex
@article{ge2020dynamicppl,
  title={DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models},
  author={Ge, Hong and Xu, Kai and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2002.02702},
  year={2020}
}
```

</details>
