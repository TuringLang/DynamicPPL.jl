# DynamicPPL.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/DynamicPPL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/DynamicPPL.jl/dev)
[![CI](https://github.com/TuringLang/DynamicPPL.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![JuliaNightly](https://github.com/TuringLang/DynamicPPL.jl/workflows/JuliaNightly/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3AJuliaNightly+branch%3Amaster)
[![IntegrationTest](https://github.com/TuringLang/DynamicPPL.jl/workflows/IntegrationTest/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3AIntegrationTest+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/DynamicPPL.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/DynamicPPL.jl?branch=master)
[![Codecov](https://codecov.io/gh/TuringLang/DynamicPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/DynamicPPL.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://colprac.sciml.ai/)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/24589)

A domain-specific language and backend for probabilistic programming languages, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).

## Do you want to contribute?

If you feel you have some relevant skills and are interested in contributing then please do get in touch and open an issue on Github.

### Contributor's Guide

This project follows the [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://colprac.sciml.ai/), apart from the following slight variation:

- The master branch contains the most recent release at any point in time. All non-breaking changes (bug fixes etc.) are merged directly into master and a new patch version is released immediately.
- A separate dev branch contains all breaking changes, and is merged into master when a minor version release happens.

For instance, suppose we are currently on version 0.13.5.

- If someone produces a bug fix, it is merged directly into master and bumps the version to 0.13.6. This change is also merged into dev so that it remains up-to-date with master.
- If someone is working on a new feature that is not breaking (performance-related, fancy new syntax that is backwards-compatible etc.), the same happens.
- New breaking changes are merged into dev until a release is ready to go, at which point dev is merged into master and version 0.14 is released.

### Bors

This project uses [Bors](https://bors.tech/) for merging PRs. Bors is a Github bot that prevents merge skew / semantic merge conflicts by testing
the exact integration of pull requests before merging them.

When a PR is good enough for merging and has been approved by at least one reviewer, instead of merging immediately, it is added to the merge queue
by commenting with `bors r+`. The Bors bot merges the pull request into a staging area, and runs the CI tests. If tests pass, the commit in the staging
area is copied to the target branch (i.e., usually master).

PRs can be tested by adding a comment with `bors try`. Additional commands can be found in the [Bors documentation](https://bors.tech/documentation/).

### Tutorial: Metropolis sampler

Let's use DynamicPPL to create a simple Metropolis sampler. DynamicPPL is the part of Turing dealing with models, likelihoods, probabilities, and all that stuff. Before reading this, you should already know:
1. How to write models in Turing.jl.
2. What MCMC is.

Let's start by creating a basic model.

```
using DynamicPPL, Distributions, Random


rng = MersenneTwister(1776);  # Set seed for reproducibility

# This model assumes our sample follows a normal distribution 
# with a standard deviation of 1 and an unknown mean.
@model function demo(x)
   μ ~ Normal(0, 1)  # Our prior is a standard normal distribution
   x .~ Normal(μ, 1)
   return nothing
end

# We instantiate the model by calling it on a dataset.
m = demo([-1, 0, 1, 1])
```

#### Models

What *is* a DynamicPPL model? This is an important question, because the way DynamicPPL treats models can be very different from how you'd intuitively think of them.

DynamicPPL is a *procedural* programming language, which means we think of a model as a series of instructions (a procedure). This procedure modifies or returns `VarInfo` objects, which hold samples taken from a probability distribution. Here's an example of a VarInfo:

```
# Calling `SimpleVarInfo(model )` creates a `SimpleVarInfo` with the correct type and fields 
# by sampling one from the prior.
x = SimpleVarInfo(m)
```

The procedural approach is a bit different from the object-oriented approach. We don't try to reason through a model by thinking of each `~` statement as an object, for instance.

Procedural programming also differs from functional programming, where we think of a model as a pure mathematical function (like a probability density). A DynamicPPL model does not consist of pure function calls: each line (`~` statement) in a model behaves differently depending on the state of the model (the sampler and context).

In DynamicPPL, we execute a model using the `DynamicPPL.evaluate!!` function. `DynamicPPL.evaluate!!` will always return a tuple consisting of the return value for the function (in our case `nothing`) and a `VarInfo`. (It will also modify `VarInfo` if `VarInfo` is mutable.)

Let's give a quick example of what a `Context` is and does. If we call a model with a `SamplingContext`, for example, it creates a new random sample from the prior:
```
_, x1 = DynamicPPL.evaluate!!(m, SimpleVarInfo(m), SamplingContext(SampleFromPrior()))
```

On the other hand, if we call a model with a `LikelihoodContext` and a preexisting `VarInfo`, the model evaluates the likelihood function (ignoring the prior) and inserts it into the `logp` field, leaving the sample unchanged. Note that the value of `logp` is now different:
```
_, x2 = DynamicPPL.evaluate!!(m, deepcopy(x1), LikelihoodContext())
```

And the value of `logp` for `x1` is equal to the likelihood plus the prior:
```
_, x3 = DynamicPPL.evaluate!!(m, deepcopy(x1), PriorContext())
getlogp(x1) ≈ getlogp(x2) + getlogp(x3)  # returns true
```

Some contexts can be nested. For instance, `SamplingContext` can be nested with a `PriorContext` to insert the log-prior, rather than the log-posterior into `logp`.
```
_, x4 = DynamicPPL.evaluate!!(m, SamplingContext(PriorContext()))
```

By default, we evaluate the log-posterior. This can be specified explicitly using `DefaultContext()`.
```
_, x5 = DynamicPPL.evaluate!!(m, deepcopy(x1), DefaultContext()); x5 == x1
```

#### Example: A simple sampler (say 5 times fast)

Let's create a Metropolis-Hastings Sampler to see how this works. (Note that to interact with the rest of the Turing ecosystem, a sampler must interface with AbstractMCMC.jl -- this tutorial ignores that step.)
```
function sample(rng, model::DynamicPPL.Model, kernel::Distribution, n_steps::Int)
	# First we create a SimpleVarInfo by sampling from the prior, to initialize the model. For convenience, SimpleVarInfo(model) samples from the prior by default.
	init = SimpleVarInfo(model)
	# Now we use a function barrier to let Julia infer the correct types for `vi` -- if we don't include one, `vi` may be slow.
	return metropolis(rng, init, model, kernel, n_steps)
end


function metropolis(
	rng,
	init::DynamicPPL.AbstractVarInfo, 
	model::DynamicPPL.Model, 
	kernel::Distribution, 
	n_steps::Int
)
	samples = Vector{typeof(init)}(undef, n_steps)
	current = init
	for step in 1:n_steps
		proposal = map(current.values) do x
			x + rand(rng, kernel)
		end
		proposal = SimpleVarInfo(proposal)
		_, proposal = DynamicPPL.evaluate!!(model, proposal, DefaultContext())
		
		log_p = getlogp(current) - getlogp(proposal)
		accept = randexp(rng) > log_p
		
		samples[step] = accept ? proposal : current
		current = samples[step]
	end
	return samples
end
```

And now we can see that if we sample, we get the right mean!
```
samples = sample(rng, m, Normal(0, .05), 1_000_000)
means = getindex.(samples, (@varname(μ),))  # ~0.2
```


