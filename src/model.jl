"""
    struct Model{F,argnames,defaultnames,missings,Targs,Tdefaults}
        name::Symbol
        f::F
        args::NamedTuple{argnames,Targs}
        defaults::NamedTuple{defaultnames,Tdefaults}
    end

A `Model` struct with model evaluation function of type `F`, arguments of names `argnames`
types `Targs`, default arguments of names `defaultnames` with types `Tdefaults`, and missing
arguments `missings`.

Here `argnames`, `defaultargnames`, and `missings` are tuples of symbols, e.g. `(:a, :b)`.

An argument with a type of `Missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in `missings`
are treated as random variables rather than observations.

The default arguments are used internally when constructing instances of the same model with
different arguments.

# Examples

```julia
julia> Model(f, (x = 1.0, y = 2.0))
Model{typeof(f),(:x, :y),(),(),Tuple{Float64,Float64},Tuple{}}(f, (x = 1.0, y = 2.0), NamedTuple())

julia> Model(f, (x = 1.0, y = 2.0), (x = 42,))
Model{typeof(f),(:x, :y),(:x,),(),Tuple{Float64,Float64},Tuple{Int64}}(f, (x = 1.0, y = 2.0), (x = 42,))

julia> Model{(:y,)}(f, (x = 1.0, y = 2.0), (x = 42,)) # with special definition of missings
Model{typeof(f),(:x, :y),(:x,),(:y,),Tuple{Float64,Float64},Tuple{Int64}}(f, (x = 1.0, y = 2.0), (x = 42,))
```
"""
struct Model{F,argnames,defaultnames,missings,Targs,Tdefaults} <: AbstractModel
    name::Symbol
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}

    """
        Model{missings}(name::Symbol, f, args::NamedTuple, defaults::NamedTuple)

    Create a model of name `name` with evaluation function `f` and missing arguments
    overwritten by `missings`.
    """
    function Model{missings}(
        name::Symbol,
        f::F,
        args::NamedTuple{argnames,Targs},
        defaults::NamedTuple{defaultnames,Tdefaults},
    ) where {missings,F,argnames,Targs,defaultnames,Tdefaults}
        return new{F,argnames,defaultnames,missings,Targs,Tdefaults}(name, f, args, defaults)
    end
end

"""
    Model(name::Symbol, f, args::NamedTuple[, defaults::NamedTuple = ()])

Create a model of name `name` with evaluation function `f` and missing arguments deduced
from `args`.

Default arguments `defaults` are used internally when constructing instances of the same
model with different arguments.
"""
@generated function Model(
    name::Symbol,
    f::F,
    args::NamedTuple{argnames,Targs},
    defaults::NamedTuple = NamedTuple(),
) where {F,argnames,Targs}
    missings = Tuple(name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing)
    return :(Model{$missings}(name, f, args, defaults))
end

"""
    (model::Model)([rng, varinfo, sampler, context])

Sample from the `model` using the `sampler` with random number generator `rng` and the
`context`, and store the sample and log joint probability in `varinfo`.

The method resets the log joint probability of `varinfo` and increases the evaluation
number of `sampler`.
"""
function (model::Model)(
    rng::Random.AbstractRNG,
    varinfo::AbstractVarInfo = VarInfo(),
    sampler::AbstractSampler = SampleFromPrior(),
    context::AbstractContext = DefaultContext(),
)
    if Threads.nthreads() == 1
        return evaluate_threadunsafe(rng, model, varinfo, sampler, context)
    else
        return evaluate_threadsafe(rng, model, varinfo, sampler, context)
    end
end
function (model::Model)(args...)
    return model(Random.GLOBAL_RNG, args...)
end

# without VarInfo
function (model::Model)(
    rng::Random.AbstractRNG,
    sampler::AbstractSampler,
    args...,
)
    return model(rng, VarInfo(), sampler, args...)
end

# without VarInfo and without AbstractSampler
function (model::Model)(rng::Random.AbstractRNG, context::AbstractContext)
    return model(rng, VarInfo(), SampleFromPrior(), context)
end

"""
    evaluate_threadunsafe(rng, model, varinfo, sampler, context)

Evaluate the `model` without wrapping `varinfo` inside a `ThreadSafeVarInfo`.

If the `model` makes use of Julia's multithreading this will lead to undefined behaviour.
This method is not exposed and supposed to be used only internally in DynamicPPL.

See also: [`evaluate_threadsafe`](@ref)
"""
function evaluate_threadunsafe(rng, model, varinfo, sampler, context)
    resetlogp!(varinfo)
    return _evaluate(rng, model, varinfo, sampler, context)
end

"""
    evaluate_threadsafe(rng, model, varinfo, sampler, context)

Evaluate the `model` with `varinfo` wrapped inside a `ThreadSafeVarInfo`.

With the wrapper, Julia's multithreading can be used for observe statements in the `model`
but parallel sampling will lead to undefined behaviour.
This method is not exposed and supposed to be used only internally in DynamicPPL.

See also: [`evaluate_threadunsafe`](@ref)
"""
function evaluate_threadsafe(rng, model, varinfo, sampler, context)
    resetlogp!(varinfo)
    wrapper = ThreadSafeVarInfo(varinfo)
    result = _evaluate(rng, model, wrapper, sampler, context)
    setlogp!(varinfo, getlogp(wrapper))
    return result
end

"""
    _evaluate(rng, model::Model, varinfo, sampler, context)

Evaluate the `model` with the arguments matching the given `sampler` and `varinfo` object.
"""
@generated function _evaluate(rng, model::Model{_F,argnames}, varinfo, sampler, context) where {_F,argnames}
    unwrap_args = [:($matchingvalue(sampler, varinfo, model.args.$var)) for var in argnames]
    return :(model.f(rng, model, varinfo, sampler, context, $(unwrap_args...)))
end

"""
    getargnames(model::Model)

Get a tuple of the argument names of the `model`.
"""
getargnames(model::Model{_F,argnames}) where {argnames,_F} = argnames


"""
    getmissings(model::Model)

Get a tuple of the names of the missing arguments of the `model`.
"""
getmissings(model::Model{_F,_a,_d,missings}) where {missings,_F,_a,_d} = missings

"""
    nameof(model::Model)

Get the name of the `model` as `Symbol`.
"""
Base.nameof(model::Model) = model.name

"""
    logjoint(model::Model, varinfo::AbstractVarInfo)

Return the log joint probability of variables `varinfo` for the probabilistic `model`.

See [`logjoint`](@ref) and [`loglikelihood`](@ref).
"""
function logjoint(model::Model, varinfo::AbstractVarInfo)
    model(varinfo, SampleFromPrior(), DefaultContext())
    return getlogp(varinfo)
end

"""
    logprior(model::Model, varinfo::AbstractVarInfo)

Return the log prior probability of variables `varinfo` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).
"""
function logprior(model::Model, varinfo::AbstractVarInfo)
    model(varinfo, SampleFromPrior(), PriorContext())
    return getlogp(varinfo)
end

"""
    loglikelihood(model::Model, varinfo::AbstractVarInfo)

Return the log likelihood of variables `varinfo` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`logprior`](@ref).
"""
function Distributions.loglikelihood(model::Model, varinfo::AbstractVarInfo)
    model(varinfo, SampleFromPrior(), LikelihoodContext())
    return getlogp(varinfo)
end

"""
    generated_quantities(model::Model, chain::AbstractChains)

Execute `model` for each of the samples in `chain` and return an array of the values
returned by the `model` for each sample.

# Examples
## General
Often you might have additional quantities computed inside the model that you want to
inspect, e.g.
```julia
@model function demo(x)
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()
    return interesting_quantity(θ, x)
end
m = demo(data)
chain = sample(m, alg, n)
# To inspect the `interesting_quantity(θ, x)` where `θ` is replaced by samples
# from the posterior/`chain`:
generated_quantities(m, chain) # <= results in a `Vector` of returned values
                               #    from `interesting_quantity(θ, x)`
```
## Concrete (and simple)
```julia
julia> using DynamicPPL, Turing

julia> @model function demo(xs)
           s ~ InverseGamma(2, 3)
           m_shifted ~ Normal(10, √s)
           m = m_shifted - 10

           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end

           return (m, )
       end
demo (generic function with 1 method)

julia> model = demo(randn(10));

julia> chain = sample(model, MH(), 10);

julia> generated_quantities(model, chain)
10×1 Array{Tuple{Float64},2}:
 (2.1964758025119338,)
 (2.1964758025119338,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.043088571494005024,)
 (-0.16489786710222099,)
 (-0.16489786710222099,)
```
"""
function generated_quantities(model::Model, chain::AbstractChains)
    varinfo = VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        setval!(varinfo, chain, sample_idx, chain_idx)
        model(varinfo)
    end
end
