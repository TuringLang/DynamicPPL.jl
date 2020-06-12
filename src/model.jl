"""
    struct Model{F,argnames,defaultnames,missings,Targs,Tdefaults}
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

julia> Model{(:y,)}(f, (x = 1.0, y = 2.0), (x = 42,))
Model{typeof(f),(:x, :y),(:x,),(:y,),Tuple{Float64,Float64},Tuple{Int64}}(f, (x = 1.0, y = 2.0), (x = 42,))
```
"""
struct Model{F,argnames,defaultnames,missings,Targs,Tdefaults} <: AbstractModel
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}

    """
        Model{missings}(f, args::NamedTuple, defaults::NamedTuple)

    Create a model with evaluation function `f` and missing arguments overwritten by `missings`.
    """
    function Model{missings}(
        f::F,
        args::NamedTuple{argnames,Targs},
        defaults::NamedTuple{defaultnames,Tdefaults},
    ) where {missings,F,argnames,Targs,defaultnames,Tdefaults}
        return new{F,argnames,defaultnames,missings,Targs,Tdefaults}(f, args, defaults)
    end
end

"""
    Model(f, args::NamedTuple[, defaults::NamedTuple = ()])

Create a model with evaluation function `f` and missing arguments deduced from `args`.

Default arguments `defaults` are used internally when constructing instances of the same
model with different arguments.
"""
@generated function Model(
    f::F,
    args::NamedTuple{argnames,Targs},
    defaults::NamedTuple = NamedTuple(),
) where {F,argnames,Targs}
    missings = Tuple(name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing)
    return :(Model{$missings}(f, args, defaults))
end

"""
    (model::Model)([rng, varinfo, sampler, context])

Sample from the `model` using the `sampler` with random number generator `rng` and the
`context`, and store the sample and log joint probability in `varinfo`.

The method resets the log joint probability of `varinfo` and increases the evaluation
number of `sampler`.
"""
function (model::Model)(args...)
    return model(VarInfo(), args...)
end

function (model::Model)(varinfo::AbstractVarInfo, args...)
    return model(Random.GLOBAL_RNG, varinfo, args...)
end

function (model::Model)(
    rng::Random.AbstractRNG,
    varinfo::AbstractVarInfo,
    sampler::AbstractSampler = SampleFromPrior(),
    context::AbstractContext = DefaultContext()
)
    if Threads.nthreads() == 1
        return evaluate_threadunsafe(rng, model, varinfo, sampler, context)
    else
        return evaluate_threadsafe(rng, model, varinfo, sampler, context)
    end
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
    if has_eval_num(sampler)
        sampler.state.eval_num += 1
    end
    return model.f(rng, model, varinfo, sampler, context)
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
    if has_eval_num(sampler)
        sampler.state.eval_num += 1
    end
    wrapper = ThreadSafeVarInfo(varinfo)
    result = model.f(rng, model, wrapper, sampler, context)
    setlogp!(varinfo, getlogp(wrapper))
    return result
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

getmissing(model::Model) = getmissings(model)
@deprecate getmissing(model) getmissings(model)

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
