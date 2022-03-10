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

julia> Model{(:y,)}(f, (x = 1.0, y = 2.0), (x = 42,)) # with special definition of missings
Model{typeof(f),(:x, :y),(:x,),(:y,),Tuple{Float64,Float64},Tuple{Int64}}(f, (x = 1.0, y = 2.0), (x = 42,))
```
"""
struct Model{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx<:AbstractContext} <:
       AbstractProbabilisticProgram
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}
    context::Ctx

    @doc """
        Model{missings}(f, args::NamedTuple, defaults::NamedTuple)

    Create a model with evaluation function `f` and missing arguments overwritten by
    `missings`.
    """
    function Model{missings}(
        f::F,
        args::NamedTuple{argnames,Targs},
        defaults::NamedTuple{defaultnames,Tdefaults},
        context::Ctx=DefaultContext(),
    ) where {missings,F,argnames,Targs,defaultnames,Tdefaults,Ctx}
        return new{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx}(
            f, args, defaults, context
        )
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
    defaults::NamedTuple=NamedTuple(),
    context::AbstractContext=DefaultContext(),
) where {F,argnames,Targs}
    missings = Tuple(name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing)
    return :(Model{$missings}(f, args, defaults, context))
end

function contextualize(model::Model, context::AbstractContext)
    return Model(model.f, model.args, model.defaults, context)
end

"""
    model | (x = 1.0, ...)

Return a `Model` which now treats variables on the right-hand side as observations.

See [`condition`](@ref) for more information and examples.
"""
Base.:|(model::Model, values) = condition(model, values)

"""
    condition(model::Model; values...)
    condition(model::Model, values::NamedTuple)

Return a `Model` which now treats the variables in `values` as observations.

See also: [`decondition`](@ref), [`conditioned`](@ref)

# Limitations

This does currently _not_ work with variables that are
provided to the model as arguments, e.g. `@model function demo(x) ... end`
means that `condition` will not affect the variable `x`.

Therefore if one wants to make use of `condition` and [`decondition`](@ref)
one should not be specifying any random variables as arguments.

This is done for the sake of backwards compatibility.

# Examples
## Simple univariate model
```jldoctest condition
julia> using Distributions; using StableRNGs; rng = StableRNG(42); # For reproducibility.

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> model(rng)
(m = -0.6702516921145671, x = -0.22312984965118443)

julia> # Create a new instance which treats `x` as observed
       # with value `100.0`, and similarly for `m=1.0`.
       conditioned_model = condition(model, x=100.0, m=1.0);

julia> conditioned_model(rng)
(m = 1.0, x = 100.0)

julia> # Let's only condition on `x = 100.0`.
       conditioned_model = condition(model, x = 100.0);

julia> conditioned_model(rng)
(m = 1.3736306979834252, x = 100.0)

julia> # We can also use the nicer `|` syntax.
       conditioned_model = model | (x = 100.0, );

julia> conditioned_model(rng)
(m = 1.3095394956381083, x = 100.0)
```

## Condition only a part of a multivariate variable

Not only can be condition on multivariate random variables, but
we can also use the standard mechanism of setting something to `missing`
in the call to `condition` to only condition on a part of the variable.

```jldoctest condition
julia> @model function demo_mv(::Type{TV}=Float64) where {TV}
           m = Vector{TV}(undef, 2)
           m[1] ~ Normal()
           m[2] ~ Normal()
           return m
       end
demo_mv (generic function with 3 methods)

julia> model = demo_mv();

julia> conditioned_model = condition(model, m = [missing, 1.0]);

julia> conditioned_model(rng) # (✓) `m[1]` sampled, `m[2]` is fixed
2-element Vector{Float64}:
 0.12607002180931043
 1.0
```

Intuitively one might also expect to be able to write `model | (x[1] = 1.0, )`.
Unfortunately this is not supported due to performance.

```jldoctest condition
julia> condition(model, var"x[2]" = 1.0)(rng) # (×) `x[2]` is not set to 1.0.
2-element Vector{Float64}:
  0.683947930996541
 -1.019202452456547
```

We will likely provide some syntactic sugar for this in the future.

## Nested models

`condition` of course also supports the use of nested models through
the use of [`@submodel`](@ref).

```jldoctest condition
julia> @model demo_inner() = m ~ Normal()
demo_inner (generic function with 2 methods)

julia> @model function demo_outer()
           @submodel m = demo_inner()
           return m
       end
demo_outer (generic function with 2 methods)

julia> model = demo_outer();

julia> model(rng)
-0.7935128416361353

julia> conditioned_model = model | (m = 1.0, );

julia> conditioned_model(rng)
1.0
```

But one needs to be careful when prefixing variables in the nested models:

```jldoctest condition
julia> @model function demo_outer_prefix()
           @submodel prefix="inner" m = demo_inner()
           return m
       end
demo_outer_prefix (generic function with 2 methods)

julia> # This doesn't work now!
       conditioned_model = demo_outer_prefix() | (m = 1.0, );

julia> conditioned_model(rng)
1.7747246334368165

julia> # `m` in `demo_inner` is referred to as `inner.m` internally, so we do:
       conditioned_model = demo_outer_prefix() | (var"inner.m" = 1.0, );

julia> conditioned_model(rng)
1.0

julia> # Note that the above `var"..."` is just standard Julia syntax:
       keys((var"inner.m" = 1.0, ))
(Symbol("inner.m"),)
```

The difference is maybe more obvious once we look at how these different
in their trace/`VarInfo`:

```jldoctest condition
julia> keys(VarInfo(demo_outer()))
1-element Vector{VarName{:m, Setfield.IdentityLens}}:
 m

julia> keys(VarInfo(demo_outer_prefix()))
1-element Vector{VarName{Symbol("inner.m"), Setfield.IdentityLens}}:
 inner.m
```

From this we can tell what the correct way to condition `m` within `demo_inner`
is in the two different models.

"""
AbstractPPL.condition(model::Model; values...) = condition(model, NamedTuple(values))
function AbstractPPL.condition(model::Model, values)
    return contextualize(model, condition(model.context, values))
end

"""
    decondition(model::Model)
    decondition(model::Model, syms...)

Return a `Model` for which `syms...` are _not_ considered observations.
If no `syms` are provided, then all variables currently considered observations
will no longer be.

This is essentially the inverse of [`condition`](@ref). This also means that
it suffers from the same limitiations.

# Examples
```jldoctest
julia> using Distributions; using StableRNGs; rng = StableRNG(42); # For reproducibility.

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> conditioned_model = condition(demo(), m = 1.0, x = 10.0);

julia> conditioned_model(rng)
(m = 1.0, x = 10.0)

julia> model = decondition(conditioned_model, :m);

julia> model(rng)
(m = -0.6702516921145671, x = 10.0)

julia> # `decondition` multiple at once:
       decondition(model, :m, :x)(rng)
(m = 0.4471218424633827, x = 1.820752540446808)

julia> # `decondition` without any symbols will `decondition` all variables.
       decondition(model)(rng)
(m = 1.3095394956381083, x = 1.4356095174474188)

julia> # Usage of `Val` to perform `decondition` at compile-time if possible
       # is also supported.
       model = decondition(conditioned_model, Val{:m}());

julia> model(rng)
(m = 0.683947930996541, x = 10.0)
```
"""
function AbstractPPL.decondition(model::Model, syms...)
    return contextualize(model, decondition(model.context, syms...))
end

"""
    observations(model::Model)

Alias for [`conditioned`](@ref).
"""
observations(model::Model) = conditioned(model)

"""
    conditioned(model::Model)

Return `NamedTuple` of values that are conditioned on under `model`.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: conditioned, contextualize

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns all the variables we have conditioned on + their values.
       conditioned(condition(m, x=100.0, m=1.0))
(x = 100.0, m = 1.0)

julia> # Nested ones also work (note that `PrefixContext` does nothing to the result).
       cm = condition(contextualize(m, PrefixContext{:a}(condition(m=1.0))), x=100.0);

julia> conditioned(cm)
(x = 100.0, m = 1.0)

julia> # Since we conditioned on `m`, not `a.m` as it will appear after prefixed,
       # `a.m` is treated as a random variable.
       keys(VarInfo(cm))
1-element Vector{VarName{Symbol("a.m"), Setfield.IdentityLens}}:
 a.m

julia> # If we instead condition on `a.m`, `m` in the model will be considered an observation.
       cm = condition(contextualize(m, PrefixContext{:a}(condition(var"a.m"=1.0))), x=100.0);

julia> conditioned(cm).x
100.0

julia> conditioned(cm).var"a.m"
1.0

julia> keys(VarInfo(cm)) # <= no variables are sampled
Any[]
```
"""
conditioned(model::Model) = conditioned(model.context)

"""
    (model::Model)([rng, varinfo, sampler, context])

Sample from the `model` using the `sampler` with random number generator `rng` and the
`context`, and store the sample and log joint probability in `varinfo`.

The method resets the log joint probability of `varinfo` and increases the evaluation
number of `sampler`.
"""
(model::Model)(args...) = first(evaluate!!(model, args...))

"""
    use_threadsafe_eval(context::AbstractContext, varinfo::AbstractVarInfo)

Return `true` if evaluation of a model using `context` and `varinfo` should
wrap `varinfo` in `ThreadSafeVarInfo`, i.e. threadsafe evaluation, and `false` otherwise.
"""
function use_threadsafe_eval(context::AbstractContext, varinfo::AbstractVarInfo)
    return Threads.nthreads() > 1
end

"""
    evaluate!!(model::Model[, rng, varinfo, sampler, context])

Sample from the `model` using the `sampler` with random number generator `rng` and the
`context`, and store the sample and log joint probability in `varinfo`.

Returns both the return-value of the original model, and the resulting varinfo.

The method resets the log joint probability of `varinfo` and increases the evaluation
number of `sampler`.
"""
function AbstractPPL.evaluate!!(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext
)
    return if use_threadsafe_eval(context, varinfo)
        evaluate_threadsafe!!(model, varinfo, context)
    else
        evaluate_threadunsafe!!(model, varinfo, context)
    end
end

function AbstractPPL.evaluate!!(
    model::Model,
    rng::Random.AbstractRNG,
    varinfo::AbstractVarInfo=VarInfo(),
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    return evaluate!!(model, varinfo, SamplingContext(rng, sampler, context))
end

function AbstractPPL.evaluate!!(model::Model, context::AbstractContext)
    return evaluate!!(model, VarInfo(), context)
end

function AbstractPPL.evaluate!!(model::Model, args...)
    return evaluate!!(model, Random.GLOBAL_RNG, args...)
end

# without VarInfo
function AbstractPPL.evaluate!!(
    model::Model, rng::Random.AbstractRNG, sampler::AbstractSampler, args...
)
    return evaluate!!(model, rng, VarInfo(), sampler, args...)
end

# without VarInfo and without AbstractSampler
function AbstractPPL.evaluate!!(
    model::Model, rng::Random.AbstractRNG, context::AbstractContext
)
    return evaluate!!(model, rng, VarInfo(), SampleFromPrior(), context)
end

"""
    evaluate_threadunsafe!!(model, varinfo, context)

Evaluate the `model` without wrapping `varinfo` inside a `ThreadSafeVarInfo`.

If the `model` makes use of Julia's multithreading this will lead to undefined behaviour.
This method is not exposed and supposed to be used only internally in DynamicPPL.

See also: [`evaluate_threadsafe!!`](@ref)
"""
function evaluate_threadunsafe!!(model, varinfo, context)
    return _evaluate!!(model, resetlogp!!(varinfo), context)
end

"""
    evaluate_threadsafe!!(model, varinfo, context)

Evaluate the `model` with `varinfo` wrapped inside a `ThreadSafeVarInfo`.

With the wrapper, Julia's multithreading can be used for observe statements in the `model`
but parallel sampling will lead to undefined behaviour.
This method is not exposed and supposed to be used only internally in DynamicPPL.

See also: [`evaluate_threadunsafe!!`](@ref)
"""
function evaluate_threadsafe!!(model, varinfo, context)
    wrapper = ThreadSafeVarInfo(resetlogp!!(varinfo))
    result, wrapper_new = _evaluate!!(model, wrapper, context)
    return result, setlogp!!(wrapper_new.varinfo, getlogp(wrapper_new))
end

"""
    _evaluate!!(model::Model, varinfo, context)

Evaluate the `model` with the arguments matching the given `context` and `varinfo` object.
"""
@generated function _evaluate!!(
    model::Model{_F,argnames}, varinfo, context
) where {_F,argnames}
    unwrap_args = [
        :($matchingvalue(context_new, varinfo, model.args.$var)) for var in argnames
    ]
    # We want to give `context` precedence over `model.context` while also
    # preserving the leaf context of `context`. We can do this by
    # 1. Set the leaf context of `model.context` to `leafcontext(context)`.
    # 2. Set leaf context of `context` to the context resulting from (1).
    # The result is:
    # `context` -> `childcontext(context)` -> ... -> `model.context`
    #  -> `childcontext(model.context)` -> ... -> `leafcontext(context)`
    return quote
        context_new = setleafcontext(
            context, setleafcontext(model.context, leafcontext(context))
        )
        model.f(model, varinfo, context_new, $(unwrap_args...))
    end
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
Base.nameof(model::Model) = Symbol(model.f)
Base.nameof(model::Model{<:Function}) = nameof(model.f)

"""
    rand([rng=Random.GLOBAL_RNG], [T=NamedTuple], model::Model)

Generate a sample of type `T` from the prior distribution of the `model`.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{T}, model::Model) where {T}
    x = last(
        evaluate!!(
            model,
            SimpleVarInfo{Float64}(),
            SamplingContext(rng, SampleFromPrior(), DefaultContext()),
        ),
    )
    return DynamicPPL.values_as(x, T)
end

# Default RNG and type
Base.rand(rng::Random.AbstractRNG, model::Model) = rand(rng, NamedTuple, model)
Base.rand(::Type{T}, model::Model) where {T} = rand(Random.GLOBAL_RNG, T, model)
Base.rand(model::Model) = rand(Random.GLOBAL_RNG, NamedTuple, model)

"""
    logjoint(model::Model, varinfo::AbstractVarInfo)

Return the log joint probability of variables `varinfo` for the probabilistic `model`.

See [`logjoint`](@ref) and [`loglikelihood`](@ref).
"""
function logjoint(model::Model, varinfo::AbstractVarInfo)
    return getlogp(last(evaluate!!(model, varinfo, DefaultContext())))
end

"""
    logprior(model::Model, varinfo::AbstractVarInfo)

Return the log prior probability of variables `varinfo` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).
"""
function logprior(model::Model, varinfo::AbstractVarInfo)
    return getlogp(last(evaluate!!(model, varinfo, PriorContext())))
end

"""
    loglikelihood(model::Model, varinfo::AbstractVarInfo)

Return the log likelihood of variables `varinfo` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`logprior`](@ref).
"""
function Distributions.loglikelihood(model::Model, varinfo::AbstractVarInfo)
    return getlogp(last(evaluate!!(model, varinfo, LikelihoodContext())))
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
        setval_and_resample!(varinfo, chain, sample_idx, chain_idx)
        model(varinfo)
    end
end

"""
    generated_quantities(model::Model, parameters::NamedTuple)
    generated_quantities(model::Model, values, keys)
    generated_quantities(model::Model, values, keys)

Execute `model` with variables `keys` set to `values` and return the values returned by the `model`.

If a `NamedTuple` is given, `keys=keys(parameters)` and `values=values(parameters)`.

# Example
```jldoctest
julia> using DynamicPPL, Distributions

julia> @model function demo(xs)
           s ~ InverseGamma(2, 3)
           m_shifted ~ Normal(10, √s)
           m = m_shifted - 10
           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end
           return (m, )
       end
demo (generic function with 2 methods)

julia> model = demo(randn(10));

julia> parameters = (; s = 1.0, m_shifted=10);

julia> generated_quantities(model, parameters)
(0.0,)

julia> generated_quantities(model, values(parameters), keys(parameters))
(0.0,)
```
"""
function generated_quantities(model::Model, parameters::NamedTuple)
    varinfo = VarInfo(model)
    setval_and_resample!(varinfo, values(parameters), keys(parameters))
    return model(varinfo)
end

function generated_quantities(model::Model, values, keys)
    varinfo = VarInfo(model)
    setval_and_resample!(varinfo, values, keys)
    return model(varinfo)
end
