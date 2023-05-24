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
    defaults::NamedTuple,
    context::AbstractContext=DefaultContext(),
) where {F,argnames,Targs}
    missings = Tuple(name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing)
    return :(Model{$missings}(f, args, defaults, context))
end

function Model(f, args::NamedTuple, context::AbstractContext=DefaultContext(); kwargs...)
    return Model(f, args, NamedTuple(kwargs), context)
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
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> m, x = model(); (m ≠ 1.0 && x ≠ 100.0)
true

julia> # Create a new instance which treats `x` as observed
       # with value `100.0`, and similarly for `m=1.0`.
       conditioned_model = condition(model, x=100.0, m=1.0);

julia> m, x = conditioned_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only condition on `x = 100.0`.
       conditioned_model = condition(model, x = 100.0);

julia> m, x =conditioned_model(); (m ≠ 1.0 && x == 100.0)
true

julia> # We can also use the nicer `|` syntax.
       conditioned_model = model | (x = 100.0, );

julia> m, x = conditioned_model(); (m ≠ 1.0 && x == 100.0)
true
```

The above uses a `NamedTuple` to hold the conditioning variables, which allows us to perform some
additional optimizations; in many cases, the above has zero runtime-overhead.

But we can also use a `Dict`, which offers more flexibility in the conditioning
(see examples further below) but generally has worse performance than the `NamedTuple`
approach:

```jldoctest condition
julia> conditioned_model_dict = condition(model, Dict(@varname(x) => 100.0));

julia> m, x = conditioned_model_dict(); (m ≠ 1.0 && x == 100.0)
true

julia> # There's also an option using `|` by letting the right-hand side be a tuple
       # with elements of type `Pair{<:VarName}`, i.e. `vn => value` with `vn isa VarName`.
       conditioned_model_dict = model | (@varname(x) => 100.0, );

julia> m, x = conditioned_model_dict(); (m ≠ 1.0 && x == 100.0)
true
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
demo_mv (generic function with 4 methods)

julia> model = demo_mv();

julia> conditioned_model = condition(model, m = [missing, 1.0]);

julia> # (✓) `m[1]` sampled while `m[2]` is fixed
       m = conditioned_model(); (m[1] ≠ 1.0 && m[2] == 1.0)
true
```

Intuitively one might also expect to be able to write `model | (m[1] = 1.0, )`.
Unfortunately this is not supported as it has the potential of increasing compilation
times but without offering any benefit with respect to runtime:

```jldoctest condition
julia> # (×) `m[2]` is not set to 1.0.
       m = condition(model, var"m[2]" = 1.0)(); m[2] == 1.0
false
```

But you _can_ do this if you use a `Dict` as the underlying storage instead:

```jldoctest condition
julia> # Alternatives:
       # - `model | (@varname(m[2]) => 1.0,)`
       # - `condition(model, Dict(@varname(m[2] => 1.0)))`
       # (✓) `m[2]` is set to 1.0.
       m = condition(model, @varname(m[2]) => 1.0)(); (m[1] ≠ 1.0 && m[2] == 1.0)
true
```

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

julia> model() ≠ 1.0
true

julia> conditioned_model = model | (m = 1.0, );

julia> conditioned_model()
1.0
```

But one needs to be careful when prefixing variables in the nested models:

```jldoctest condition
julia> @model function demo_outer_prefix()
           @submodel prefix="inner" m = demo_inner()
           return m
       end
demo_outer_prefix (generic function with 2 methods)

julia> # (×) This doesn't work now!
       conditioned_model = demo_outer_prefix() | (m = 1.0, );

julia> conditioned_model() == 1.0
false

julia> # (✓) `m` in `demo_inner` is referred to as `inner.m` internally, so we do:
       conditioned_model = demo_outer_prefix() | (var"inner.m" = 1.0, );

julia> conditioned_model()
1.0

julia> # Note that the above `var"..."` is just standard Julia syntax:
       keys((var"inner.m" = 1.0, ))
(Symbol("inner.m"),)
```

And similarly when using `Dict`:

```jldoctest condition
julia> conditioned_model_dict = demo_outer_prefix() | (@varname(var"inner.m") => 1.0);

julia> conditioned_model_dict()
1.0
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
function AbstractPPL.condition(model::Model, value, values...)
    return contextualize(model, condition(model.context, value, values...))
end

"""
    decondition(model::Model)
    decondition(model::Model, variables...)

Return a `Model` for which `variables...` are _not_ considered observations.
If no `variables` are provided, then all variables currently considered observations
will no longer be.

This is essentially the inverse of [`condition`](@ref). This also means that
it suffers from the same limitiations.

Note that currently we only support `variables` to take on explicit values
provided to `condition.

# Examples
```jldoctest decondition
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> conditioned_model = condition(demo(), m = 1.0, x = 10.0);

julia> conditioned_model()
(m = 1.0, x = 10.0)

julia> # By specifying the `VarName` to `decondition`.
       model = decondition(conditioned_model, @varname(m));

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # When `NamedTuple` is used as the underlying, you can also provide
       # the symbol directly (though the `@varname` approach is preferable if
       # if the variable is known at compile-time).
       model = decondition(conditioned_model, :m);

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # `decondition` multiple at once:
       (m, x) = decondition(model, :m, :x)(); (m ≠ 1.0 && x ≠ 10.0)
true

julia> # `decondition` without any symbols will `decondition` all variables.
       (m, x) = decondition(model)(); (m ≠ 1.0 && x ≠ 10.0)
true

julia> # Usage of `Val` to perform `decondition` at compile-time if possible
       # is also supported.
       model = decondition(conditioned_model, Val{:m}());

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true
```

Similarly when using a `Dict`:

```jldoctest decondition
julia> conditioned_model_dict = condition(demo(), @varname(m) => 1.0, @varname(x) => 10.0);

julia> conditioned_model_dict()
(m = 1.0, x = 10.0)

julia> deconditioned_model_dict = decondition(conditioned_model_dict, @varname(m));

julia> (m, x) = deconditioned_model_dict(); m ≠ 1.0 && x == 10.0
true
```

But, as mentioned, `decondition` is only supported for variables explicitly
provided to `condition` earlier;

```jldoctest decondition
julia> @model function demo_mv(::Type{TV}=Float64) where {TV}
           m = Vector{TV}(undef, 2)
           m[1] ~ Normal()
           m[2] ~ Normal()
           return m
       end
demo_mv (generic function with 4 methods)

julia> model = demo_mv();

julia> conditioned_model = condition(model, @varname(m) => [1.0, 2.0]);

julia> conditioned_model()
2-element Vector{Float64}:
 1.0
 2.0

julia> deconditioned_model = decondition(conditioned_model, @varname(m[1]));

julia> deconditioned_model()  # (×) `m[1]` is still conditioned
2-element Vector{Float64}:
 1.0
 2.0

julia> # (✓) this works though
       deconditioned_model_2 = deconditioned_model | (@varname(m[1]) => missing);

julia> m = deconditioned_model_2(); (m[1] ≠ 1.0 && m[2] == 2.0)
true
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
VarName[]
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
    return evaluate!!(model, Random.default_rng(), args...)
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
function _evaluate!!(model::Model, varinfo::AbstractVarInfo, context::AbstractContext)
    args, kwargs = make_evaluate_args_and_kwargs(model, varinfo, context)
    return model.f(args...; kwargs...)
end

"""
    make_evaluate_args_and_kwargs(model, varinfo, context)

Return the arguments and keyword arguments to be passed to the evaluator of the model, i.e. `model.f`e.
"""
@generated function make_evaluate_args_and_kwargs(
    model::Model{_F,argnames}, varinfo::AbstractVarInfo, context::AbstractContext
) where {_F,argnames}
    unwrap_args = [
        if is_splat_symbol(var)
            :($matchingvalue(context_new, varinfo, model.args.$var)...)
        else
            :($matchingvalue(context_new, varinfo, model.args.$var))
        end for var in argnames
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
        args = (
            model,
            # Maybe perform `invlink!!` once prior to evaluation to avoid
            # lazy `invlink`-ing of the parameters. This can be useful for
            # speeding up computation. See docs for `maybe_invlink_before_eval!!`
            # for more information.
            maybe_invlink_before_eval!!(varinfo, context_new, model),
            context_new,
            $(unwrap_args...),
        )
        kwargs = model.defaults
        return args, kwargs
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
    rand([rng=Random.default_rng()], [T=NamedTuple], model::Model)

Generate a sample of type `T` from the prior distribution of the `model`.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{T}, model::Model) where {T}
    x = last(
        evaluate!!(
            model,
            SimpleVarInfo{Float64}(OrderedDict()),
            SamplingContext(rng, SampleFromPrior(), DefaultContext()),
        ),
    )
    return values_as(x, T)
end

# Default RNG and type
Base.rand(rng::Random.AbstractRNG, model::Model) = rand(rng, NamedTuple, model)
Base.rand(::Type{T}, model::Model) where {T} = rand(Random.default_rng(), T, model)
Base.rand(model::Model) = rand(Random.default_rng(), NamedTuple, model)

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
