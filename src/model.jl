"""
    struct Model{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx<:AbstractContext,Threaded}
        f::F
        args::NamedTuple{argnames,Targs}
        defaults::NamedTuple{defaultnames,Tdefaults}
        context::Ctx=DefaultContext()
    end

A `Model` struct with model evaluation function of type `F`, arguments of names `argnames`
types `Targs`, default arguments of names `defaultnames` with types `Tdefaults`, missing
arguments `missings`, and evaluation context of type `Ctx`.

Here `argnames`, `defaultargnames`, and `missings` are tuples of symbols, e.g. `(:a, :b)`.
`context` is by default `DefaultContext()`.

An argument with a type of `Missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in `missings`
are treated as random variables rather than observations.

The `Threaded` type parameter indicates whether the model requires threadsafe evaluation
(i.e., whether the model contains statements which modify the internal VarInfo that are
executed in parallel). By default, this is set to `false`.

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
struct Model{
    F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx<:AbstractContext,Threaded
} <: AbstractProbabilisticProgram
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}
    context::Ctx

    @doc """
        Model{Threaded,missings}(f, args::NamedTuple, defaults::NamedTuple)

    Create a model with evaluation function `f` and missing arguments overwritten by
    `missings`.
    """
    function Model{Threaded,missings}(
        f::F,
        args::NamedTuple{argnames,Targs},
        defaults::NamedTuple{defaultnames,Tdefaults},
        context::Ctx=DefaultContext(),
    ) where {missings,F,argnames,Targs,defaultnames,Tdefaults,Ctx,Threaded}
        return new{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,Threaded}(
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
@generated function Model{Threaded}(
    f::F,
    args::NamedTuple{argnames,Targs},
    defaults::NamedTuple{kwargnames,Tkwargs},
    context::AbstractContext=DefaultContext(),
) where {Threaded,F,argnames,Targs,kwargnames,Tkwargs}
    missing_args = Tuple(
        name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing
    )
    missing_kwargs = Tuple(
        name for (name, typ) in zip(kwargnames, Tkwargs.types) if typ <: Missing
    )
    return :(Model{Threaded,$(missing_args..., missing_kwargs...)}(
        f, args, defaults, context
    ))
end

function Model{Threaded}(
    f, args::NamedTuple, context::AbstractContext=DefaultContext(); kwargs...
) where {Threaded}
    return Model{Threaded}(f, args, NamedTuple(kwargs), context)
end

"""
    requires_threadsafe(model::Model)

Return whether `model` has been marked as needing threadsafe evaluation (using
`setthreadsafe`).
"""
function requires_threadsafe(
    ::Model{F,A,D,M,Ta,Td,Ctx,Threaded}
) where {F,A,D,M,Ta,Td,Ctx,Threaded}
    return Threaded
end

"""
    contextualize(model::Model, context::AbstractContext)

Return a new `Model` with the same evaluation function and other arguments, but
with its underlying context set to `context`.
"""
function contextualize(model::Model, context::AbstractContext)
    return Model{requires_threadsafe(model)}(model.f, model.args, model.defaults, context)
end

"""
    setleafcontext(model::Model, context::AbstractContext)

Return a new `Model` with its leaf context set to `context`. This is a convenience shortcut
for `contextualize(model, setleafcontext(model.context, context)`).
"""
function setleafcontext(model::Model, context::AbstractContext)
    return contextualize(model, setleafcontext(model.context, context))
end

"""
    setthreadsafe(model::Model, threadsafe::Bool)

Returns a new `Model` with its threadsafe flag set to `threadsafe`.

Threadsafe evaluation ensures correctness when executing model statements that mutate the
internal `VarInfo` object in parallel. For example, this is needed if tilde-statements are
nested inside `Threads.@threads` or similar constructs.

It is not needed for generic multithreaded operations that don't involve VarInfo. For
example, calculating a log-likelihood term in parallel and then calling `@addlogprob!`
outside of the parallel region is safe without needing to set `threadsafe=true`.

It is also not needed for multithreaded sampling with AbstractMCMC's `MCMCThreads()`.

Setting `threadsafe` to `true` increases the overhead in evaluating the model. Please see
[the Turing.jl docs](https://turinglang.org/docs/usage/threadsafe-evaluation/) for more
details.
"""
function setthreadsafe(model::Model{F,A,D,M}, threadsafe::Bool) where {F,A,D,M}
    return if requires_threadsafe(model) == threadsafe
        model
    else
        Model{threadsafe,M}(model.f, model.args, model.defaults, model.context)
    end
end

"""
    model | (x = 1.0, ...)

Return a `Model` which now treats variables on the right-hand side as observations.

See [`condition`](@ref) for more information and examples.
"""
Base.:|(model::Model, values::Union{NamedTuple,AbstractDict,Pair,Tuple,VarNamedTuple}) =
    condition(model, values)

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

julia> m, x = model(); (m != 1.0 && x != 100.0)
true

julia> # Create a new instance which treats `x` as observed
       # with value `100.0`, and similarly for `m=1.0`.
       conditioned_model = condition(model, x=100.0, m=1.0);

julia> m, x = conditioned_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only condition on `x = 100.0`.
       conditioned_model = condition(model, x = 100.0);

julia> m, x = conditioned_model(); (m != 1.0 && x == 100.0)
true

julia> # We can also use the nicer `|` syntax.
       conditioned_model = model | (x = 100.0, );

julia> m, x = conditioned_model(); (m != 1.0 && x == 100.0)
true
```

In the above we have specified the conditioning variables via keyword arguments. You can also
provide a `NamedTuple`, `AbstractDict{<:VarName}`, or a `VarNamedTuple`; internally these are
all converted to a `VarNamedTuple`.

For example, here we use a `Dict`:

```jldoctest condition
julia> conditioned_model_dict = condition(model, Dict(@varname(x) => 100.0));

julia> m, x = conditioned_model_dict(); (m != 1.0 && x == 100.0)
true

julia> # There's also an option using `|` by letting the right-hand side be a tuple
       # with elements of type `Pair{<:VarName}`, i.e. `vn => value` with `vn isa VarName`.
       conditioned_model_pairs = model | (@varname(x) => 100.0);

julia> m, x = conditioned_model_pairs(); (m != 1.0 && x == 100.0)
true
```

## Condition only a part of a multivariate variable

When conditioning on multiple variables at a time, we can use `missing` to signal that a
part of the variable should not be conditioned on.

However, note that in this case each element of the multivariate random variable must be on
its own tilde-statement. In other words, if we write `m ~ MvNormal(...)`, then we cannot
condition on only `m[1]`. (In principle, for some distributions this can be possible,
specifically when the distribution can be factorised into independent components, like an
MvNormal with a diagonal covariance matrix. However, this is not currently implemented.)

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
       m = conditioned_model(); (m[1] != 1.0 && m[2] == 1.0)
true
```

Intuitively one might also expect to be able to write `model | (m[2] = 1.0, )`. You cannot
do this with a `NamedTuple` because the `VarName` `m[2]` cannot be represented as a `Symbol`
(i.e., `Symbol("m[2]")` is not the same as `@varname(m[2])`).

```jldoctest condition
julia> # (×) `m[2]` is not set to 1.0.
       m = condition(model, var"m[2]" = 1.0)(); m[2] == 1.0
false
```

But you _can_ do this if you use a `Dict` or a `VarNamedTuple` as the underlying storage
instead:

```jldoctest condition
julia> vnt = @vnt begin
           @template m = zeros(2)
           m[2] := 1.0
       end
VarNamedTuple
└─ m => PartialArray size=(2,) data::Vector{Float64}
        └─ (2,) => 1.0

julia> m = condition(model, vnt)(); (m[1] != 1.0 && m[2] == 1.0)
true
```

## Nested models

`condition` also supports the use of nested models through the use of [`to_submodel`](@ref).

```jldoctest condition
julia> @model demo_inner() = m ~ Normal()
demo_inner (generic function with 2 methods)

julia> @model function demo_outer()
           # By default, `to_submodel` prefixes the variables using the left-hand side of `~`.
           inner ~ to_submodel(demo_inner())
           return inner
       end
demo_outer (generic function with 2 methods)

julia> model = demo_outer();

julia> model() ≠ 1.0
true

julia> # To condition the variable inside `demo_inner` we need to refer to it as `inner.m`.
       conditioned_model = model | (@varname(inner.m) => 1.0, );

julia> conditioned_model()
1.0

julia> # If you attempt to condition on `inner` itself, it must refer to the prefixed
       # latent variables, not the return value. For example, this will work:
       conditioned_model2 = model | (inner = (m = 1.0,), );

julia> conditioned_model2()
1.0

julia> # However, if `inner` does not contain `m` inside it as a field, this will not
       # result in any conditioning:
       conditioned_model_fail = model | (inner = "something else", );

julia> conditioned_model_fail == 1.0
false
```
"""
function AbstractPPL.condition(model::Model, values...)
    # Positional arguments - need to handle cases carefully
    return contextualize(
        model, CondFixContext{Condition}(_make_condfix_values(values...), model.context)
    )
end
function AbstractPPL.condition(model::Model; values...)
    return contextualize(
        model, CondFixContext{Condition}(VarNamedTuple(NamedTuple(values)), model.context)
    )
end

"""
    _make_condfix_values(vals...)

Convert different types of input to a `VarNamedTuple` of values, suitable for storage in a
`CondFixContext`.

This handles all the cases where `vals` is either already a `NamedTuple` or `AbstractDict`
(e.g. `model | (x=1, y=2)`), as well as if they are splatted (e.g. `condition(model, x=1,
y=2)`).
"""
_make_condfix_values(values::NamedTuple) = VarNamedTuple(values)
_make_condfix_values(values::VarNamedTuple) = values
_make_condfix_values(values::AbstractDict{<:VarName}) = VarNamedTuple(pairs(values))
function _make_condfix_values(values::Pair{<:Union{VarName,Symbol}}...)
    pairs = map(
        v -> ((v.first isa Symbol ? VarName{v.first}() : v.first) => v.second), values
    )
    return VarNamedTuple(pairs)
end
function _make_condfix_values(values::NTuple{N,Pair{<:Union{VarName,Symbol}}}) where {N}
    return _make_condfix_values(values...)
end

"""
    decondition(model::Model)
    decondition(model::Model, variables...)

Return a `Model` for which `variables...` are _not_ conditioned on. If no `variables` are
provided, then all conditioned variables will be removed.

Note that this function cannot decondition variables that are provided as arguments to the
model function itself; it can only decondition variables that were provided via `condition`
or `|`.

This is essentially the inverse of [`condition`](@ref).

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

julia> # `decondition` also accepts symbols, although VarNames are preferable for
       # type stability reasons.
       model = decondition(conditioned_model, :m);

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # `decondition` multiple at once:
       (m, x) = decondition(model, :m, :x)(); (m ≠ 1.0 && x ≠ 10.0)
true

julia> # `decondition` without any symbols will `decondition` all variables.
       (m, x) = decondition(model)(); (m ≠ 1.0 && x ≠ 10.0)
true
```

Note that `decondition` is only guaranteed to work when you decondition variables that were
explicitly provided to `condition` earlier. In this example we condition on `@varname(m)`
but decondition on `@varname(m[1])`, which fails because `m[1]` was not explicitly
conditioned on:

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
```
"""
function AbstractPPL.decondition(model::Model, syms::Union{Symbol,VarName}...)
    return contextualize(model, decondition_context(model.context, syms...))
end

"""
    conditioned(model::Model)

Return the conditioned values in `model`.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: conditioned, contextualize, PrefixContext, CondFixContext, Condition

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns all the variables we have conditioned on + their values.
       conditioned(condition(m, x=100.0, m=1.0))
VarNamedTuple
├─ x => 100.0
└─ m => 1.0

julia> # Nested ones also work. (Note that `PrefixContext` also prefixes
       # the variables of any `CondFixContext` that is _inside_ it.)
       new_context = PrefixContext(@varname(a), CondFixContext{Condition}(VarNamedTuple(m=1.0,)));
       cm = condition(contextualize(m, new_context), x=100.0);

julia> conditioned(cm)
VarNamedTuple
├─ a => VarNamedTuple
│       └─ m => 1.0
└─ x => 100.0

julia> # Since we conditioned on `a.m`, it is not treated as a random variable.
       # However, `a.x` will still be a random variable.
       keys(VarInfo(cm))
1-element Vector{VarName}:
 a.x

julia> # We can also condition on `a.m` _outside_ of the PrefixContext:
       cm = condition(contextualize(m, PrefixContext(@varname(a))), (@varname(a.m) => 1.0));

julia> conditioned(cm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Now `a.x` will be sampled.
       keys(VarInfo(cm))
1-element Vector{VarName}:
 a.x
```
"""
conditioned(model::Model) = conditioned(model.context)

"""
    fix(model::Model; values...)
    fix(model::Model, values::NamedTuple)

Return a `Model` which now treats the variables in `values` as fixed.

See also: [`unfix`](@ref), [`fixed`](@ref)

# Examples
## Simple univariate model
```jldoctest fix
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
       fixed_model = fix(model, x=100.0, m=1.0);

julia> m, x = fixed_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only fix on `x = 100.0`.
       fixed_model = fix(model, x = 100.0);

julia> m, x = fixed_model(); (m != 1.0 && x == 100.0)
true
```

## Other ways of specifying fixed values

Specifying fixed values can be done exactly in the same way as for [`condition`](@ref);
please see its docstring for more examples.

## Difference from `condition`

The only difference between fixing and conditioning is as follows:

- Conditioned variables are considered to be observations, and are thus included in the
  computation log-joint and log-likelihood, but not the log-prior.
- Fixed variables are considered to be constant, and are thus not included
  in any log-probability computations.

```jldoctest; setup=:(using DynamicPPL, Distributions)
julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> model_fixed = fix(model, m = 1.0);

julia> model_conditioned = condition(model, m = 1.0);

julia> logjoint(model_fixed, (x=1.0,))
-0.9189385332046728

julia> logjoint(model_conditioned, (x=1.0,))
-2.3378770664093453

julia> # The difference is the missing log-probability of `m`:
       logpdf(Normal(), 1.0)
-1.4189385332046727
```
"""
function fix(model::Model, values...)
    return contextualize(
        model, CondFixContext{Fix}(_make_condfix_values(values...), model.context)
    )
end
function fix(model::Model; values...)
    return contextualize(
        model, CondFixContext{Fix}(VarNamedTuple(NamedTuple(values)), model.context)
    )
end

"""
    unfix(model::Model)
    unfix(model::Model, variables...)

Return a `Model` for which `variables...` are _not_ considered fixed. If no `variables` are
provided, then all fixed variables will be removed.

This is essentially the inverse of [`fix`](@ref).

Conceptually this is very similar to [`decondition`](@ref) and thus the same limitations
apply; please see its docstring for more details.

# Examples
```jldoctest unfix
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> fixed_model = fix(demo(), m = 1.0, x = 10.0);

julia> fixed_model()
(m = 1.0, x = 10.0)

julia> # By specifying the `VarName` to `unfix`.
       model = unfix(fixed_model, @varname(m));

julia> (m, x) = model(); (m != 1.0 && x == 10.0)
true

julia> # When `NamedTuple` is used as the underlying, you can also provide
       # the symbol directly (though the `@varname` approach is preferable if
       # if the variable is known at compile-time).
       model = unfix(fixed_model, :m);

julia> (m, x) = model(); (m != 1.0 && x == 10.0)
true

julia> # `unfix` multiple at once:
       (m, x) = unfix(model, :m, :x)(); (m != 1.0 && x != 10.0)
true

julia> # `unfix` without any symbols will `unfix` all variables.
       (m, x) = unfix(model)(); (m != 1.0 && x != 10.0)
true
```
"""
unfix(model::Model, syms::Union{Symbol,VarName}...) =
    contextualize(model, unfix_context(model.context, syms...))

"""
    fixed(model::Model)

Return the fixed values in `model`.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: fixed, contextualize, PrefixContext, CondFixContext, Fix

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns all the variables we have fixed on + their values.
       fixed(fix(m, x=100.0, m=1.0))
VarNamedTuple
├─ x => 100.0
└─ m => 1.0

julia> # The rest of this is the same as the `condition` example above.
       fm = fix(contextualize(m, PrefixContext(@varname(a), CondFixContext{Fix}(VarNamedTuple(m=1.0)))), x=100.0);

julia> Set(keys(fixed(fm))) == Set([@varname(a.m), @varname(x)])
true

julia> keys(VarInfo(fm))
1-element Vector{VarName}:
 a.x

julia> # We can also fix `a.m` _outside_ of the PrefixContext:
       fm = fix(contextualize(m, PrefixContext(@varname(a))), (@varname(a.m) => 1.0));

julia> fixed(fm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Now `a.x` will be sampled.
       keys(VarInfo(fm))
1-element Vector{VarName}:
 a.x
```
"""
fixed(model::Model) = fixed(model.context)

"""
    prefix(model::Model, x::VarName)
    prefix(model::Model, x::Val{sym})
    prefix(model::Model, x::Any)

Return `model` but with all random variables prefixed by `x`, where `x` is either:
- a `VarName` (e.g. `@varname(a)`),
- a `Val{sym}` (e.g. `Val(:a)`), or
- for any other type, `x` is converted to a Symbol and then to a `VarName`. Note that
  this will introduce runtime overheads so is not recommended unless absolutely
  necessary.

# Examples

```jldoctest
julia> using DynamicPPL: prefix

julia> @model demo() = x ~ Dirac(1)
demo (generic function with 2 methods)

julia> rand(prefix(demo(), @varname(my_prefix)))
VarNamedTuple
└─ my_prefix => VarNamedTuple
                └─ x => 1

julia> rand(prefix(demo(), Val(:my_prefix)))
VarNamedTuple
└─ my_prefix => VarNamedTuple
                └─ x => 1
```
"""
prefix(model::Model, x::VarName) = contextualize(model, PrefixContext(x, model.context))
function prefix(model::Model, ::Val{sym}) where {sym}
    return contextualize(model, PrefixContext(VarName{sym}(), model.context))
end
function prefix(model::Model, x)
    return contextualize(model, PrefixContext(VarName{Symbol(x)}(), model.context))
end

"""
    (model::Model)([rng, varinfo])

Sample from the prior of the `model` with random number generator `rng`.

Returns the model's return value.

Note that calling this with an existing `varinfo` object will mutate it.
"""
(model::Model)() = model(Random.default_rng(), VarInfo())
function (model::Model)(varinfo::AbstractVarInfo)
    return model(Random.default_rng(), varinfo)
end
# ^ Weird Documenter.jl bug means that we have to write the two above separately
# as it can only detect the `function`-less syntax.
function (model::Model)(
    rng::Random.AbstractRNG, varinfo::AbstractVarInfo=OnlyAccsVarInfo(())
)
    return first(init!!(rng, model, varinfo, InitFromPrior(), UnlinkAll()))
end

"""
    init!!(
        [rng::Random.AbstractRNG,]
        model::Model,
        varinfo::AbstractVarInfo,
        init_strategy::AbstractInitStrategy,
        [transform_strategy::AbstractTransformStrategy=get_transform_strategy(varinfo),]
    )

Evaluate the `model` and replace the values of the model's random variables in the given
`varinfo` with new values, using a specified initialisation strategy. If the values in
`varinfo` are not set, they will be added using a specified initialisation strategy.

`transform_strategy` tells the model evaluation whether variables should be interpreted as
linked or unlinked. Right now, it is slightly complicated because the default behaviour
depends on the `varinfo` provided. If `varinfo isa VarInfo`, then the transform strategy is
inferred from the VarInfo, i.e., linked variables in the VarInfo are treated as linked
during evaluation. Conversely, if `varinfo isa OnlyAccsVarInfo`, then you must specify the
transform strategy explicitly, since an `OnlyAccsVarInfo` does not contain any information
about which variables are transformed.

Returns a tuple of the model's return value, plus the updated `varinfo` object.
"""
function init!!(
    rng::Random.AbstractRNG,
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy=get_transform_strategy(vi),
)
    ctx = InitContext(rng, init_strategy, transform_strategy)
    model = DynamicPPL.setleafcontext(model, ctx)
    return DynamicPPL.evaluate_nowarn!!(model, vi)
end
function init!!(
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=get_transform_strategy(vi),
)
    return init!!(Random.default_rng(), model, vi, init_strategy, transform_strategy)
end

"""
    evaluate!!(model::Model, varinfo)

Evaluate the `model` with the given `varinfo`, wrapping it in a `ThreadSafeVarInfo` if the
model is marked as needing threadsafe evaluation.

!!! warning
    The semantics of this method are complicated. We **strongly** recommend that users do
    *not* use this method unless absolutely necessary. In the future this method will be
    deprecated and removed. As far as possible (and it should **always** be possible --
    please open an issue if you do not know how to adapt your code!) you should use the
    five-argument `init!!([rng,] model, ::OnlyAccsVarInfo, init_strategy,
    transform_strategy)` method, which has more explicit semantics and allows you to have
    more control over each part of the evaluation process.

The exact semantics depend on the `model`'s context. Fundamentally, this method executes the
model evaluation function (i.e., the function used to define the model) using the given
`varinfo` as an argument. At each tilde-statement, `tilde_assume!!` or `tilde_observe!!` is
called, whose behaviour depends on the model's context.

Broadly speaking, if the leaf context is an `InitContext`, then this function:

- uses the initialisation strategy inside the `InitContext`;
- uses the transform strategy inside the `InitContext`;
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- overwrites the values in `varinfo` with the new values obtained from the initialisation strategy.

If the leaf context is a `DefaultContext`, then this function:

- uses the values inside the `varinfo` as the initialisation strategy;
- derives a transform strategy from the `varinfo`'s stored variables (if a linked variable is
  stored, then the transform strategy will treat that variable as linked; likewise for
  unlinked)
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- does not overwrite the values in the `varinfo` (that is unnecessary since the values used
  for evaluation are already stored in `varinfo`).

The long-term plan for this method is to:

- Replace `DefaultContext` with `InitContext` by splitting up the functionality of `DefaultContext`
  into its constituent components
- Remove the `VarInfo` argument, and instead use only an `AccumulatorTuple`
- Separate the initialisation and transform strategies into separate arguments, instead of storing
  them inside the model's context.
"""
function AbstractPPL.evaluate!!(model::Model, varinfo::AbstractVarInfo)
    @warn (
        "Calling `evaluate!!(model, varinfo)` directly is not recommended and will be" *
        " deprecated in the future. Please switch to using `init!!([rng,] model," *
        " ::OnlyAccsVarInfo, init_strategy, transform_strategy)` instead, which" *
        " has more explicit semantics and allows you to have more control over each" *
        " part of the evaluation process. Please see the DynamicPPL documentation" *
        " for more details: https://turinglang.org/DynamicPPL.jl/stable/evaluation"
    ) maxlog = 5
    return DynamicPPL.evaluate_nowarn!!(model, varinfo)
end

"""
    evaluate_nowarn!!(model::Model, varinfo)

This is the same as `evaluate!!(model, varinfo)` but without the deprecation warning.

!!! warning
    This is meant for internal use in DynamicPPL.jl only! If you rely on this method in your
    code, please note that it may break at any time.
"""
function evaluate_nowarn!!(model::Model, varinfo::AbstractVarInfo)
    return if requires_threadsafe(model)
        # Use of float_type_with_fallback(eltype(x)) is necessary to deal with cases where x is
        # a gradient type of some AD backend.
        # TODO(mhauru) How could we do this more cleanly? The problem case is map_accumulator!!
        # for ThreadSafeVarInfo. In that one, if the map produces e.g a ForwardDiff.Dual, but
        # the accumulators in the VarInfo are plain floats, we error since we can't change the
        # element type of ThreadSafeVarInfo.accs_by_thread. However, doing this conversion here
        # messes with cases like using Float32 of logprobs and Float64 for x. Also, this is just
        # plain ugly and hacky.
        # The below line is finicky for type stability. For instance, assigning the eltype to
        # convert to into an intermediate variable makes this unstable (constant propagation
        # fails). Take care when editing.
        param_eltype = DynamicPPL.get_param_eltype(varinfo, model.context)
        accs = map(DynamicPPL.getaccs(varinfo)) do acc
            DynamicPPL.convert_eltype(float_type_with_fallback(param_eltype), acc)
        end
        varinfo = DynamicPPL.setaccs!!(varinfo, accs)
        wrapper = ThreadSafeVarInfo(resetaccs!!(varinfo))
        result, wrapper_new = _evaluate!!(model, wrapper)
        # TODO(penelopeysm): If seems that if you pass a TSVI to this method, it
        # will return the underlying VI, which is a bit counterintuitive (because
        # calling TSVI(::TSVI) returns the original TSVI, instead of wrapping it
        # again).
        return result, setaccs!!(wrapper_new.varinfo, getaccs(wrapper_new))
    else
        _evaluate!!(model, resetaccs!!(varinfo))
    end
end

"""
    _evaluate!!(model::Model, varinfo)

Evaluate the `model` with the given `varinfo`.

This function does not wrap the varinfo in a `ThreadSafeVarInfo`. It also does not
reset the log probability of the `varinfo` before running.
"""
function _evaluate!!(model::Model, varinfo::AbstractVarInfo)
    args, kwargs = make_evaluate_args_and_kwargs(model, varinfo)
    return model.f(args...; kwargs...)
end

is_splat_symbol(s::Symbol) = startswith(string(s), "#splat#")

"""
    make_evaluate_args_and_kwargs(model, varinfo)

Return the arguments and keyword arguments to be passed to the evaluator of the model, i.e. `model.f`e.
"""
@generated function make_evaluate_args_and_kwargs(
    model::Model{_F,argnames}, varinfo::AbstractVarInfo
) where {_F,argnames}
    unwrap_args = [
        if is_splat_symbol(var)
            :(
                $convert_model_argument(
                    $get_param_eltype(varinfo, model.context), model.args.$var
                )...
            )
        else
            :($convert_model_argument(
                $get_param_eltype(varinfo, model.context), model.args.$var
            ))
        end for var in argnames
    ]
    return quote
        args = (model, varinfo, $(unwrap_args...))
        kwargs = model.defaults
        return args, kwargs
    end
end

"""
    get_param_eltype(varinfo::AbstractVarInfo, context::AbstractContext)

Get the element type of the parameters being used to evaluate a model, using a `varinfo`
under the given `context`. For example, when evaluating a model with ForwardDiff AD, this
should return `ForwardDiff.Dual`.

By default, this uses `eltype(varinfo)` which is slightly cursed. This relies on the fact
that typically, before evaluation, the parameters will have been inserted into the VarInfo's
metadata field.

For `InitContext`, it's quite different: because `InitContext` is responsible for supplying
the parameters, we can avoid using `eltype(varinfo)` and instead query the parameters inside
it. See the docstring of `get_param_eltype(strategy::AbstractInitStrategy)` for more
explanation.
"""
function get_param_eltype(vi::AbstractVarInfo, ctx::AbstractParentContext)
    return get_param_eltype(vi, DynamicPPL.childcontext(ctx))
end
get_param_eltype(vi::AbstractVarInfo, ::AbstractContext) = eltype(vi)
function get_param_eltype(::AbstractVarInfo, ctx::InitContext)
    return get_param_eltype(ctx.strategy)
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
    rand([rng=Random.default_rng()], model::Model)

Sample a `VarNamedTuple` of raw values from the prior of `model`.
"""
function Base.rand(rng::Random.AbstractRNG, model::Model)
    vi = OnlyAccsVarInfo((RawValueAccumulator(false),))
    vi = last(init!!(rng, model, vi, InitFromPrior(), UnlinkAll()))
    return get_raw_values(vi)
end
Base.rand(model::Model) = rand(Random.default_rng(), model)

"""
    logjoint(model::Model, params)
    logjoint(model::Model, varinfo::AbstractVarInfo)

Return the log joint probability of variables `params` for the probabilistic `model`, or the
log joint of the data in `varinfo` if provided.

Note that this probability always refers to the parameters in unlinked space, i.e., the
return value of `logjoint` does not depend on whether `VarInfo` has been linked or not.

See also [`logprior`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 2 methods)

julia> # Using a `NamedTuple`.
       logjoint(demo([1.0]), (m = 100.0, ))
-9902.33787706641

julia> # Using a `OrderedDict`.
       logjoint(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-9902.33787706641

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0) + logpdf(Normal(), 100.0)
-9902.33787706641
```
"""
function logjoint(model::Model, params)
    vi = OnlyAccsVarInfo(
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator())
    )
    init_strategy = InitFromParams(params, nothing)
    return getlogjoint(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function logjoint(model::Model, varinfo::AbstractVarInfo)
    return logjoint(model, get_values(varinfo))
end

"""
    logprior(model::Model, params)
    logprior(model::Model, varinfo::AbstractVarInfo)

Return the log prior probability of variables `params` for the probabilistic `model`, or the
log prior of the data in `varinfo` if provided.

Note that this probability always refers to the parameters in unlinked space, i.e., the
return value of `logprior` does not depend on whether `VarInfo` has been linked or not.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 2 methods)

julia> # Using a `NamedTuple`.
       logprior(demo([1.0]), (m = 100.0, ))
-5000.918938533205

julia> # Using a `OrderedDict`.
       logprior(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-5000.918938533205

julia> # Truth.
       logpdf(Normal(), 100.0)
-5000.918938533205
```
"""
function logprior(model::Model, params)
    vi = OnlyAccsVarInfo(AccumulatorTuple(LogPriorAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getlogprior(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function logprior(model::Model, varinfo::AbstractVarInfo)
    return logprior(model, get_values(varinfo))
end

"""
    loglikelihood(model::Model, params)
    loglikelihood(model::Model, varinfo::AbstractVarInfo)

Return the log likelihood of variables `params` for the probabilistic `model`, or the log
likelihood of the data in `varinfo` if provided.

See also [`logjoint`](@ref) and [`logprior`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 2 methods)

julia> # Using a `NamedTuple`.
       loglikelihood(demo([1.0]), (m = 100.0, ))
-4901.418938533205

julia> # Using a `OrderedDict`.
       loglikelihood(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-4901.418938533205

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0)
-4901.418938533205
"""
function Distributions.loglikelihood(model::Model, params)
    vi = OnlyAccsVarInfo(AccumulatorTuple(LogLikelihoodAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getloglikelihood(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function Distributions.loglikelihood(model::Model, varinfo::AbstractVarInfo)
    return loglikelihood(model, get_values(varinfo))
end

# Implemented & documented in DynamicPPLMCMCChainsExt
function predict end

"""
    returned(model::Model, parameters...)

Initialise a `model` using the given `parameters` and return the model's return value. The
parameters must be provided in a format that can be wrapped in an `InitFromParams`, i.e.,
`InitFromParams(parameters..., nothing)` must be a valid `AbstractInitStrategy` (where
`nothing` is the fallback strategy to use if parameters are not provided).

As far as DynamicPPL is concerned, `parameters` can be either a singular `NamedTuple` or an
`AbstractDict{<:VarName}`; however this method is left flexible to allow for other packages
that wish to extend `InitFromParams`.

# Example
```jldoctest
julia> using DynamicPPL, Distributions

julia> @model function demo()
           m ~ Normal()
           return (mp1 = m + 1,)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> returned(model, (; m = 1.0))
(mp1 = 2.0,)

julia> returned(model, Dict{VarName,Float64}(@varname(m) => 2.0))
(mp1 = 3.0,)
```
"""
function returned(model::Model, parameters...)
    # Note: we can't use `fix(model, parameters)` because
    # https://github.com/TuringLang/DynamicPPL.jl/issues/1097
    return first(
        init!!(
            model,
            DynamicPPL.OnlyAccsVarInfo(DynamicPPL.AccumulatorTuple()),
            # Use `nothing` as the fallback to ensure that any missing parameters cause an
            # error
            InitFromParams(parameters..., nothing),
            UnlinkAll(),
        ),
    )
end
