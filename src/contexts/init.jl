"""
    AbstractInitStrategy

Abstract type representing the possible ways of initialising new values for the random
variables in a model (e.g., when creating a new VarInfo).

Any subtype of `AbstractInitStrategy` must implement the [`DynamicPPL.init`](@ref) method,
and in some cases, [`DynamicPPL.get_param_eltype`](@ref) (see its docstring for details).
"""
abstract type AbstractInitStrategy end

"""
    init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, strategy::AbstractInitStrategy)

Generate a new value for a random variable with the given distribution.

This function must return an `AbstractTransformedValue`.

If `strategy` provides values that are already untransformed (e.g., a Float64 within (0, 1)
for `dist::Beta`, then you should return an `UntransformedValue`.

Otherwise, often there are cases where this will return either a `VectorValue` or a
`LinkedVectorValue`, for example, if the strategy is reading from an existing `VarInfo`.
"""
function init end

"""
    DynamicPPL.get_param_eltype(strategy::AbstractInitStrategy)

Return the element type of the parameters generated from the given initialisation strategy.

The default implementation returns `Any`. However, for `InitFromParams` which provides known
parameters for evaluating the model, methods are sometimes implemented in order to return
more specific types.

In general, if you are implementing a custom `AbstractInitStrategy`, correct behaviour can
only be guaranteed if you implement this method as well. However, quite often, the default
return value of `Any` will actually suffice. The cases where this does *not* suffice, and
where you _do_ have to manually implement `get_param_eltype`, are explained in the extended
help (see `??DynamicPPL.get_param_eltype` in the REPL).

# Extended help

There are a few edge cases in DynamicPPL where the element type is needed. These largely
relate to determining the element type of accumulators ahead of time (_before_ evaluation),
as well as promoting type parameters in model arguments. The classic case is when evaluating
a model with ForwardDiff: the log-probability accumulators must be promoted to contain
`Dual`s, and any `Vector{Float64}` arguments must be promoted to `Vector{Dual}`. Other
tracer types, for example those in SparseConnectivityTracer.jl, also require similar
treatment.

If the `AbstractInitStrategy` is never used in combination with tracer types, then it is
perfectly safe to return `Any`. This does not lead to type instability downstream because
the actual accumulators will still be created with concrete Float types (the `Any` is just
used to determine whether the float type needs to be modified).

In case that wasn't enough: in fact, even the above is not always true. Firstly, the
accumulator argument is only true when evaluating with ThreadSafeVarInfo. See the comments
in `DynamicPPL.unflatten!!` for more details. For non-threadsafe evaluation, Julia is
capable of automatically promoting the types on its own. Secondly, the promotion only
matters if you are trying to directly assign into a `Vector{Float64}` with a
`ForwardDiff.Dual` or similar tracer type, for example using `xs[i] = MyDual`. This doesn't
actually apply to tilde-statements like `xs[i] ~ ...` because those use `Accessors.set`
under the hood, which also does the promotion for you. For the gory details, see the
following issues:

- https://github.com/TuringLang/DynamicPPL.jl/issues/906 for accumulator types
- https://github.com/TuringLang/DynamicPPL.jl/issues/823 for type argument promotion
"""
get_param_eltype(::AbstractInitStrategy) = Any

"""
    InitFromPrior()

Obtain new values by sampling from the prior distribution.
"""
struct InitFromPrior <: AbstractInitStrategy end
function init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, ::InitFromPrior)
    return UntransformedValue(rand(rng, dist))
end

"""
    InitFromUniform()
    InitFromUniform(lower, upper)

Obtain new values by first transforming the distribution of the random variable to
unconstrained space, then sampling a value uniformly between `lower` and `upper`.

If `lower` and `upper` are unspecified, they default to `(-2, 2)`, which mimics Stan's
default initialisation strategy (see the [Stan reference manual page on
initialisation](https://mc-stan.org/docs/reference-manual/execution.html#initialization) for
more details).

Requires that `lower <= upper`.
"""
struct InitFromUniform{T<:AbstractFloat} <: AbstractInitStrategy
    lower::T
    upper::T
    function InitFromUniform(lower::T, upper::T) where {T<:AbstractFloat}
        lower > upper &&
            throw(ArgumentError("`lower` must be less than or equal to `upper`"))
        return new{T}(lower, upper)
    end
    InitFromUniform() = InitFromUniform(-2.0, 2.0)
end
function init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, u::InitFromUniform)
    b = Bijectors.bijector(dist)
    sz = Bijectors.output_size(b, dist)
    # TODO(penelopeysm): This is stupid, and is really just because `output_size` doesn't
    # give us **exactly** the info that we want. VectorBijectors will solve this since we
    # get `linked_vec_length(dist)` which directly tells us how many elements we need to
    # sample.
    real_sz = prod(sz)
    y = u.lower .+ ((u.upper - u.lower) .* rand(rng, real_sz))
    return LinkedVectorValue(y, dist)
end

"""
    InitFromParams(
        params::Any
        fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )

Obtain new values by extracting them from the given set of `params`.

The most common use case is to provide a `VarNamedTuple`, which provides a mapping from
variable names to values. However, we leave the type of `params` open in order to allow for
custom parameter storage types.

## Custom parameter storage types

For `InitFromParams` to work correctly with a custom `params::P`, you need to implement

```julia
DynamicPPL.init(rng, vn::VarName, dist::Distribution, p::InitFromParams{P}) where {P}
```

This tells you how to obtain values for the random variable `vn` from `p.params`. Note that
the last argument is `InitFromParams(params)`, not just `params` itself. Please see the
docstring of [`DynamicPPL.init`](@ref) for more information on the expected behaviour.

In some cases (specifically, when you expect that the type of log-probabilities will need to
be expanded: the most common example is when running AD with ForwardDiff.jl), you *may* also
need to implement:

```julia
DynamicPPL.get_param_eltype(p::InitFromParams{P}) where {P}
```

See the docstring of [`DynamicPPL.get_param_eltype`](@ref) for more information on when this
is needed.

The argument `fallback` specifies how new values are to be obtained if they cannot be found
in `params`, or they are specified as `missing`. `fallback` can either be an initialisation
strategy itself, in which case it will be used to obtain new values, or it can be `nothing`,
in which case an error will be thrown. The default for `fallback` is `InitFromPrior()`.
"""
struct InitFromParams{P,S<:Union{AbstractInitStrategy,Nothing}} <: AbstractInitStrategy
    params::P
    fallback::S
end
InitFromParams(params) = InitFromParams(params, InitFromPrior())
# For NamedTuple and Dict, we just convert to VNT internally. This saves us from having to
# implement separate `init()` methods for those. It also means that when someone provides
# a Dict with @varname(x[1]) => 1.0 we will issue a warning for untemplated VNT.
function InitFromParams(params::NamedTuple, fallback::Union{AbstractInitStrategy,Nothing})
    return InitFromParams(VarNamedTuple(params), fallback)
end
function InitFromParams(
    params::AbstractDict{<:VarName}, fallback::Union{AbstractInitStrategy,Nothing}
)
    return InitFromParams(VarNamedTuple(pairs(params)), fallback)
end

function init(::Random.AbstractRNG, vn::VarName, ::Distribution, ::Nothing)
    error("No value was provided for the variable `$(vn)`.")
end

function init(
    rng::Random.AbstractRNG,
    vn::VarName,
    dist::Distribution,
    p::InitFromParams{<:VarNamedTuple},
)
    if hasvalue(p.params, vn, dist)
        x = getvalue(p.params, vn, dist)
        x !== missing && return init_transform(x, dist)
    end
    return init(rng, vn, dist, p.fallback)
end

init_transform(x, ::Distribution) = UntransformedValue(x)
function init_transform(x::XT, dist::Distribution) where {XT<:AbstractTransformedValue}
    return set_internal_transform(x, dist)
end

"""
Like InitFromParams, but it is always assumed that the VNT contains _exactly_ the
correct set of variables, and that indexing into them will always return _exactly_
the values for those variables.

The main difference is that InitFromParams will call hasvalue(p.params, vn, dist)
rather than just hasvalue(p.params, vn), which can be substantially slower.

TODO(penelopeysm): Get rid of MCMCChains and never call the three-value argument again.
Seriously. It's just nuts that I have to do these workarounds because of a package that
isn't even DynamicPPL.
"""
struct InitFromParamsUnsafe{P<:VarNamedTuple} <: AbstractInitStrategy
    params::P
end
function init(
    ::Random.AbstractRNG,
    vn::VarName,
    dist::Distribution,
    p::InitFromParamsUnsafe{<:VarNamedTuple},
)
    return if haskey(p.params, vn)
        x = p.params[vn]
        return init_transform(x, dist)
    else
        error("No value was provided for the variable `$(vn)`.")
    end
end

function DynamicPPL.get_param_eltype(p::InitFromParamsUnsafe)
    # TODO(penelopeysm): Ugly hack. Currently this is not used anywhere except in Turing's
    # ADTypeCheckContext tests. However, when we stop using DefaultContext and start using
    # this as its replacement, we will need this function so that we can promote the
    # accumulators' eltype accordingly (unless we find a better solution than eltypes).
    # 
    # Note that pair.second returns internal values.
    vals = mapfoldl(
        pair -> tovec(DynamicPPL.get_internal_value(pair.second)),
        vcat,
        p.params;
        init=Union{}[],
    )
    return eltype(vals)
end

"""
    RangeAndLinked

Suppose we have vectorised parameters `params::AbstractVector{<:Real}`. Each random variable
in the model will in general correspond to a sub-vector of `params`. This struct stores
information about that range, as well as whether the sub-vector represents a linked value or
an unlinked value.

$(TYPEDFIELDS)
"""
struct RangeAndLinked
    # indices that the variable corresponds to in the vectorised parameter
    range::UnitRange{Int}
    # whether the variable is linked or unlinked
    is_linked::Bool
end

"""
    InitFromVector(
        vect::AbstractVector{<:Real},
        varname_ranges::VarNamedTuple,
        transform_strategy::AbstractTransformStrategy
    ) <: AbstractInitStrategy

!!! warning
    This constructor is only meant for internal use. Please use `InitFromVector(vect,
    ldf::LogDensityFunction)` instead, which will automatically construct the
    `varname_ranges` and `transform_strategy` arguments for you.

A struct that wraps a vector of parameter values, plus information about how random
variables map to ranges in that vector.

The `transform_strategy` argument in fact duplicates information stored inside `varname_ranges`.
For example, if every `RangeAndLinked` in `varname_ranges` has `is_linked == true`, then
`transform_strategy` will be `LinkAll()`.

However, storing `transform_strategy` here is a way to communicate at the type level whether all
variables are linked or unlinked, which provides much better performance in the case where
all variables are linked or unlinked, due to improved type stability.
"""
struct InitFromVector{
    T<:AbstractVector{<:Real},V<:VarNamedTuple,L<:AbstractTransformStrategy
} <: AbstractInitStrategy
    # The full parameter vector which we index into to get variable values
    vect::T
    # Ranges for all VarNames
    varname_ranges::V
    # Transform strategy. The main reason why this is stored is to allow for greater type
    # stability: in the case where `transform_strategy` is `LinkAll()` or `UnlinkAll()`, we can
    # statically know that the transforms to be used are always linked or unlinked.
    transform_strategy::L
end

"""
    maybe_view_ad(vect::AbstractArray, range)

For the most part, this function is just `view(vect, range)`. The problem is that for
ReverseDiff this tends to be very slow
(https://github.com/JuliaDiff/ReverseDiff.jl/issues/281), so in the ReverseDiffExt we
overload this for ReverseDiff's tracked arrays to just do `getindex` instead, which is much
faster.
"""
@inline maybe_view_ad(vect::AbstractArray, range) = view(vect, range)

function _get_range_and_linked(ifv::InitFromVector, vn::VarName)
    # The type assertion does nothing if `varname_ranges` has concrete element types, as is
    # the case for all type stable models. However, if the model is not type stable,
    # vr.varname_ranges[vn] may infer to have type `Any`. In this case it is helpful to
    # assert that it is a RangeAndLinked, because even though it remains non-concrete, it'll
    # allow the compiler to infer the types of `range` and `is_linked`.
    return ifv.varname_ranges[vn]::RangeAndLinked
end

function transform_vector(
    vect::AbstractVector, dist::Distribution, ::RangeAndLinked, ::UnlinkAll
)
    return VectorValue(vect, dist)
end
function transform_vector(
    vect::AbstractVector, dist::Distribution, ::RangeAndLinked, ::LinkAll
)
    return LinkedVectorValue(vect, dist)
end
function transform_vector(
    vect::AbstractVector,
    dist::Distribution,
    ral::RangeAndLinked,
    ::AbstractTransformStrategy
)
    return ral.is_linked ? LinkedVectorValue(vect, dist) : VectorValue(vect, dist)
end

function init(::Random.AbstractRNG, vn::VarName, dist::Distribution, ifv::InitFromVector)
    range_and_linked = _get_range_and_linked(ifv, vn)
    vect = maybe_view_ad(ifv.vect, range_and_linked.range)
    # This block here is why we store transform_strategy inside the InitFromVector, as it
    # allows for type stability.
    return transform_vector(vect, dist, range_and_linked, ifv.transform_strategy)
end
function get_param_eltype(strategy::InitFromVector)
    return eltype(strategy.vect)
end

"""
    InitContext(
        [rng::Random.AbstractRNG=Random.default_rng()],
        strategy::AbstractInitStrategy,
        transform_strategy::AbstractTransformStrategy,
    )

A leaf context that indicates that new values for random variables are currently being
obtained through sampling. Used e.g. when initialising a fresh VarInfo.

The `strategy` argument specifies how new values are to be obtained (see
[`AbstractInitStrategy`](@ref) for details), while the `transform_strategy` argument specifies
whether values should be treated as being in linked or unlinked space. That also means that
`transform_strategy` determines whether the log-Jacobian of the link transform is included when
evaluating the model.

!!! note
    If `leafcontext(model.context) isa InitContext`, then `evaluate!!(model, varinfo)` will
    override all values in the VarInfo.
"""
struct InitContext{
    R<:Random.AbstractRNG,S<:AbstractInitStrategy,L<:AbstractTransformStrategy
} <: AbstractContext
    rng::R
    strategy::S
    transform_strategy::L

    function InitContext(
        rng::Random.AbstractRNG,
        strategy::AbstractInitStrategy,
        transform_strategy::AbstractTransformStrategy,
    )
        return new{typeof(rng),typeof(strategy),typeof(transform_strategy)}(
            rng, strategy, transform_strategy
        )
    end
    function InitContext(
        strategy::AbstractInitStrategy, transform_strategy::AbstractTransformStrategy
    )
        return InitContext(Random.default_rng(), strategy, transform_strategy)
    end
end

function tilde_assume!!(
    ctx::InitContext, dist::Distribution, vn::VarName, template::Any, vi::AbstractVarInfo
)
    init_tval = init(ctx.rng, vn, dist, ctx.strategy)
    x, tval, logjac = apply_transform_strategy(ctx.transform_strategy, init_tval, vn, dist)
    vi = setindex_with_dist!!(vi, tval, dist, vn, template)
    vi = accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
    # We always return the untransformed value here, as that will determine
    # what the lhs of the tilde-statement is set to.
    return x, vi
end

function tilde_observe!!(
    ::InitContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template::Any,
    vi::AbstractVarInfo,
)
    return tilde_observe!!(DefaultContext(), right, left, vn, template, vi)
end
