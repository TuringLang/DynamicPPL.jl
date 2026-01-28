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
parameters for evaluating the model, methods are implemented in order to return more specific
types.

In general, if you are implementing a custom `AbstractInitStrategy`, correct behaviour can
only be guaranteed if you implement this method as well. However, quite often, the default
return value of `Any` will actually suffice. The cases where this does *not* suffice, and
where you _do_ have to manually implement `get_param_eltype`, are explained in the extended
help (see `??DynamicPPL.get_param_eltype` in the REPL).

# Extended help

There are a few edge cases in DynamicPPL where the element type is needed. These largely
relate to determining the element type of accumulators ahead of time (_before_ evaluation),
as well as promoting type parameters in model arguments. The classic case is when evaluating
a model with ForwardDiff: the accumulators must be set to `Dual`s, and any `Vector{Float64}`
arguments must be promoted to `Vector{Dual}`. Other tracer types, for example those in
SparseConnectivityTracer.jl, also require similar treatment.

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

Obtain new values by first transforming the distribution of the random variable
to unconstrained space, then sampling a value uniformly between `lower` and
`upper`, and transforming that value back to the original space.

If `lower` and `upper` are unspecified, they default to `(-2, 2)`, which mimics
Stan's default initialisation strategy.

Requires that `lower <= upper`.

# References

[Stan reference manual page on initialization](https://mc-stan.org/docs/reference-manual/execution.html#initialization)
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
    y = u.lower .+ ((u.upper - u.lower) .* rand(rng, sz...))
    b_inv = Bijectors.inverse(b)
    x = b_inv(y)
    # 0-dim arrays: https://github.com/TuringLang/Bijectors.jl/issues/398
    if x isa Array{<:Any,0}
        x = x[]
    end
    # NOTE: We don't return `LinkedVectorValue(y, ...)` here because we don't want the
    # logjac of this transform to be included when evaluating the model! The fact that
    # b_inv(y) has a non-trivial logjacobian is just an artefact of how the sampling is done
    # and has nothing to do with the model.
    return UntransformedValue(x)
end

"""
    InitFromParams(
        params::Any
        fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )

Obtain new values by extracting them from the given set of `params`.

The most common use case is to provide a `NamedTuple` or `AbstractDict{<:VarName}`, which
provides a mapping from variable names to values. However, we leave the type of `params`
open in order to allow for custom parameter storage types.

## Custom parameter storage types

For `InitFromParams` to work correctly with a custom `params::P`, you need to implement

```julia
DynamicPPL.init(rng, vn::VarName, dist::Distribution, p::InitFromParams{P}) where {P}
```

This tells you how to obtain values for the random variable `vn` from `p.params`. Note that
the last argument is `InitFromParams(params)`, not just `params` itself. Please see the
docstring of [`DynamicPPL.init`](@ref) for more information on the expected behaviour.

If you only use `InitFromParams` with `DynamicPPL.OnlyAccsVarInfo`, as is usually the case,
then you will not need to implement anything else. So far, this is the same as you would do
for creating any new `AbstractInitStrategy` subtype.

However, to use `InitFromParams` with a full `DynamicPPL.VarInfo`, you *may* also need to
implement

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

function init(
    rng::Random.AbstractRNG, vn::VarName, dist::Distribution, p::InitFromParams{P}
) where {P<:Union{AbstractDict{<:VarName},NamedTuple,VarNamedTuple}}
    return if hasvalue(p.params, vn, dist)
        x = getvalue(p.params, vn, dist)
        if x === missing
            p.fallback === nothing &&
                error("A `missing` value was provided for the variable `$(vn)`.")
            init(rng, vn, dist, p.fallback)
        elseif x isa VectorValue
            # In this case, we can't trust the transform stored in x because the _value_
            # in x may have been changed via unflatten!! without the transform being
            # updated. Therefore, we always recompute the transform here.
            VectorValue(x.val, from_vec_transform(dist), x.size)
        elseif x isa LinkedVectorValue
            # Same as above.
            LinkedVectorValue(x.val, from_linked_vec_transform(dist), x.size)
        elseif x isa UntransformedValue
            x
        else
            UntransformedValue(x)
        end
    else
        p.fallback === nothing && error("No value was provided for the variable `$(vn)`.")
        init(rng, vn, dist, p.fallback)
    end
end
function get_param_eltype(
    strategy::InitFromParams{<:Union{AbstractDict{<:VarName},NamedTuple}}
)
    return infer_nested_eltype(typeof(strategy.params))
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
        if x isa VectorValue
            # In this case, we can't trust the transform stored in x because the _value_
            # in x may have been changed via unflatten!! without the transform being
            # updated. Therefore, we always recompute the transform here.
            VectorValue(x.val, from_vec_transform(dist), x.size)
        elseif x isa LinkedVectorValue
            # Same as above.
            LinkedVectorValue(x.val, from_linked_vec_transform(dist), x.size)
        elseif x isa UntransformedValue
            x
        else
            UntransformedValue(x)
        end
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
struct RangeAndLinked{T<:Tuple}
    # indices that the variable corresponds to in the vectorised parameter
    range::UnitRange{Int}
    # whether it's linked
    is_linked::Bool
    # original size of the variable before vectorisation
    original_size::T
end

VarNamedTuples.vnt_size(ral::RangeAndLinked) = ral.original_size

"""
    VectorWithRanges{Tlink}(
        varname_ranges::VarNamedTuple,
        vect::AbstractVector{<:Real},
    )

A struct that wraps a vector of parameter values, plus information about how random
variables map to ranges in that vector.

The type parameter `Tlink` can be either `true` or `false`, to mark that the variables in
this `VectorWithRanges` are linked/not linked, or `nothing` if either the linking status is
not known or is mixed, i.e. some are linked while others are not. Using `nothing` does not
affect functionality or correctness, but causes more work to be done at runtime, with
possible impacts on type stability and performance.
"""
struct VectorWithRanges{Tlink,VNT<:VarNamedTuple,T<:AbstractVector{<:Real}}
    # Ranges for all VarNames
    varname_ranges::VNT
    # The full parameter vector which we index into to get variable values
    vect::T

    function VectorWithRanges{Tlink}(varname_ranges::VNT, vect::T) where {Tlink,VNT,T}
        if !(Tlink isa Union{Bool,Nothing})
            throw(
                ArgumentError(
                    "VectorWithRanges type parameter has to be one of `true`, `false`, or `nothing`.",
                ),
            )
        end
        return new{Tlink,VNT,T}(varname_ranges, vect)
    end
end

function _get_range_and_linked(vr::VectorWithRanges, vn::VarName)
    # The type assertion does nothing if VectorWithRanges has concrete element types, as is
    # the case for all type stable models. However, if the model is not type stable,
    # vr.varname_ranges[vn] may infer to have type `Any`. In this case it is helpful to
    # assert that it is a RangeAndLinked, because even though it remains non-concrete,
    # it'll allow the compiler to infer the types of `range` and `is_linked`.
    return vr.varname_ranges[vn]::RangeAndLinked
end
function init(
    ::Random.AbstractRNG,
    vn::VarName,
    dist::Distribution,
    p::InitFromParams{<:VectorWithRanges{T}},
) where {T}
    vr = p.params
    range_and_linked = _get_range_and_linked(vr, vn)
    # T can either be `nothing` (i.e., link status is mixed, in which
    # case we use the stored link status), or `true` / `false`, which
    # indicates that all variables are linked / unlinked.
    linked = isnothing(T) ? range_and_linked.is_linked : T
    return if linked
        LinkedVectorValue(
            view(vr.vect, range_and_linked.range),
            from_linked_vec_transform(dist),
            range_and_linked.original_size,
        )
    else
        VectorValue(
            view(vr.vect, range_and_linked.range),
            from_vec_transform(dist),
            range_and_linked.original_size,
        )
    end
end
function get_param_eltype(strategy::InitFromParams{<:VectorWithRanges})
    return eltype(strategy.params.vect)
end

"""
    InitContext(
            [rng::Random.AbstractRNG=Random.default_rng()],
            [strategy::AbstractInitStrategy=InitFromPrior()],
    )

A leaf context that indicates that new values for random variables are
currently being obtained through sampling. Used e.g. when initialising a fresh
VarInfo. Note that, if `leafcontext(model.context) isa InitContext`, then
`evaluate!!(model, varinfo)` will override all values in the VarInfo.
"""
struct InitContext{R<:Random.AbstractRNG,S<:AbstractInitStrategy} <: AbstractContext
    rng::R
    strategy::S
    function InitContext(
        rng::Random.AbstractRNG, strategy::AbstractInitStrategy=InitFromPrior()
    )
        return new{typeof(rng),typeof(strategy)}(rng, strategy)
    end
    function InitContext(strategy::AbstractInitStrategy=InitFromPrior())
        return InitContext(Random.default_rng(), strategy)
    end
end

function tilde_assume!!(
    ctx::InitContext, dist::Distribution, vn::VarName, template::Any, vi::AbstractVarInfo
)
    tval = init(ctx.rng, vn, dist, ctx.strategy)
    x, init_logjac = with_logabsdet_jacobian(get_transform(tval), get_internal_value(tval))
    # TODO(penelopeysm): This could be inefficient if `tval` is already linked and
    # `setindex_with_dist!!` tells it to create a new linked value again. In particular,
    # this is inefficient if we use `InitFromParams` that provides linked values. The answer
    # to this is to stop using setindex_with_dist!! and just use the TransformedValue
    # accumulator.
    vi, logjac, _ = setindex_with_dist!!(vi, x, dist, vn, template)
    # `accumulate_assume!!` wants untransformed values as the second argument.
    vi = accumulate_assume!!(vi, x, tval, init_logjac + logjac, vn, dist, template)
    # We always return the untransformed value here, as that will determine
    # what the lhs of the tilde-statement is set to.
    return x, vi
end

function tilde_observe!!(
    ::InitContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    return tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
