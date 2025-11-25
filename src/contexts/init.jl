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

This function must return a tuple `(x, trf)`, where

- `x` is the generated value
- `trf` is a function that transforms the generated value back to the unlinked space. If the
  value is already in unlinked space, then this should be `DynamicPPL.typed_identity`. You
  can also use `Base.identity`, but if you use this, you **must** be confident that
  `zero(eltype(x))` will **never** error. See the docstring of `typed_identity` for more
  information.
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
in `DynamicPPL.unflatten` for more details. For non-threadsafe evaluation, Julia is capable
of automatically promoting the types on its own. Secondly, the promotion only matters if you
are trying to directly assign into a `Vector{Float64}` with a `ForwardDiff.Dual` or similar
tracer type, for example using `xs[i] = MyDual`. This doesn't actually apply to
tilde-statements like `xs[i] ~ ...` because those use `Accessors.@set` under the hood, which
also does the promotion for you. For the gory details, see the following issues:

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
    return rand(rng, dist), typed_identity
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
    return x, typed_identity
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
) where {P<:Union{AbstractDict{<:VarName},NamedTuple}}
    # TODO(penelopeysm): It would be nice to do a check to make sure that all
    # of the parameters in `p.params` were actually used, and either warn or
    # error if they aren't. This is actually quite non-trivial though because
    # the structure of Dicts in particular can have arbitrary nesting.
    return if hasvalue(p.params, vn, dist)
        x = getvalue(p.params, vn, dist)
        if x === missing
            p.fallback === nothing &&
                error("A `missing` value was provided for the variable `$(vn)`.")
            init(rng, vn, dist, p.fallback)
        else
            # TODO(penelopeysm): Since x is user-supplied, maybe we could also
            # check here that the type / size of x matches the dist?
            x, typed_identity
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
    # whether it's linked
    is_linked::Bool
end

"""
    VectorWithRanges{Tlink}(
        iden_varname_ranges::NamedTuple,
        varname_ranges::Dict{VarName,RangeAndLinked},
        vect::AbstractVector{<:Real},
    )

A struct that wraps a vector of parameter values, plus information about how random
variables map to ranges in that vector.

In the simplest case, this could be accomplished only with a single dictionary mapping
VarNames to ranges and link status. However, for performance reasons, we separate out
VarNames with identity optics into a NamedTuple (`iden_varname_ranges`). All
non-identity-optic VarNames are stored in the `varname_ranges` Dict.

It would be nice to improve the NamedTuple and Dict approach. See, e.g.
https://github.com/TuringLang/DynamicPPL.jl/issues/1116.
"""
struct VectorWithRanges{Tlink,N<:NamedTuple,T<:AbstractVector{<:Real}}
    # This NamedTuple stores the ranges for identity VarNames
    iden_varname_ranges::N
    # This Dict stores the ranges for all other VarNames
    varname_ranges::Dict{VarName,RangeAndLinked}
    # The full parameter vector which we index into to get variable values
    vect::T

    function VectorWithRanges{Tlink}(
        iden_varname_ranges::N, varname_ranges::Dict{VarName,RangeAndLinked}, vect::T
    ) where {Tlink,N,T}
        return new{Tlink,N,T}(iden_varname_ranges, varname_ranges, vect)
    end
end

function _get_range_and_linked(
    vr::VectorWithRanges, ::VarName{sym,typeof(identity)}
) where {sym}
    return vr.iden_varname_ranges[sym]
end
function _get_range_and_linked(vr::VectorWithRanges, vn::VarName)
    return vr.varname_ranges[vn]
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
    transform = if linked
        from_linked_vec_transform(dist)
    else
        from_vec_transform(dist)
    end
    return (@view vr.vect[range_and_linked.range]), transform
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
    ctx::InitContext, dist::Distribution, vn::VarName, vi::AbstractVarInfo
)
    in_varinfo = haskey(vi, vn)
    val, transform = init(ctx.rng, vn, dist, ctx.strategy)
    x, inv_logjac = with_logabsdet_jacobian(transform, val)
    # Determine whether to insert a transformed value into the VarInfo.
    # If the VarInfo alrady had a value for this variable, we will
    # keep the same linked status as in the original VarInfo. If not, we
    # check the rest of the VarInfo to see if other variables are linked.
    # is_transformed(vi) returns true if vi is nonempty and all variables in vi
    # are linked.
    insert_transformed_value = in_varinfo ? is_transformed(vi, vn) : is_transformed(vi)
    val_to_insert, logjac = if insert_transformed_value
        # Calculate the forward logjac and sum them up.
        y, fwd_logjac = with_logabsdet_jacobian(link_transform(dist), x)
        # Note that if we use VectorWithRanges with a full VarInfo, this double-Jacobian
        # calculation wastes a lot of time going from linked vectorised -> unlinked ->
        # linked, and `inv_logjac` will also just be the negative of `fwd_logjac`.
        #
        # However, `VectorWithRanges` is only really used with `OnlyAccsVarInfo`, in which
        # case this branch is never hit (since `in_varinfo` will always be false). It does
        # mean that the combination of InitFromParams{<:VectorWithRanges} with a full,
        # linked, VarInfo will be very slow. That should never really be used, though. So
        # (at least for now) we can leave this branch in for full generality with other
        # combinations of init strategies / VarInfo.
        #
        # TODO(penelopeysm): Figure out one day how to refactor this. The crux of the issue
        # is that the transform used by `VectorWithRanges` is `from_linked_VEC_transform`,
        # which is NOT the same as `inverse(link_transform)` (because there is an additional
        # vectorisation step). We need `init` and `tilde_assume!!` to share this information
        # but it's not clear right now how to do this. In my opinion, there are a couple of
        # potential ways forward:
        #
        # 1. Just remove metadata entirely so that there is never any need to construct
        # a linked vectorised value again. This would require us to use VAIMAcc as the only
        # way of getting values. I consider this the best option, but it might take a long
        # time.
        #
        # 2. Clean up the behaviour of bijectors so that we can have a complete separation
        # between the linking and vectorisation parts of it. That way, `x` can either be
        # unlinked, unlinked vectorised, linked, or linked vectorised, and regardless of
        # which it is, we should only need to apply at most one linking and one
        # vectorisation transform. Doing so would allow us to remove the first call to
        # `with_logabsdet_jacobian`, and instead compose and/or uncompose the
        # transformations before calling `with_logabsdet_jacobian` once.
        y, -inv_logjac + fwd_logjac
    else
        x, -inv_logjac
    end
    # Add the new value to the VarInfo. `push!!` errors if the value already
    # exists, hence the need for setindex!!.
    if in_varinfo
        vi = setindex!!(vi, val_to_insert, vn)
    else
        vi = push!!(vi, vn, val_to_insert, dist)
    end
    # Neither of these set the `trans` flag so we have to do it manually if
    # necessary.
    if insert_transformed_value
        vi = set_transformed!!(vi, true, vn)
    end
    # `accumulate_assume!!` wants untransformed values as the second argument.
    vi = accumulate_assume!!(vi, x, logjac, vn, dist)
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
