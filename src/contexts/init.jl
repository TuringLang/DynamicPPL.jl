"""
    AbstractInitStrategy

Abstract type representing the possible ways of initialising new values for
the random variables in a model (e.g., when creating a new VarInfo).

Any subtype of `AbstractInitStrategy` must implement the
[`DynamicPPL.init`](@ref) method.
"""
abstract type AbstractInitStrategy end

"""
    init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, strategy::AbstractInitStrategy)

Generate a new value for a random variable with the given distribution.

This function must return a tuple of:

- the generated value
- a function that transforms the generated value back to the unlinked space. If the value is
  already in unlinked space, then this should be `identity`.
"""
function init end

"""
    InitFromPrior()

Obtain new values by sampling from the prior distribution.
"""
struct InitFromPrior <: AbstractInitStrategy end
function init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, ::InitFromPrior)
    return rand(rng, dist), _typed_identity
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
    return x, _typed_identity
end

"""
    InitFromParams(
        params::Union{AbstractDict{<:VarName},NamedTuple},
        fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )

Obtain new values by extracting them from the given dictionary or NamedTuple.

The parameter `fallback` specifies how new values are to be obtained if they
cannot be found in `params`, or they are specified as `missing`. `fallback`
can either be an initialisation strategy itself, in which case it will be
used to obtain new values, or it can be `nothing`, in which case an error
will be thrown. The default for `fallback` is `InitFromPrior()`.

!!! note
    The values in `params` must be provided in the space of the untransformed
    distribution.
"""
struct InitFromParams{P,S<:Union{AbstractInitStrategy,Nothing}} <: AbstractInitStrategy
    params::P
    fallback::S

    function InitFromParams(
        params::P, fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    ) where {P}
        return new{P,typeof(fallback)}(params, fallback)
    end
end

function init(
    rng::Random.AbstractRNG,
    vn::VarName,
    dist::Distribution,
    p::InitFromParams{<:Union{AbstractDict{<:VarName},NamedTuple}},
)
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
            x, _typed_identity
        end
    else
        p.fallback === nothing && error("No value was provided for the variable `$(vn)`.")
        init(rng, vn, dist, p.fallback)
    end
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
    VectorWithRanges(
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
struct VectorWithRanges{N<:NamedTuple,T<:AbstractVector{<:Real}}
    # This NamedTuple stores the ranges for identity VarNames
    iden_varname_ranges::N
    # This Dict stores the ranges for all other VarNames
    varname_ranges::Dict{VarName,RangeAndLinked}
    # The full parameter vector which we index into to get variable values
    vect::T
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
    p::InitFromParams{<:VectorWithRanges},
)
    vr = p.params
    range_and_linked = _get_range_and_linked(vr, vn)
    transform = if range_and_linked.is_linked
        from_linked_vec_transform(dist)
    else
        from_vec_transform(dist)
    end
    return (@view vr.vect[range_and_linked.range]), transform
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
        # but it's not clear right now how to do this. In my opinion, the most productive
        # way forward would be to clean up the behaviour of bijectors so that we can have a
        # clean separation between the linking and vectorisation parts of it. That way, `x`
        # can either be unlinked, unlinked vectorised, linked, or linked vectorised, and
        # regardless of which it is, we should only need to apply at most one linking and
        # one vectorisation transform.
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
