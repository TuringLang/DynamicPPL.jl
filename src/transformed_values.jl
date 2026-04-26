# TODO(mhauru) The policy of vectorising all values was set when the old VarInfo type was
# using a Vector as the internal storage in all cases. We should revisit this, and allow
# values to be stored "raw", since VarNamedTuple supports it.
#
# NOTE(penelopeysm): The main problem with unvectorising values is that when calling
# `unflatten!!`, it is not clear how many elements to take from the vector. In general
# we would need to know the distribution to get this data, which is fine if we are 
# executing the model, but `unflatten!!` does not have that information. As long as we
# depend on the behaviour of `unflatten!!` somewhere, we cannot get rid of it.

"""
    abstract type AbstractTransform end

An abstract type to represent the intended transformation for a variable.
"""
abstract type AbstractTransform end

"""
    DynamicLink <: AbstractTransform

A type indicating that a target transformation should be derived by recomputing
`Bijectors.VectorBijectors.from_linked_vec(dist)`, where `dist` is the distribution on the
right-hand side of the tilde.
"""
struct DynamicLink <: AbstractTransform end

"""
    Unlink <: AbstractTransform

A type indicating that a target transformation should be derived by recomputing
`Bijectors.VectorBijectors.from_vec(dist)`, where `dist` is the distribution on the
right-hand side of the tilde.
"""
struct Unlink <: AbstractTransform end

"""
    NoTransform <: AbstractTransform

A type indicating that the value is not transformed.
"""
struct NoTransform <: AbstractTransform end

"""
    FixedTransform{F} <: AbstractTransform

A type to represent a fixed (static) transformation of type `F`.
"""
struct FixedTransform{F} <: AbstractTransform
    transform::F
end
Base.:(==)(ft1::FixedTransform, ft2::FixedTransform) = ft1.transform == ft2.transform
function Base.isequal(ft1::FixedTransform, ft2::FixedTransform)
    return isequal(ft1.transform, ft2.transform)
end

"""
    TransformedValue{V,T<:AbstractTransform}

A struct to represent a value that has undergone some transformation.

The *transformed* value is stored in the `value` field, and the *inverse* transformation is
stored in the `transform` field. 

That means that `get_transform(tv)(get_internal_value(tv))` should return the raw,
untransformed, value associated with `tv`.
"""
struct TransformedValue{V,T<:AbstractTransform}
    value::V
    transform::T
end
function Base.:(==)(tv1::TransformedValue, tv2::TransformedValue)
    return (get_internal_value(tv1) == get_internal_value(tv2)) &
           (get_transform(tv1) == get_transform(tv2))
end
function Base.isequal(tv1::TransformedValue, tv2::TransformedValue)
    return isequal(get_internal_value(tv1), get_internal_value(tv2)) &&
           isequal(get_transform(tv1), get_transform(tv2))
end

"""
    DynamicPPL.get_transform(tv::TransformedValue)

Get the subtype of `AbstractTransform` that is stored inside `tv`. Note that this is not
always a function that can be used to obtain the raw, untransformed value. If you need the
raw value, please use [`DynamicPPL.get_raw_value`](@ref).
"""
get_transform(tv::TransformedValue) = tv.transform

"""
    DynamicPPL.get_internal_value(tv::TransformedValue)

Get the internal value stored in `tv`.
"""
get_internal_value(tv::TransformedValue) = tv.value

"""
    DynamicPPL.set_internal_value(tv::TransformedValue, new_val)

Create a new `TransformedValue` with the same transformation as `tv`, but with
internal value `new_val`.
"""
function set_internal_value(tv::TransformedValue, new_val)
    TransformedValue(new_val, get_transform(tv))
end

"""
    DynamicPPL.get_raw_value(tv::TransformedValue)
    DynamicPPL.get_raw_value(tv::TransformedValue, dist::Distribution)

Get the raw (untransformed) value from a `TransformedValue`.

The two-argument version, with a `dist::Distribution` argument, is required when the
`TransformedValue` holds a *dynamic* transform (i.e., `tv.transform` is either
[`DynamicLink`](@ref) or [`Unlink`](@ref).

For [`FixedTransform`](@ref) or [`NoTransform`](@ref), the `dist` argument is not needed
(and if supplied, will be ignored).
"""
get_raw_value(tv::TransformedValue{<:Any,NoTransform}) = get_internal_value(tv)
get_raw_value(tv::TransformedValue{<:Any,NoTransform}, ::Distribution) = get_raw_value(tv)
function get_raw_value(tv::TransformedValue{<:Any,<:FixedTransform})
    return tv.transform.transform(get_internal_value(tv))
end
function get_raw_value(tv::TransformedValue{<:Any,<:FixedTransform}, ::Distribution)
    return get_raw_value(tv)
end
function get_raw_value(::TransformedValue{<:Any,<:Union{DynamicLink,Unlink}})
    return throw(
        ArgumentError(
            "dynamic transforms including `DynamicLink` and `Unlink` must be calculated" *
            " from the distribution of the variable: please use `get_raw_value(tv, dist)`" *
            " instead, or alternatively fix the transforms if you know that they are" *
            " constant.",
        ),
    )
end
function get_raw_value(
    tv::TransformedValue{<:AbstractVector{<:Real},DynamicLink}, dist::Distribution
)
    finvlink = Bijectors.VectorBijectors.from_linked_vec(dist)
    return finvlink(get_internal_value(tv))
end
function get_raw_value(
    tv::TransformedValue{<:AbstractVector{<:Real},Unlink}, dist::Distribution
)
    invlink = Bijectors.VectorBijectors.from_vec(dist)
    return invlink(get_internal_value(tv))
end

"""
    abstract type AbstractTransformStrategy end

An abstract type for strategies that determine how each variable should be transformed.

For subtypes of `AbstractTransformStrategy`, the only method that needs to be overloaded is
[`DynamicPPL.target_transform(::AbstractTransformStrategy, vn::VarName)`](@ref
DynamicPPL.target_transform), which returns an [`AbstractTransform`](@ref) that
specifies how the variable with name `vn` should be transformed.

The transform strategy dictates how the log-Jacobian is accumulated during model evaluation.
Regardless of what initialisation strategy is used (and what kind of transformed value
`init()` returns, the log-Jacobian that is accumulated is always the log-Jacobian for the
forward transform specified by `target_transform(strategy, vn)`.

That is, even if `init()` returns an unlinked or untransformed value, if the transform
strategy is `LinkAll()` (which returns `DynamicLink` for all variables), then the
log-Jacobian for linking will be accumulated during model evaluation.

Subtypes in DynamicPPL are [`LinkAll`](@ref), [`UnlinkAll`](@ref), [`LinkSome`](@ref), and
[`UnlinkSome`](@ref).
"""
abstract type AbstractTransformStrategy end

"""
    target_transform(linker::AbstractTransformStrategy, vn::VarName)::AbstractTransform

Determine whether a variable with name `vn` should be linked according to the `linker`
strategy. Returns a subtype of `AbstractTransform` that indicates the intended
transformation for the variable.
"""
function target_transform end

"""
    LinkAll() <: AbstractTransformStrategy

Indicate that all variables should be linked.
"""
struct LinkAll <: AbstractTransformStrategy end
target_transform(::LinkAll, ::VarName) = DynamicLink()

"""
    UnlinkAll() <: AbstractTransformStrategy

Indicate that all variables should be unlinked.
"""
struct UnlinkAll <: AbstractTransformStrategy end
target_transform(::UnlinkAll, ::VarName) = Unlink()

"""
    WithTransforms(transforms::VarNamedTuple, fallback) <: AbstractTransformStrategy

Indicate that the variables in `transforms` should be transformed according to their
corresponding values in `transforms`, which should be subtypes of `AbstractTransform`. The
link statuses of other variables are determined by the `fallback` strategy.
"""
struct WithTransforms{V<:VarNamedTuple,L<:AbstractTransformStrategy} <:
       AbstractTransformStrategy
    transforms::V
    fallback::L
    function WithTransforms(transforms::VarNamedTuple, fallback::AbstractTransformStrategy)
        # Check that all values in transforms are subtypes of AbstractTransform
        if !all(x -> x isa AbstractTransform, values(transforms))
            throw(
                ArgumentError(
                    "All values in `transforms` must be subtypes of `AbstractTransform`."
                ),
            )
        end
        return new{typeof(transforms),typeof(fallback)}(transforms, fallback)
    end
end
function Base.:(==)(wt1::WithTransforms, wt2::WithTransforms)
    return (wt1.transforms == wt2.transforms) & (wt1.fallback == wt2.fallback)
end
function Base.isequal(wt1::WithTransforms, wt2::WithTransforms)
    return isequal(wt1.transforms, wt2.transforms) && isequal(wt1.fallback, wt2.fallback)
end
function target_transform(linker::WithTransforms, vn::VarName)
    return if haskey(linker.transforms, vn)
        linker.transforms[vn]
    else
        target_transform(linker.fallback, vn)
    end
end

"""
    LinkSome(vns::Set{<:VarName}, fallback) <: AbstractTransformStrategy

Indicate that the variables in `vns` must be linked. The link statuses of other variables
are determined by the `fallback` strategy.
"""
struct LinkSome{V<:Set{<:VarName},L<:AbstractTransformStrategy} <: AbstractTransformStrategy
    vns::V
    fallback::L
end
LinkSome(::Set{<:VarName}, ::LinkAll) = LinkAll()
function target_transform(linker::LinkSome, vn::VarName)
    return if any(linker_vn -> subsumes(linker_vn, vn), linker.vns)
        DynamicLink()
    else
        target_transform(linker.fallback, vn)
    end
end
function Base.:(==)(ls1::LinkSome, ls2::LinkSome)
    return (ls1.vns == ls2.vns) & (ls1.fallback == ls2.fallback)
end
function Base.isequal(ls1::LinkSome, ls2::LinkSome)
    return isequal(ls1.vns, ls2.vns) && isequal(ls1.fallback, ls2.fallback)
end

"""
    UnlinkSome(vns::Set{<:VarName}, fallback) <: AbstractTransformStrategy

Indicate that the variables in `vns` must not be linked. The link statuses of other
variables are determined by the `fallback` strategy.
"""
struct UnlinkSome{V<:Set{<:VarName},L<:AbstractTransformStrategy} <:
       AbstractTransformStrategy
    vns::V
    fallback::L
end
UnlinkSome(::Set{<:VarName}, ::UnlinkAll) = UnlinkAll()
function target_transform(linker::UnlinkSome, vn::VarName)
    return if any(linker_vn -> subsumes(linker_vn, vn), linker.vns)
        Unlink()
    else
        target_transform(linker.fallback, vn)
    end
end
function Base.:(==)(us1::UnlinkSome, us2::UnlinkSome)
    return (us1.vns == us2.vns) & (us1.fallback == us2.fallback)
end
function Base.isequal(us1::UnlinkSome, us2::UnlinkSome)
    return isequal(us1.vns, us2.vns) && isequal(us1.fallback, us2.fallback)
end

"""
    DynamicPPL.apply_transform_strategy(
        strategy::AbstractTransformStrategy,
        tv::TransformedValue,
        vn::VarName,
        dist::Distribution,
    )

Apply the given `strategy` to the transformed value `tv` for a tilde-statement `vn ~ dist`.

Specifically, this function does a number of things:

- Calculates the raw value associated with `tv`.

- Checks whether the `strategy` expects the VarName `vn` to be linked or unlinked. If the
  current link status of `tv` matches the expected link status, `tv` is returned unchanged.
  Otherwise, either linking or unlinking is applied as necessary. Note that this function
  does not perform vectorisation unless it is needed.

  A table summarising the possible transformations is as follows:

  | tv.transform isa ...| `target_transform(...) isa DynamicLink` | `target_transform(...) isa Unlink` |
  |---------------------|---------------------------------|------------------------------------|
  | `DynamicLink`       | -> `DynamicLink`                | -> `NoTransform`                   |
  | `Unlink`            | -> `DynamicLink`                | -> `Unlink`                        |
  | `NoTransform`       | -> `DynamicLink`                | -> `NoTransform`                   |
  | `FixedTransform`    | errors                          | errors                             |

  Note that, for the last row, when using `FixedTransform` we require that `target_transform`
  exactly matches the fixed transform, otherwise an error is thrown.

- If `vn` is supposed to be linked, calculates the associated log-Jacobian adjustment for
  the **forward** linking transformation (i.e., from unlinked to linked).

This function returns a tuple of `(raw_value, new_tv, logjac)`.

!!! note
    This function is therefore the single source of truth for whether `logjac` should be
    incremented during model evaluation.
"""
function apply_transform_strategy(
    strategy::AbstractTransformStrategy,
    tv::TransformedValue{T,DynamicLink},
    vn::VarName,
    dist::Distribution,
) where {T<:AbstractVector{<:Real}}
    # tval is already linked. We need to get the raw value plus logjac
    finvlink = Bijectors.VectorBijectors.from_linked_vec(dist)
    raw_value, inv_logjac = with_logabsdet_jacobian(finvlink, get_internal_value(tv))
    target = target_transform(strategy, vn)
    return if target isa DynamicLink
        # No need to transform further
        (raw_value, tv, -inv_logjac)
    elseif target isa Unlink || target isa NoTransform
        # Need to return an unlinked value. We _could_ vectorise and generate a Unlink()
        # here, with the vectorisation transform. However, sometimes that's not needed (e.g.
        # when evaluating with an OnlyAccsVarInfo). So we just return an untransformed
        # value. If a downstream function requires a vectorised value, it's on them to
        # generate it.
        (raw_value, TransformedValue(raw_value, NoTransform()), zero(LogProbType))
    elseif target isa FixedTransform
        fwd_transform = inverse(target.transform)
        transformed_value, logjac = with_logabsdet_jacobian(fwd_transform, raw_value)
        transformed_tv = TransformedValue(transformed_value, target)
        # TODO: Check whether this should return `logjac` rather than
        # `logjac - inv_logjac`. When `tv` is already `DynamicLink` and the target is a
        # link-equivalent `FixedTransform`, the accumulator should represent only the
        # target transform's log-Jacobian. Subtracting the inverse-link Jacobian here may
        # double-count the link correction.
        (raw_value, transformed_tv, logjac - inv_logjac)
    else
        error("unknown target transform: $target")
    end
end

function apply_transform_strategy(
    strategy::AbstractTransformStrategy,
    tv::TransformedValue{T,Unlink},
    vn::VarName,
    dist::Distribution,
) where {T<:AbstractVector{<:Real}}
    invlink = Bijectors.VectorBijectors.from_vec(dist)
    raw_value = invlink(get_internal_value(tv))
    target = target_transform(strategy, vn)
    return if target isa DynamicLink
        # Need to link the value. We calculate the logjac
        flink = Bijectors.VectorBijectors.to_linked_vec(dist)
        linked_value, logjac = with_logabsdet_jacobian(flink, raw_value)
        linked_tv = TransformedValue(linked_value, DynamicLink())
        (raw_value, linked_tv, logjac)
    elseif target isa Unlink || target isa NoTransform
        # No need to transform further
        (raw_value, tv, zero(LogProbType))
    elseif target isa FixedTransform
        fwd_transform = inverse(target.transform)
        transformed_value, logjac = with_logabsdet_jacobian(fwd_transform, raw_value)
        transformed_tv = TransformedValue(transformed_value, target)
        (raw_value, transformed_tv, logjac)
    else
        error("unknown target transform: $target")
    end
end

function apply_transform_strategy(
    strategy::AbstractTransformStrategy,
    tv::TransformedValue{T,NoTransform},
    vn::VarName,
    dist::Distribution,
) where {T}
    raw_value = get_internal_value(tv)
    target = target_transform(strategy, vn)
    return if target isa DynamicLink
        # Need to link the value. We calculate the logjac
        flink = Bijectors.VectorBijectors.to_linked_vec(dist)
        linked_value, logjac = with_logabsdet_jacobian(flink, raw_value)
        linked_tv = TransformedValue(linked_value, DynamicLink())
        (raw_value, linked_tv, logjac)
    elseif target isa Unlink || target isa NoTransform
        # No need to transform further
        (raw_value, tv, zero(LogProbType))
    elseif target isa FixedTransform
        fwd_transform = inverse(target.transform)
        transformed_value, logjac = with_logabsdet_jacobian(fwd_transform, raw_value)
        transformed_tv = TransformedValue(transformed_value, target)
        (raw_value, transformed_tv, logjac)
    else
        error("unknown target transform: $target")
    end
end

function apply_transform_strategy(
    strategy::AbstractTransformStrategy,
    tv::TransformedValue{T,FixedTransform{F}},
    vn::VarName,
    dist::Distribution,
) where {T,F}
    target = target_transform(strategy, vn)
    return if target isa DynamicLink
        raw_value = get_raw_value(tv)
        flink = Bijectors.VectorBijectors.to_linked_vec(dist)
        linked_value, logjac = with_logabsdet_jacobian(flink, raw_value)
        linked_tv = TransformedValue(linked_value, DynamicLink())
        (raw_value, linked_tv, logjac)
    elseif target isa Unlink || target isa NoTransform
        raw_value = get_raw_value(tv)
        new_tv = TransformedValue(raw_value, NoTransform())
        (raw_value, new_tv, zero(LogProbType))
    elseif target isa FixedTransform
        # TODO(penelopeysm): Note that in principle we could probably allow different target
        # fixed transforms. However, for now let's keep it simple and error if it doesn't
        # match.
        if target != tv.transform
            error(
                "Variable $vn has a fixed transform, but the transform strategy expects it to be transformed differently.",
            )
        end
        raw_value, inv_logjac = with_logabsdet_jacobian(
            tv.transform.transform, get_internal_value(tv)
        )
        (raw_value, tv, -inv_logjac)
    else
        error("unknown target transform: $target")
    end
end

"""
    infer_transform_strategy_from_values(vnt::VarNamedTuple)

Takes a VNT of things with transforms, and infers a transform strategy that is consistent
with the transforms specified in the VNT. For all values `v` in the VNT, `get_transform(v)`
should return an `AbstractTransform`.
"""
function infer_transform_strategy_from_values(vnt::VarNamedTuple)
    # map_values!! might mutate the VNT, so deepcopy to avoid this
    transforms_vnt = map_values!!(get_transform, deepcopy(vnt))
    tfms = values(transforms_vnt)
    # TODO(penelopeysm): In an ideal world, could we reliably use eltype(tfms) to infer
    # this? I'm just worried about the possibility of tfms having an overly abstract type,
    # hence the check on every element individually.
    return if all(x -> x isa DynamicLink, tfms)
        LinkAll()
    elseif all(x -> (x isa Unlink || x isa NoTransform), tfms)
        UnlinkAll()
    else
        # Bundle all the transforms into a single one, with a default fallback of
        # unlinked.
        WithTransforms(transforms_vnt, UnlinkAll())
    end
end
