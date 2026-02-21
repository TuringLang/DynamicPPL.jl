"""
    VarInfo{
        Tfm<:AbstractTransformStrategy,
        T<:VarNamedTuple,
        Accs<:AccumulatorTuple
    } <: AbstractVarInfo

The default implementation of `AbstractVarInfo`, storing variable values and accumulators.

`VarInfo` is quite a thin wrapper around a `VarNamedTuple` storing the variable values, and
a tuple of accumulators. The only really noteworthy thing about it is that it stores the
values of variables vectorised as instances of [`AbstractTransformedValue`](@ref). That is,
it stores each value as a special vector with a flag indicating whether it is just a
vectorised value ([`VectorValue`](@ref)), or whether it is also linked
([`LinkedVectorValue`](@ref)). It also stores the size of the actual post-transformation
value. These are all accessible via [`AbstractTransformedValue`](@ref).

`VarInfo` additionally stores a transform strategy, which reflects the linked status of
variables inside the `VarInfo`. For example, a `VarInfo{LinkAll}` should contain only
`LinkedVectorValue`s in its `values` field.

Because the job of `VarInfo` is to store transformed values, there is no generic
`setindex!!` implementation on `VarInfo` itself. Instead, all storage must go via
[`setindex_with_dist!!`](@ref), which takes care of storing the value in the correct
transformed form. This in turn means that the distribution on the right-hand side of a
tilde-statement must be available when modifying a VarInfo.

You can use `getindex` on `VarInfo` to obtain values in the support of the original
distribution. To directly get access to the internal vectorised values, use
[`getindex_internal`](@ref), [`setindex_internal!!`](@ref), and [`unflatten!!`](@ref).

For more details on the internal storage, see documentation of
[`AbstractTransformedValue`](@ref) and [`VarNamedTuple`](@ref).

# Fields
$(TYPEDFIELDS)

"""
struct VarInfo{Tfm<:AbstractTransformStrategy,T<:VarNamedTuple,Accs<:AccumulatorTuple} <:
       AbstractVarInfo
    transform_strategy::Tfm
    values::T
    accs::Accs

    function VarInfo(
        tfm_strategy::Tfm, values::T, accs::Accs
    ) where {Tfm<:AbstractTransformStrategy,T<:VarNamedTuple,Accs<:AccumulatorTuple}
        return new{Tfm,T,Accs}(tfm_strategy, values, accs)
    end
    function VarInfo(
        tfm_strategy::Tfm, values::T, accs::NTuple{N,AbstractAccumulator}
    ) where {Tfm<:AbstractTransformStrategy,T<:VarNamedTuple,N}
        return VarInfo(tfm_strategy, values, AccumulatorTuple(accs))
    end
end

function Base.:(==)(vi1::VarInfo, vi2::VarInfo)
    return (vi1.transform_strategy == vi2.transform_strategy) &
           (vi1.values == vi2.values) &
           (vi1.accs == vi2.accs)
end
function Base.isequal(vi1::VarInfo, vi2::VarInfo)
    return isequal(vi1.transform_strategy, vi2.transform_strategy) &&
           isequal(vi1.values, vi2.values) &&
           isequal(vi1.accs, vi2.accs)
end

VarInfo() = VarInfo(UnlinkAll(), VarNamedTuple(), default_accumulators())

"""
    VarInfo(
       [rng::AbstractRNG,]
       model::Model,
       init_strategy::AbstractInitStrategy=InitFromPrior()
       transform_strategy::AbstractTransformStrategy=UnlinkAll(),
    )

Create a fresh `VarInfo` for the given model by running the model and populating it
according to the given initialisation strategy. The resulting variables in the `VarInfo` can
be linked or unlinked according to the given linking strategy.

# Arguments

- `rng::AbstractRNG`: An optional random number generator to use for any stochastic
  initialisation. If not provided, `Random.default_rng()` is used.
- `model::Model`: The model for which to create the `VarInfo`.
- `init_strategy::AbstractInitStrategy`: An optional initialisation strategy (see
  [`AbstractInitStrategy`](@ref)). Defaults to `InitFromPrior()`, i.e., all variables are
  initialised by sampling from their prior distributions.
- `transform_strategy::AbstractTransformStrategy`: An optional linking strategy (see
  [`AbstractTransformStrategy`](@ref)). Defaults to [`UnlinkAll()`](@ref), i.e., all
  variables are vectorised but not linked.

# Extended help

## Performance characteristics of linked VarInfo

This method allows the immediate generation of a linked VarInfo, which was not possible in
previous versions of DynamicPPL. It is guaranteed that `link!!(VarInfo(rng, model), model)`
(the old way of instantiating a linked `VarInfo`) is equivalent to `VarInfo(rng, model,
LinkAll())`.

Depending on the model, each of these two methods may be more performant, although the
reasons for this are still somewhat unclear. Small models tend to do better with
instantiating an unlinked `VarInfo` and then linking it, while large models tend to do
better with directly instantiating a linked `VarInfo`. The hope is that this generally does
not impact usage since linking is not typically something done in performance-critical
sections of Turing.jl. If linking performance is critical, it is recommended to benchmark
both methods for the specific model in question.
"""
function DynamicPPL.VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy,
    ::Union{UnlinkAll,UnlinkSome},
)
    # In this case, no variables are to be linked. We can optimise performance by directly
    # calling init!! and not faffing about with accumulators. (This does lead to significant
    # performance improvements for the typical use case of generating an unlinked VarInfo.)
    return last(init!!(rng, model, VarInfo(), init_strategy, UnlinkAll()))
end
function DynamicPPL.VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy,
)
    vi = OnlyAccsVarInfo((VectorValueAccumulator(), default_accumulators()...))
    vi = last(init!!(rng, model, vi, init_strategy, transform_strategy))
    # Now we just need to shuffle the VectorValueAccumulator values into the VarInfo.
    # Extract the vectorised values values.
    vec_val_acc = getacc(vi, Val(VECTORVAL_ACCNAME))
    return VarInfo(
        transform_strategy,
        vec_val_acc.values,
        DynamicPPL.deleteacc!!(vi.accs, Val(VECTORVAL_ACCNAME)),
    )
end
function DynamicPPL.VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return VarInfo(rng, model, init_strategy, UnlinkAll())
end
function DynamicPPL.VarInfo(
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    return VarInfo(Random.default_rng(), model, init_strategy, transform_strategy)
end

get_values(vi::VarInfo) = vi.values
getaccs(vi::VarInfo) = vi.accs
function setaccs!!(vi::VarInfo, accs::AccumulatorTuple)
    return VarInfo(vi.transform_strategy, vi.values, accs)
end

transformation(::VarInfo) = DynamicTransformation()

function Base.copy(vi::VarInfo)
    return VarInfo(vi.transform_strategy, copy(vi.values), copy(getaccs(vi)))
end
Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.values, vn)
Base.length(vi::VarInfo) = length(vi.values)
Base.keys(vi::VarInfo) = keys(vi.values)
# TODO(penelopeysm): Right now, this returns Vector{Any}. We could use init=Union{} and
# BangBang.push!! instead of push!, which would give the resulting vector as concrete an
# eltype as possible. However, that is type unstable since it is inferred as
# Union{Vector{Union{}}, Vector{Float64}} (I suppose this is because it can't tell whether
# the result will be empty or not...? Not sure).
function Base.values(vi::VarInfo)
    return mapreduce(
        p -> DynamicPPL.get_transform(p.second)(DynamicPPL.get_internal_value(p.second)),
        push!,
        vi.values;
        init=Any[],
    )
end

function Base.show(io::IO, ::MIME"text/plain", vi::VarInfo)
    printstyled(io, "VarInfo"; bold=true)
    print(io, "\n ├─ ")
    printstyled(io, "transform_strategy: "; bold=true)
    print(io, vi.transform_strategy)
    println(io)
    print(io, " ├─ ")
    if isempty(vi.values)
        printstyled(io, "values"; bold=true)
        println(io, " (empty)")
    else
        printstyled(io, "values"; bold=true)
        print(io, "\n │  ")
        DynamicPPL.VarNamedTuples.vnt_pretty_print(io, vi.values, " │  ", 0)
        println(io)
    end
    print(io, " └─ ")
    printstyled(io, "accs"; bold=true)
    print(io, "\n    ")
    DynamicPPL.pretty_print(io, vi.accs, "    ")
    return nothing
end

function Base.getindex(vi::VarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return DynamicPPL.get_transform(tv)(DynamicPPL.get_internal_value(tv))
end
function Base.getindex(vi::VarInfo, vns::AbstractVector{<:VarName})
    return [getindex(vi, vn) for vn in vns]
end

Base.isempty(vi::VarInfo) = isempty(vi.values)
Base.empty(vi::VarInfo) = VarInfo(UnlinkAll(), empty(vi.values), map(reset, vi.accs))
function BangBang.empty!!(vi::VarInfo)
    return VarInfo(UnlinkAll(), empty!!(vi.values), map(reset, vi.accs))
end

"""
    setindex_internal!!(vi::VarInfo, val, vn::VarName)

Set the internal (vectorised) value of variable `vn` in `vi` to `val`.

This does not change the transformation or linked status of the variable.
"""
function setindex_internal!!(vi::VarInfo, val, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = set_internal_value(old_tv, val)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VarInfo(vi.transform_strategy, new_values, vi.accs)
end

"""
    update_transform_strategy(
        tfm_strategy::AbstractTransformStrategy,
        vi_is_empty::Bool,
        new_vn::VarName,
        new_vn_is_linked::Bool
    )

Given an old transform strategy `tfm_strategy`, and the linked status of a new variable
`new_vn` to be added to a `VarInfo` with that transform strategy, return an updated
transform strategy that accounts for the addition of `new_vn`.
"""
function update_transform_strategy(
    tfm_strategy::AbstractTransformStrategy,
    vi_is_empty::Bool,
    new_vn::VarName,
    new_vn_is_linked::Bool,
)
    if new_vn_is_linked
        if tfm_strategy isa LinkAll || vi_is_empty
            LinkAll()
        elseif target_transform(tfm_strategy, new_vn) isa DynamicLink
            # can reuse
            tfm_strategy
        else
            # have to wrap
            LinkSome(Set([new_vn]), tfm_strategy)
        end
    else
        if tfm_strategy isa UnlinkAll || vi_is_empty
            UnlinkAll()
        elseif target_transform(tfm_strategy, new_vn) isa Unlink
            tfm_strategy
        else
            UnlinkSome(Set([new_vn]), tfm_strategy)
        end
    end
end

"""
    setindex_with_dist!!(
        vi::VarInfo,
        tval::Union{VectorValue,LinkedVectorValue},
        dist::Distribution,
        vn::VarName,
        template::Any,
    )

Set the value of `vn` in `vi` to `tval`. Note that this will cause the linked status of `vi`
to update according to what `tval` is. That means that whether or not a variable is
considered to be 'linked' is determined by `tval` rather than the previous status of `vi`.
"""
function setindex_with_dist!!(
    vi::VarInfo,
    tval::Union{VectorValue,LinkedVectorValue},
    ::Distribution,
    vn::VarName,
    template::Any,
)
    new_transform_strategy = update_transform_strategy(
        vi.transform_strategy, isempty(vi), vn, tval isa LinkedVectorValue
    )
    return VarInfo(
        new_transform_strategy, templated_setindex!!(vi.values, tval, vn, template), vi.accs
    )
end

"""
    setindex_with_dist!!(
        vi::VarInfo,
        tval::UntransformedValue,
        dist::Distribution,
        vn::VarName,
        template::Any
    )

Vectorise `tval` (into a `VectorValue`) and store it. (Note that if `setindex_with_dist!!`
receives an `UntransformedValue`, the variable is always considered unlinked, since if it
were to be linked, `apply_transform_strategy` will already have done so.)
"""
function setindex_with_dist!!(
    vi::VarInfo, tval::UntransformedValue, dist::Distribution, vn::VarName, template
)
    raw_value = DynamicPPL.get_internal_value(tval)
    tval = VectorValue(to_vec_transform(dist)(raw_value), from_vec_transform(dist))
    return setindex_with_dist!!(vi, tval, dist, vn, template)
end

"""
    set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)

Set the linked status of variable `vn` in `vi` to `linked`.

Note that this function is potentially unsafe as it does not change the value or
transformation of the variable!
"""
function set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = if linked
        LinkedVectorValue(old_tv.val, old_tv.transform)
    else
        VectorValue(old_tv.val, old_tv.transform)
    end
    new_values = setindex!!(vi.values, new_tv, vn)
    new_transform_strategy = update_transform_strategy(
        vi.transform_strategy, isempty(vi), vn, linked
    )
    return VarInfo(new_transform_strategy, new_values, vi.accs)
end

# VarInfo does not care whether the transformation was Static or Dynamic, it just tracks
# whether one was applied at all.
function set_transformed!!(vi::VarInfo, ::AbstractTransformation, vn::VarName)
    return set_transformed!!(vi, true, vn)
end

set_transformed!!(vi::VarInfo, ::AbstractTransformation) = set_transformed!!(vi, true)

function set_transformed!!(vi::VarInfo, ::NoTransformation, vn::VarName)
    return set_transformed!!(vi, false, vn)
end

set_transformed!!(vi::VarInfo, ::NoTransformation) = set_transformed!!(vi, false)

function set_transformed!!(vi::VarInfo, linked::Bool)
    ctor = linked ? LinkedVectorValue : VectorValue
    new_values = map_values!!(vi.values) do tv
        ctor(tv.val, tv.transform)
    end
    new_transform_strategy = linked ? LinkAll() : UnlinkAll()
    return VarInfo(new_transform_strategy, new_values, vi.accs)
end

"""
    getindex_internal(vi::VarInfo, vn::VarName)

Get the internal (vectorised) value of variable `vn` in `vi`.
"""
getindex_internal(vi::VarInfo, vn::VarName) = getindex(vi.values, vn).val
# TODO(mhauru) The below should be removed together with unflatten!!.
getindex_internal(vi::VarInfo, ::Colon) = internal_values_as_vector(vi)

"""
    get_transformed_value(vi::VarInfo, vn::VarName)

Get the entire `AbstractTransformedValue` for variable `vn` in `vi`.
"""
get_transformed_value(vi::VarInfo, vn::VarName) = getindex(vi.values, vn)

function is_transformed(vi::VarInfo, vn::VarName)
    return if vi.transform_strategy isa LinkAll
        true
    elseif vi.transform_strategy isa UnlinkAll
        false
    else
        getindex(vi.values, vn) isa LinkedVectorValue
    end
end

function from_internal_transform(::VarInfo, ::VarName, dist::Distribution)
    return from_vec_transform(dist)
end

function from_linked_internal_transform(::VarInfo, ::VarName, dist::Distribution)
    return from_linked_vec_transform(dist)
end

function from_internal_transform(vi::VarInfo, vn::VarName)
    return DynamicPPL.get_transform(getindex(vi.values, vn))
end

function from_linked_internal_transform(vi::VarInfo, vn::VarName)
    if !is_transformed(vi, vn)
        error("Variable $vn is not linked; cannot get linked transformation.")
    end
    return DynamicPPL.get_transform(getindex(vi.values, vn))
end

"""
    internal_values_as_vector(vi::VarInfo)

Convert `vi.values` into a single vector by concatenating the internal (vectorised) values
of all variables in `vi`. The order of concatenation is determined by the order of
`keys(vi.values)`.

Note that this is a lossy operation as it discards all information about the transformations
and variable names in `vi`.

This is the inverse of [`unflatten!!`](@ref).
"""
internal_values_as_vector(vi::VarInfo) = internal_values_as_vector(vi.values)

"""
    DynamicPPL.update_link_status!!(
        orig_vi::VarInfo, transform_strategy::AbstractTransformStrategy, model::Model
    )

Given an original `VarInfo` `orig_vi`, update the link status of its variables according to
the new `transform_strategy`.
"""
function update_link_status!!(
    orig_vi::VarInfo, transform_strategy::AbstractTransformStrategy, model::Model
)
    # We'll just recalculate logjac from the start, rather than trying to adjust the old
    # one.
    new_vi = OnlyAccsVarInfo((VectorValueAccumulator(), LogJacobianAccumulator()))
    new_vi = last(
        init!!(model, new_vi, InitFromParamsUnsafe(orig_vi.values), transform_strategy)
    )
    new_vector_vals = getacc(new_vi, Val(VECTORVAL_ACCNAME))
    if hasacc(orig_vi, Val(:LogJacobian))
        orig_vi = setlogjac!!(orig_vi, getlogjac(new_vi))
    end
    return VarInfo(transform_strategy, new_vector_vals.values, orig_vi.accs)
end

function link!!(::DynamicTransformation, vi::VarInfo, vns, model::Model)
    return update_link_status!!(vi, LinkSome(Set(vns), get_transform_strategy(vi)), model)
end
function invlink!!(::DynamicTransformation, vi::VarInfo, vns, model::Model)
    return update_link_status!!(vi, UnlinkSome(Set(vns), get_transform_strategy(vi)), model)
end
function link!!(::DynamicTransformation, vi::VarInfo, model::Model)
    return update_link_status!!(vi, LinkAll(), model)
end
function invlink!!(::DynamicTransformation, vi::VarInfo, model::Model)
    return update_link_status!!(vi, UnlinkAll(), model)
end

function link!!(t::StaticTransformation{<:Bijectors.Transform}, vi::VarInfo, ::Model)
    # TODO(mhauru) This assumes that the user has defined the bijector using the same
    # variable ordering as what `vi[:]` and `unflatten!!(vi, x)` use. This is a bad user
    # interface.
    b = inverse(t.bijector)
    x = vi[:]
    y, logjac = with_logabsdet_jacobian(b, x)
    # TODO(mhauru) This doesn't set the transforms of `vi`. With the old Metadata that meant
    # that getindex(vi, vn) would apply the default link transform of the distribution. With
    # the new VarNamedTuple-based VarInfo it means that getindex(vi, vn) won't apply any
    # link transform. Neither is correct, rather the transform should be the inverse of b.
    vi = unflatten!!(vi, y)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, logjac)
    end
    return set_transformed!!(vi, t)
end

function invlink!!(t::StaticTransformation{<:Bijectors.Transform}, vi::VarInfo, ::Model)
    b = t.bijector
    y = vi[:]
    x, inv_logjac = with_logabsdet_jacobian(b, y)

    # Mildly confusing: we need to _add_ the logjac of the inverse transform,
    # because we are trying to remove the logjac of the forward transform
    # that was previously accumulated when linking.
    vi = unflatten!!(vi, x)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, inv_logjac)
    end
    return set_transformed!!(vi, NoTransformation())
end

"""
    get_transform_strategy(vi::VarInfo)

In `tilde_assume!!(::InitContext, ...)`, we use a transform strategy to determine whether
variables should be considered to be in linked space or unlinked space. This allows us to
determine whether `logjac` should be accumulated or not.

However, there are still a number of places in DynamicPPL where we want to make this
decision as to whether a variable is linked or unlinked based on the current status of the
variable inside a `VarInfo`.

This function acts as the bridge between the two: it extracts an appropriate
`AbstractTransformStrategy` from the current status of variables in a `VarInfo`.
"""
get_transform_strategy(vi::VarInfo) = vi.transform_strategy

"""
    VectorChunkIterator{T<:AbstractVector}

A tiny struct for getting chunks of a vector sequentially.

The only function provided is `get_next_chunk!`, which takes a length and returns
a view into the next chunk of that length, updating the internal index.
"""
mutable struct VectorChunkIterator!{T<:AbstractVector}
    vec::T
    index::Int
end
for T in (:VectorValue, :LinkedVectorValue)
    @eval begin
        function (vci::VectorChunkIterator!)(tv::$T)
            old_val = tv.val
            len = length(old_val)
            new_val = @view vci.vec[(vci.index):(vci.index + len - 1)]
            vci.index += len
            return $T(new_val, tv.transform)
        end
    end
end
function unflatten!!(vi::VarInfo, vec::AbstractVector)
    vci = VectorChunkIterator!(vec, 1)
    new_values = map_values!!(vci, vi.values)
    return VarInfo(vi.transform_strategy, new_values, vi.accs)
end

"""
    subset(varinfo::VarInfo, vns)

Create a new `VarInfo` containing only the variables in `vns`.

`vns` can be almost any collection of `VarName`s, e.g. a `Set`, `Vector`, or `Tuple`.
"""
function subset(varinfo::VarInfo, vns)
    new_values = subset(varinfo.values, vns)
    # TODO(penelopeysm): We could potentially be smarter here and see whether the transform
    # strategy can be updated to be LinkAll or UnlinkAll.
    return VarInfo(varinfo.transform_strategy, new_values, map(copy, getaccs(varinfo)))
end

"""
    merge(varinfo_left::VarInfo, varinfo_right::VarInfo)

Merge two `VarInfo`s into a new `VarInfo` containing all variables from both.

The accumulators are taken exclusively from `varinfo_right`.

If a variable exists in both `varinfo_left` and `varinfo_right`, the value from
`varinfo_right` is used.
"""
function Base.merge(varinfo_left::VarInfo, varinfo_right::VarInfo)
    new_values = merge(varinfo_left.values, varinfo_right.values)
    new_accs = map(copy, getaccs(varinfo_right))
    new_transform_strategy =
        if varinfo_left.transform_strategy isa LinkAll &&
            varinfo_right.transform_strategy isa LinkAll
            LinkAll()
        elseif varinfo_left.transform_strategy isa UnlinkAll &&
            varinfo_right.transform_strategy isa UnlinkAll
            UnlinkAll()
        else
            linked_vns = Set{VarName}()
            unlinked_vns = Set{VarName}()
            for (vn, tval) in pairs(new_values)
                if tval isa LinkedVectorValue
                    push!(linked_vns, vn)
                else
                    push!(unlinked_vns, vn)
                end
            end
            if isempty(linked_vns)
                UnlinkAll()
            elseif isempty(unlinked_vns)
                LinkAll()
            else
                LinkSome(linked_vns, UnlinkSome(unlinked_vns, LinkAll()))
            end
        end
    return VarInfo(new_transform_strategy, new_values, new_accs)
end
