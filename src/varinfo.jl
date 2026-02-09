"""
    VarInfo{Linked,T<:VarNamedTuple,Accs<:AccumulatorTuple} <: AbstractVarInfo

The default implementation of `AbstractVarInfo`, storing variable values and accumulators.

The `Linked` type parameter is either `true` or `false` to mark that all variables in this
`VarInfo` are linked, or `nothing` to indicate that some variables may be linked and some
not, and a runtime check is needed.

`VarInfo` is quite a thin wrapper around a `VarNamedTuple` storing the variable values, and
a tuple of accumulators. The only really noteworthy thing about it is that it stores the
values of variables vectorised as instances of [`AbstractTransformedValue`](@ref). That is,
it stores each value as a special vector with a flag indicating whether it is just a
vectorised value ([`VectorValue`](@ref)), or whether it is also linked
([`LinkedVectorValue`](@ref)). It also stores the size of the actual post-transformation
value. These are all accessible via [`AbstractTransformedValue`](@ref).

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
struct VarInfo{Linked,T<:VarNamedTuple,Accs<:AccumulatorTuple} <: AbstractVarInfo
    values::T
    accs::Accs

    function VarInfo{Linked}(
        values::T, accs::Accs
    ) where {Linked,T<:VarNamedTuple,Accs<:AccumulatorTuple}
        return new{Linked,T,Accs}(values, accs)
    end
    function VarInfo{Linked}(
        values::T, accs::NTuple{N,AbstractAccumulator}
    ) where {Linked,T<:VarNamedTuple,N}
        return VarInfo{Linked}(values, AccumulatorTuple(accs))
    end
end

function Base.:(==)(vi1::VarInfo, vi2::VarInfo)
    return (vi1.values == vi2.values) & (vi1.accs == vi2.accs)
end
function Base.isequal(vi1::VarInfo, vi2::VarInfo)
    return isequal(vi1.values, vi2.values) && isequal(vi1.accs, vi2.accs)
end

VarInfo() = VarInfo{false}(VarNamedTuple(), default_accumulators())

"""
    VarInfo(
       [rng::AbstractRNG,]
       model::Model,
       link::AbstractTransformStrategy=UnlinkAll(),
       init::AbstractInitStrategy=InitFromPrior()
    )

Create a fresh `VarInfo` for the given model by running the model and populating it
according to the given initialisation strategy. The resulting variables in the `VarInfo` can
be linked or unlinked according to the given linking strategy.

# Arguments

- `rng::AbstractRNG`: An optional random number generator to use for any stochastic
  initialisation. If not provided, `Random.default_rng()` is used.
- `model::Model`: The model for which to create the `VarInfo`.
- `link::AbstractTransformStrategy`: An optional linking strategy (see
  [`AbstractTransformStrategy`](@ref)). Defaults to [`UnlinkAll()`](@ref), i.e., all
  variables are vectorised but not linked.
- `init::AbstractInitStrategy`: An optional initialisation strategy (see
  [`AbstractInitStrategy`](@ref)). Defaults to `InitFromPrior()`, i.e., all variables are
  initialised by sampling from their prior distributions.

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
    ::Union{UnlinkAll,UnlinkSome},
    initstrat::AbstractInitStrategy,
)
    # In this case, no variables are to be linked. We can optimise performance by directly
    # calling init!! and not faffing about with accumulators. (This does lead to significant
    # performance improvements for the typical use case of generating an unlinked VarInfo.)
    return last(init!!(rng, model, VarInfo(), initstrat, UnlinkAll()))
end
function DynamicPPL.VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    linkstrat::AbstractTransformStrategy,
    initstrat::AbstractInitStrategy=InitFromPrior(),
)
    vi = OnlyAccsVarInfo((VectorValueAccumulator(), default_accumulators()...))
    vi = last(init!!(rng, model, vi, initstrat, linkstrat))
    # Now we just need to shuffle the VectorValueAccumulator values into the VarInfo.
    # Extract the vectorised values values.
    vec_val_acc = getacc(vi, Val(VECTORVAL_ACCNAME))
    new_vi_is_linked = if linkstrat isa LinkAll
        true
    else
        # TODO(penelopeysm): We can definitely do better here. The linking accumulator can
        # keep track of whether any variables were linked or unlinked, and we can use that
        # here. It won't be type-stable, but that's fine, right now it isn't either.
        nothing
    end
    return VarInfo{new_vi_is_linked}(
        vec_val_acc.values, DynamicPPL.deleteacc!!(vi.accs, Val(VECTORVAL_ACCNAME))
    )
end
function DynamicPPL.VarInfo(
    model::Model,
    linkstrat::AbstractTransformStrategy,
    initstrat::AbstractInitStrategy=InitFromPrior(),
)
    return DynamicPPL.VarInfo(Random.default_rng(), model, linkstrat, initstrat)
end
function VarInfo(
    rng::Random.AbstractRNG, model::Model, initstrat::AbstractInitStrategy=InitFromPrior()
)
    return VarInfo(rng, model, UnlinkAll(), initstrat)
end
function VarInfo(model::Model, initstrat::AbstractInitStrategy=InitFromPrior())
    return VarInfo(Random.default_rng(), model, initstrat)
end

get_values(vi::VarInfo) = vi.values

getaccs(vi::VarInfo) = vi.accs
function setaccs!!(vi::VarInfo{Linked}, accs::AccumulatorTuple) where {Linked}
    return VarInfo{Linked}(vi.values, accs)
end

transformation(::VarInfo) = DynamicTransformation()

function Base.copy(vi::VarInfo{Linked}) where {Linked}
    return VarInfo{Linked}(copy(vi.values), copy(getaccs(vi)))
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

function Base.show(io::IO, ::MIME"text/plain", vi::VarInfo{link}) where {link}
    printstyled(io, "VarInfo"; bold=true)
    print(io, " {linked=$link}\n")
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
Base.empty(vi::VarInfo) = VarInfo{false}(empty(vi.values), map(reset, vi.accs))
BangBang.empty!!(vi::VarInfo) = VarInfo{false}(empty!!(vi.values), map(reset, vi.accs))

"""
    setindex_internal!!(vi::VarInfo, val, vn::VarName)

Set the internal (vectorised) value of variable `vn` in `vi` to `val`.

This does not change the transformation or linked status of the variable.
"""
function setindex_internal!!(vi::VarInfo{Linked}, val, vn::VarName) where {Linked}
    old_tv = getindex(vi.values, vn)
    new_tv = set_internal_value(old_tv, val)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VarInfo{Linked}(new_values, vi.accs)
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
    vi::VarInfo{Linked},
    tval::Union{VectorValue,LinkedVectorValue},
    ::Distribution,
    vn::VarName,
    template::Any,
) where {Linked}
    NewLinked = if tval isa LinkedVectorValue && ((Linked == true) || isempty(vi))
        true
    elseif tval isa VectorValue && ((Linked == false) || isempty(vi))
        false
    else
        nothing
    end
    return VarInfo{NewLinked}(templated_setindex!!(vi.values, tval, vn, template), vi.accs)
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
    vi::VarInfo{Linked}, tval::UntransformedValue, dist::Distribution, vn::VarName, template
) where {Linked}
    raw_value = DynamicPPL.get_internal_value(tval)
    sz = hasmethod(size, (typeof(raw_value),)) ? size(raw_value) : ()
    tval = VectorValue(to_vec_transform(dist)(raw_value), from_vec_transform(dist), sz)
    return setindex_with_dist!!(vi, tval, dist, vn, template)
end

"""
    set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)

Set the linked status of variable `vn` in `vi` to `linked`.

Note that this function is potentially unsafe as it does not change the value or
transformation of the variable!
"""
function set_transformed!!(vi::VarInfo{Linked}, linked::Bool, vn::VarName) where {Linked}
    old_tv = getindex(vi.values, vn)
    new_tv = if linked
        LinkedVectorValue(old_tv.val, old_tv.transform, old_tv.size)
    else
        VectorValue(old_tv.val, old_tv.transform, old_tv.size)
    end
    new_values = setindex!!(vi.values, new_tv, vn)
    new_linked = Linked == linked ? Linked : nothing
    return VarInfo{new_linked}(new_values, vi.accs)
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
        ctor(tv.val, tv.transform, tv.size)
    end
    return VarInfo{linked}(new_values, vi.accs)
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

function is_transformed(vi::VarInfo{Linked}, vn::VarName) where {Linked}
    return if Linked === nothing
        getindex(vi.values, vn) isa LinkedVectorValue
    else
        Linked
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

function _update_link_status!!(
    orig_vi::VarInfo,
    transform_strategy::AbstractTransformStrategy,
    model::Model,
    ::Val{link},
) where {link}
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
    return VarInfo{link}(new_vector_vals.values, orig_vi.accs)
end

"""
    DynamicPPL.update_link_status!!(
        orig_vi::VarInfo, linker::AbstractTransformStrategy, model::Model,
    )::VarInfo

Create a new VarInfo based on `orig_vi`, but with the link statuses of variables updated
according to `linker`.
"""
function update_link_status!!(vi::VarInfo, ::LinkAll, model::Model)
    return _update_link_status!!(vi, LinkAll(), model, Val(true))
end
function update_link_status!!(vi::VarInfo, ::UnlinkAll, model::Model)
    return _update_link_status!!(vi, UnlinkAll(), model, Val(false))
end
function update_link_status!!(vi::VarInfo, l::AbstractTransformStrategy, model::Model)
    # In other cases, we can't (easily) infer anything about the overall linked status of
    # the VarInfo.
    return _update_link_status!!(vi, l, model, Val(nothing))
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
get_transform_strategy(::VarInfo{true}) = LinkAll()
get_transform_strategy(::VarInfo{false}) = UnlinkAll()
function get_transform_strategy(vi::VarInfo{nothing})
    all_vns = keys(vi)
    linked_vns = Set{VarName}()
    unlinked_vns = Set{VarName}()
    for vn in all_vns
        if is_transformed(vi, vn)
            push!(linked_vns, vn)
        else
            push!(unlinked_vns, vn)
        end
    end
    return if isempty(linked_vns)
        UnlinkAll()
    elseif isempty(unlinked_vns)
        LinkAll()
    else
        # Link exactly those that are linked, unlink exactly those that are unlinked,
        # and for everything that is completely new, link it.
        LinkSome(linked_vns, UnlinkSome(unlinked_vns, LinkAll()))
    end
end

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
            return $T(new_val, tv.transform, tv.size)
        end
    end
end
function unflatten!!(vi::VarInfo{Linked}, vec::AbstractVector) where {Linked}
    vci = VectorChunkIterator!(vec, 1)
    new_values = map_values!!(vci, vi.values)
    return VarInfo{Linked}(new_values, vi.accs)
end

"""
    subset(varinfo::VarInfo, vns)

Create a new `VarInfo` containing only the variables in `vns`.

`vns` can be almost any collection of `VarName`s, e.g. a `Set`, `Vector`, or `Tuple`.
"""
function subset(varinfo::VarInfo{Linked}, vns) where {Linked}
    new_values = subset(varinfo.values, vns)
    return VarInfo{Linked}(new_values, map(copy, getaccs(varinfo)))
end

"""
    merge(varinfo_left::VarInfo, varinfo_right::VarInfo)

Merge two `VarInfo`s into a new `VarInfo` containing all variables from both.

The accumulators are taken exclusively from `varinfo_right`.

If a variable exists in both `varinfo_left` and `varinfo_right`, the value from
`varinfo_right` is used.
"""
function Base.merge(
    varinfo_left::VarInfo{LinkedLeft}, varinfo_right::VarInfo{LinkedRight}
) where {LinkedLeft,LinkedRight}
    new_values = merge(varinfo_left.values, varinfo_right.values)
    new_accs = map(copy, getaccs(varinfo_right))
    new_linked = if LinkedLeft == LinkedRight
        LinkedLeft
    else
        # TODO(mhauru) Consider doing something more clever here, e.g. checking whether
        # either varinfo_left or varinfo_right is empty, or actually iterating over all the
        # values to check their linked status. Needs to balance keeping the type parameter
        # alive vs runtime costs.
        nothing
    end
    return VarInfo{new_linked}(new_values, new_accs)
end
