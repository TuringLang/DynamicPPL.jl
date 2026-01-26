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

Note that `setindex!!` and `getindex` on `VarInfo` take and return values in the support of
the original distribution. To get access to the internal vectorised values, use
[`getindex_internal`](@ref), [`setindex_internal!!`](@ref), and [`unflatten!!`](@ref).

There's also a `VarInfo`-specific function [`setindex_with_dist!!`](@ref), which sets a
variable's value with a transformation based on the statistical distribution this value is
a sample for.

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
end

function Base.:(==)(vi1::VarInfo, vi2::VarInfo)
    return (vi1.values == vi2.values) & (vi1.accs == vi2.accs)
end
function Base.isequal(vi1::VarInfo, vi2::VarInfo)
    return isequal(vi1.values, vi2.values) && isequal(vi1.accs, vi2.accs)
end

VarInfo() = VarInfo{false}(VarNamedTuple(), default_accumulators())

function VarInfo(values::Union{NamedTuple,AbstractDict})
    vi = VarInfo()
    for (k, v) in pairs(values)
        vn = k isa Symbol ? VarName{k}() : k
        vi = setindex!!(vi, v, vn)
    end
    return vi
end

"""
    VarInfo(
       [rng::AbstractRNG,]
       model::Model,
       link::AbstractLinkStrategy=UnlinkAll(),
       init::AbstractInitStrategy=InitFromPrior()
    )

Create a fresh `VarInfo` for the given model by running the model and populating it
according to the given initialisation strategy. The resulting variables in the `VarInfo` can
be linked or unlinked according to the given linking strategy.

# Arguments

- `rng::AbstractRNG`: An optional random number generator to use for any stochastic
  initialisation. If not provided, `Random.default_rng()` is used.
- `model::Model`: The model for which to create the `VarInfo`.
- `link::AbstractLinkStrategy`: An optional linking strategy (see `AbstractLinkStrategy`).
  Defaults to `UnlinkAll()`, i.e., all variables are vectorised but not linked.
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
    return last(init!!(rng, model, VarInfo(), initstrat))
end
function DynamicPPL.VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    linkstrat::AbstractLinkStrategy,
    initstrat::AbstractInitStrategy=InitFromPrior(),
)
    linked_value_acc = VNTAccumulator{LINK_ACCNAME}(Link!(linkstrat))
    vi = OnlyAccsVarInfo((linked_value_acc, default_accumulators()...))
    vi = last(init!!(rng, model, vi, initstrat))
    # Extract the linked values and the change in logjac.
    link_acc = getacc(vi, Val(LINK_ACCNAME))
    new_vi_is_linked = if linkstrat isa LinkAll
        true
    else
        # TODO(penelopeysm): We can definitely do better here. The linking accumulator can
        # keep track of whether any variables were linked or unlinked, and we can use that
        # here. It won't be type-stable, but that's fine, right now it isn't either.
        nothing
    end
    vi = VarInfo{new_vi_is_linked}(
        link_acc.values, DynamicPPL.deleteacc(vi.accs, Val(LINK_ACCNAME))
    )
    vi = acclogjac!!(vi, link_acc.f.logjac)
    return vi
end
function DynamicPPL.VarInfo(
    model::Model,
    linkstrat::AbstractLinkStrategy,
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
    return mapreduce(p -> DynamicPPL.get_true_value(p.second), push!, vi.values; init=Any[])
end

function Base.show(io::IO, ::MIME"text/plain", vi::VarInfo)
    printstyled(io, "VarInfo\n"; bold=true)
    print(io, " ├─ ")
    printstyled("values"; bold=true)
    print(io, "\n │  ")
    DynamicPPL.VarNamedTuples.vnt_pretty_print(io, vi.values, " │  ", 0)
    println(io)
    print(io, " └─ ")
    printstyled("accs"; bold=true)
    print(io, "\n    ")
    DynamicPPL.pretty_print(io, vi.accs, "    ")
    return nothing
end

function Base.getindex(vi::VarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return DynamicPPL.get_true_value(tv)
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
    new_tv = update_value(old_tv, val)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VarInfo{Linked}(new_values, vi.accs)
end

# TODO(mhauru) It shouldn't really be VarInfo's business to know about `dist`. However,
# we need `dist` to determine the linking transformation (or even just the vectorisation
# transformation in the case of ProductNamedTupleDistribions), and if we leave the work
# of doing the transformation to the caller (tilde_assume!!), it'll be done even when e.g.
# using OnlyAccsVarInfo. Hence having this function. It should eventually hopefully be
# removed once VAIMAcc is the only way to get values out of an evaluation.
"""
    setindex_with_dist!!(vi::VarInfo, val, dist::Distribution, vn::VarName)

Set the value of `vn` in `vi` to `val`, applying a transformation based on `dist`.

`val` is taken to be the actual value of the variable, and is transformed into the internal
(vectorised) representation using a transformation based on `dist`. If the variable is
currently linked in `vi`, or doesn't exist in `vi` but all other variables in `vi` are
linked, the linking transformation is used; otherwise, the standard vector transformation is
used.

Returns three things:
 - the modified `vi`,
 - the log absolute determinant of the Jacobian of the transformation applied,
 - the `AbstractTransformedValue` used to store the value.
"""
function setindex_with_dist!!(
    vi::VarInfo{Linked}, val, dist::Distribution, vn::VarName, template
) where {Linked}
    link = if Linked === nothing
        haskey(vi, vn) ? is_transformed(vi, vn) : is_transformed(vi)
    else
        Linked
    end
    transform = if link
        to_linked_vec_transform(dist)
    else
        to_vec_transform(dist)
    end
    transformed_val, logjac = with_logabsdet_jacobian(transform, val)
    # All values for which `size` is not defined are assumed to be scalars.
    val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    tv = if link
        LinkedVectorValue(transformed_val, inverse(transform), val_size)
    else
        VectorValue(transformed_val, inverse(transform), val_size)
    end
    new_linked = Linked == link ? Linked : nothing
    vi = VarInfo{new_linked}(templated_setindex!!(vi.values, tv, vn, template), vi.accs)
    return vi, logjac, tv
end

# TODO(mhauru) The below is somewhat unsafe or incomplete: For instance, from_vec_transform
# isn't defined for NamedTuples. However, this is needed in some places where values for
# in a VarInfo are set outside the context of a `tilde_assume!!` and no distribution is
# available. Hopefully we'll get rid of this eventually.
"""
    setindex!!(vi::VarInfo, val, vn::VarName)

Set the value of `vn` in `vi` to `val`.

The transformation for `vn` is reset to be the standard vector transformation for values of
the type of `val` and linking status is set to false.
"""
function BangBang.setindex!!(vi::VarInfo{Linked}, val, vn::VarName) where {Linked}
    # TODO(penelopeysm) This function is BS, really should get rid of it asap
    new_linked = Linked == false ? false : nothing
    transform = from_vec_transform(val)
    transformed_val = inverse(transform)(val)
    tv = VectorValue(transformed_val, transform, size(val))
    return VarInfo{new_linked}(setindex!!(vi.values, tv, vn), vi.accs)
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
getindex_internal(vi::VarInfo, ::Colon) = values_as(vi, Vector)

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

# TODO(penelopeysm): In principle, `link` can be statically determined from the type of
# `linker`. However, I'm not sure if doing that could mess with type stability.
function _link_or_invlink!!(
    orig_vi::VarInfo, linker::AbstractLinkStrategy, model::Model, ::Val{link}
) where {link}
    linked_value_acc = VNTAccumulator{LINK_ACCNAME}(Link!(linker))
    new_vi = OnlyAccsVarInfo((linked_value_acc,))
    new_vi = last(init!!(model, new_vi, InitFromParamsUnsafe(orig_vi.values)))
    link_acc = getacc(new_vi, Val(LINK_ACCNAME))
    new_vi = VarInfo{link}(link_acc.values, orig_vi.accs)
    if hasacc(new_vi, Val(:LogJacobian))
        new_vi = acclogjac!!(new_vi, link_acc.f.logjac)
    end
    return new_vi
end

function link!!(::DynamicTransformation, vi::VarInfo, vns, model::Model)
    return _link_or_invlink!!(vi, LinkSome(Set(vns)), model, Val(nothing))
end
function invlink!!(::DynamicTransformation, vi::VarInfo, vns, model::Model)
    return _link_or_invlink!!(vi, UnlinkSome(Set(vns)), model, Val(nothing))
end
function link!!(::DynamicTransformation, vi::VarInfo, model::Model)
    return _link_or_invlink!!(vi, LinkAll(), model, Val(true))
end
function invlink!!(::DynamicTransformation, vi::VarInfo, model::Model)
    return _link_or_invlink!!(vi, UnlinkAll(), model, Val(false))
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

# TODO(mhauru) I don't think this should return the internal values, but that's the current
# convention.
function values_as(vi::VarInfo, ::Type{Vector})
    return mapfoldl(
        pair -> tovec(DynamicPPL.get_internal_value(pair.second)),
        vcat,
        vi.values;
        init=Union{}[],
    )
end

function values_as(vi::VarInfo, ::Type{T}) where {T<:AbstractDict}
    return mapfoldl(identity, function (cumulant, pair)
        vn, tv = pair
        val = DynamicPPL.get_true_value(tv)
        return setindex!!(cumulant, val, vn)
    end, vi.values; init=T())
end

# TODO(mhauru) I really dislike this sort of conversion to Symbols, but it's the current
# interface provided by rand(::Model). We should change that to return a VarNamedTuple
# instead, and then this method (and any other values_as methods for NamedTuple) could be
# removed.
function values_as(vi::VarInfo, ::Type{NamedTuple})
    return mapfoldl(
        identity,
        function (cumulant, pair)
            vn, tv = pair
            val = DynamicPPL.get_true_value(tv)
            return setindex!!(cumulant, val, Symbol(vn))
        end,
        vi.values;
        init=NamedTuple(),
    )
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
