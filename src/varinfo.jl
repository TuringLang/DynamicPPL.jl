"""
    VarInfo{
        Tfm<:AbstractTransformStrategy,
        T<:VarNamedTuple,
        Accs<:AccumulatorTuple
    } <: AbstractVarInfo

The default implementation of `AbstractVarInfo`, storing variable values and accumulators.

A `VarInfo`, `vi`, is quite a thin wrapper around

 - `vi.values`: a `VarNamedTuple` storing the variable values, and
 - `vi.accs`: a tuple of accumulators.

The only really noteworthy thing about it is that `vi.values` specifically stores the values
of variables as `[TransformedValue{<:AbstractVector}](@ref TransformedValue)`.

That is, regardless of what the value of a variable is in the original distribution, the
VarInfo stores a *vectorised* version of the value. It is not particularly concerned about
whether the variable is linked or not: you can mix unlinked variables with linked variables
in a `VarInfo`.

!!! note
    This functionality is identical to that in [`VectorValueAccumulator`](@ref), and going
    forward we recommend using that instead of `VarInfo`.

On top of that, `VarInfo` also stores a transform strategy, which reflects the linked status
of variables inside the `VarInfo`. For example, a `VarInfo{LinkAll}` should contain only
`TransformedValue{T,LinkAll}`s in its `values` field. This unfortunately leads to redundancy
of information, but is necessary for type stability, since that allows us to have
compile-time knowledge of what transformations are applied.

Because the job of `VarInfo` is to store transformed values, there is no generic
`setindex!!` implementation on `VarInfo` itself. Instead, all storage must go via
[`setindex_with_dist!!`](@ref), which takes care of storing the value in the correct
transformed form. This in turn means that the distribution on the right-hand side of a
tilde-statement must be available when modifying a VarInfo.

Furthermore, since no untransformed (raw) values are stored in `VarInfo`, there is no
generic `getindex` implementation that returns raw values. If you need this functionality,
you should make sure that `vi.accs` contains a `RawValueAccumulator` and use that to get the
raw values. To directly get access to the internal vectorised values in `vi.values`, you can
use [`getindex_internal`](@ref), [`setindex_internal!!`](@ref), and [`unflatten!!`](@ref).

For more details on the internal storage, see documentation of [`TransformedValue`](@ref)
and [`VarNamedTuple`](@ref).

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
       init_strategy::AbstractInitStrategy=InitFromPrior(),
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
    new_tv = TransformedValue(val, old_tv.transform)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VarInfo(vi.transform_strategy, new_values, vi.accs)
end

"""
    update_transform_strategy(
        tfm_strategy::AbstractTransformStrategy,
        vi_is_empty::Bool,
        new_vn::VarName,
        new_vn_transform::AbstractTransform
    )

Given an old transform strategy `tfm_strategy`, and the transformation of a new variable
`new_vn` to be added to a `VarInfo` which has that transform strategy, return an updated
transform strategy that accounts for the addition of `new_vn`.
"""
function update_transform_strategy(::LinkAll, ::Bool, ::VarName, ::DynamicLink)
    return LinkAll()
end
function update_transform_strategy(::LinkAll, vi_is_empty::Bool, new_vn::VarName, ::Unlink)
    return vi_is_empty ? UnlinkAll() : UnlinkSome(Set([new_vn]), LinkAll())
end
function update_transform_strategy(::UnlinkAll, ::Bool, ::VarName, ::Unlink)
    return UnlinkAll()
end
function update_transform_strategy(
    ::UnlinkAll, vi_is_empty::Bool, new_vn::VarName, ::DynamicLink
)
    return vi_is_empty ? LinkAll() : LinkSome(Set([new_vn]), UnlinkAll())
end
function update_transform_strategy(
    ::AbstractTransformStrategy, ::Bool, ::VarName, ::FixedTransform
)
    # TODO!
    return error("not implemented")
end

"""
    setindex_with_dist!!(
        vi::VarInfo,
        tval::TransformedValue{<:AbstractVector{<:Real},<:Any},
        dist::Distribution,
        vn::VarName,
        template::Any,
    )

Store a transformed value that has already been vectorised. This might include dynamically
transformed variables (which have `tval.transform` as a `DynamicLink` or `Unlink`), or
statically transformed variables (which have `tval.transform` as a `FixedTransform`).
However, in either case, it is mandatory that `tval.value` is a vector.

Note that this will cause the linked status of `vi` to update according to what `tval` is.
That means that whether or not a variable is considered to be 'linked' is determined by
`tval` rather than the previous status of `vi`.
"""
function setindex_with_dist!!(
    vi::VarInfo, tval::TransformedValue{T,V}, ::Distribution, vn::VarName, template::Any
) where {T<:AbstractVector{<:Real},V}
    new_transform_strategy = update_transform_strategy(
        vi.transform_strategy, isempty(vi), vn, tval.transform
    )
    return VarInfo(
        new_transform_strategy, templated_setindex!!(vi.values, tval, vn, template), vi.accs
    )
end

"""
    setindex_with_dist!!(
        vi::VarInfo,
        utval::TransformedValue{<:Any,NoTransform},
        dist::Distribution,
        vn::VarName,
        template::Any
    )

Vectorise `utval` and store it. (Note that if `setindex_with_dist!!` receives an
untransformed value, the variable is always considered unlinked, since if it were to be
linked, `apply_transform_strategy` will already have done so.)
"""
function setindex_with_dist!!(
    vi::VarInfo,
    tval::TransformedValue{V,NoTransform},
    dist::Distribution,
    vn::VarName,
    template,
) where {V}
    raw_value = DynamicPPL.get_internal_value(tval)
    vectorised_value = Bijectors.VectorBijectors.to_vec(dist)(raw_value)
    tval = TransformedValue(vectorised_value, Unlink())
    return setindex_with_dist!!(vi, tval, dist, vn, template)
end
function setindex_with_dist!!(
    vi::VarInfo,
    tval::TransformedValue{V,NoTransform},
    dist::Distribution,
    vn::VarName,
    template,
) where {V<:AbstractVector{<:Real}}
    # This method is needed for resolving ambiguities. It does the same thing as
    # above, but skipping the vectorisation step, since to_vec(dist) for a vector
    # is always identity.
    tval = TransformedValue(DynamicPPL.get_internal_value(tval), Unlink())
    return setindex_with_dist!!(vi, tval, dist, vn, template)
end

"""
    set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)

If `linked`, set the variable `vn` in `vi` to be linked (i.e., change its stored transform
to be `DynamicLink()`). Otherwise, set it to be unlinked (i.e., change its stored transform
to be `Unlink()`). This will also update the transform strategy of `vi` accordingly.

!!! warning
    Note that this function is potentially unsafe as it does not change the value of the
    variable!
"""
function set_transformed!!(vi::VarInfo, linked::Bool, vn::VarName)
    # TODO!(penelopeysm): Why do we still need this?
    old_tv = getindex(vi.values, vn)
    new_transform = linked ? DynamicLink() : Unlink()
    new_tv = TransformedValue(old_tv.value, new_transform)
    new_values = setindex!!(vi.values, new_tv, vn)
    new_transform_strategy = update_transform_strategy(
        vi.transform_strategy, isempty(vi), vn, new_transform
    )
    return VarInfo(new_transform_strategy, new_values, vi.accs)
end

function set_transformed!!(vi::VarInfo, linked::Bool)
    tfm = linked ? DynamicLink() : Unlink()
    new_values = map_values!!(vi.values) do tv
        TransformedValue(tv.value, tfm)
    end
    new_transform_strategy = linked ? LinkAll() : UnlinkAll()
    return VarInfo(new_transform_strategy, new_values, vi.accs)
end

"""
    getindex_internal(vi::VarInfo, vn::VarName)

Get the internal (vectorised) value of variable `vn` in `vi`.
"""
getindex_internal(vi::VarInfo, vn::VarName) = get_internal_value(getindex(vi.values, vn))

"""
    get_transformed_value(vi::VarInfo, vn::VarName)

Get the entire `TransformedValue` for variable `vn` in `vi`.
"""
get_transformed_value(vi::VarInfo, vn::VarName) = getindex(vi.values, vn)

function is_transformed(vi::VarInfo, vn::VarName)
    return if vi.transform_strategy isa LinkAll
        true
    elseif vi.transform_strategy isa UnlinkAll
        false
    else
        get_transformed_value(vi, vn).transform isa DynamicLink
    end
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
    DynamicPPL.update_transform_status!!(
        orig_vi::VarInfo, transform_strategy::AbstractTransformStrategy, model::Model
    )

Given an original `VarInfo` `orig_vi`, update the link status of its variables according to
the new `transform_strategy`.
"""
function update_transform_status!!(
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

# These are mostly convenience functions
function link!!(vi::VarInfo, vns, model::Model)
    return update_transform_status!!(
        vi, LinkSome(Set(vns), get_transform_strategy(vi)), model
    )
end
function invlink!!(vi::VarInfo, vns, model::Model)
    return update_transform_status!!(
        vi, UnlinkSome(Set(vns), get_transform_strategy(vi)), model
    )
end
function link!!(vi::VarInfo, model::Model)
    return update_transform_status!!(vi, LinkAll(), model)
end
function invlink!!(vi::VarInfo, model::Model)
    return update_transform_status!!(vi, UnlinkAll(), model)
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
function (vci::VectorChunkIterator!)(
    tv::TransformedValue{V,T}
) where {V<:AbstractVector{<:Real},T}
    old_val = tv.value
    len = length(old_val)
    new_val = @view vci.vec[(vci.index):(vci.index + len - 1)]
    vci.index += len
    return TransformedValue(new_val, tv.transform)
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
    # Some shortcircuits to maximise type stability
        if varinfo_left.transform_strategy isa LinkAll &&
            varinfo_right.transform_strategy isa LinkAll
            LinkAll()
        elseif varinfo_left.transform_strategy isa UnlinkAll &&
            varinfo_right.transform_strategy isa UnlinkAll
            UnlinkAll()
        else
            infer_transform_strategy_from_values(new_values)
        end
    return VarInfo(new_transform_strategy, new_values, new_accs)
end
