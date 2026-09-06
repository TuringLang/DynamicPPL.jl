"""
    VarInfo(accs...)
    VarInfo(accs::Tuple)
    VarInfo(accs::AccumulatorTuple)

Collect model-evaluation outputs in accumulators.

The default accumulators record log prior, log likelihood, and log Jacobian. Add a
`RawValueAccumulator` or `VectorValueAccumulator` to record parameter values.
Inputs, including the transform strategy, belong to `Context`; this type stores
no independent parameter values or transform state.
"""
struct VarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
VarInfo() = VarInfo(default_accumulators())
VarInfo(accs::NTuple{N,AbstractAccumulator}) where {N} = VarInfo(AccumulatorTuple(accs))
VarInfo(accs::Vararg{AbstractAccumulator}) = VarInfo(AccumulatorTuple(accs))

Base.copy(vi::VarInfo) = VarInfo(copy(vi.accs))
Base.:(==)(left::VarInfo, right::VarInfo) = left.accs == right.accs
Base.isequal(left::VarInfo, right::VarInfo) = isequal(left.accs, right.accs)
Base.hash(vi::VarInfo, h::UInt) = hash((VarInfo, vi.accs), h)
getaccs(vi::VarInfo) = vi.accs
setaccs!!(::VarInfo, accs::AccumulatorTuple) = VarInfo(accs)

function Base.show(io::IO, ::MIME"text/plain", vi::VarInfo)
    printstyled(io, "VarInfo"; bold=true)
    println(io)
    print(io, " └─ ")
    pretty_print(io, vi.accs, "    ")
    return nothing
end

"""
    VarInfo([rng::Random.AbstractRNG,] model::Model, init_strategy=InitFromPrior(), transform_strategy=UnlinkAll())

Evaluate `model` and collect vectorised parameter values and log densities.

To select different outputs, pass `VarInfo(accumulators...)` to `evaluate!!` with
an explicit `Context`.
"""
function VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    vi = VarInfo(VectorValueAccumulator(), default_accumulators()...)
    return last(evaluate!!(model, Context(rng, init_strategy, transform_strategy), vi))
end
function VarInfo(
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    return VarInfo(Random.default_rng(), model, init_strategy, transform_strategy)
end

"""
    get_vector_values(vi::AbstractVarInfo)

Extract vectorised `TransformedValue`s from the `VectorValueAccumulator` in `vi`.
Throw an error if that accumulator is absent.
"""
get_vector_values(vi::AbstractVarInfo) = getacc(vi, Val(VECTORVAL_ACCNAME)).values
get_values(vi::AbstractVarInfo) = get_vector_values(vi)

Base.keys(vi::VarInfo) = keys(get_vector_values(vi))
Base.haskey(vi::VarInfo, vn::VarName) = haskey(get_vector_values(vi), vn)
Base.length(vi::VarInfo) = length(get_vector_values(vi))
Base.values(vi::VarInfo) = values(get_vector_values(vi))
Base.isempty(vi::VarInfo) = isempty(get_vector_values(vi))
Base.empty(vi::VarInfo) = resetaccs!!(copy(vi))
BangBang.empty!!(vi::VarInfo) = resetaccs!!(vi)

get_transformed_value(vi::AbstractVarInfo, vn::VarName) = get_vector_values(vi)[vn]
function getindex_internal(vi::AbstractVarInfo, vn::VarName)
    return get_internal_value(get_transformed_value(vi, vn))
end
function internal_values_as_vector(vi::AbstractVarInfo)
    return internal_values_as_vector(get_vector_values(vi))
end
function is_transformed(vi::VarInfo, vn::VarName)
    return get_transform(get_transformed_value(vi, vn)) isa DynamicLink
end
function get_transform_strategy(vi::AbstractVarInfo)
    return infer_transform_strategy_from_values(get_vector_values(vi))
end

function _set_vector_values!!(vi::AbstractVarInfo, values::VarNamedTuple)
    acc = getacc(vi, Val(VECTORVAL_ACCNAME))
    return setacc!!(vi, update_values(acc, values))
end
"""
    setindex_internal!!(vi::VarInfo, val, vn::VarName)

Replace the vectorised value of `vn` in the value accumulator, preserving its transform.
"""
function setindex_internal!!(vi::VarInfo, val, vn::VarName)
    values = get_vector_values(vi)
    old = values[vn]
    return _set_vector_values!!(
        vi, setindex!!(values, TransformedValue(val, old.transform), vn)
    )
end

"""
    update_transform_status!!(vi::VarInfo, strategy::AbstractTransformStrategy, model::Model)

Re-evaluate recorded parameters with `strategy`, updating vectorised values and log-Jacobian.
Leave other accumulators unchanged.
"""
function update_transform_status!!(
    vi::VarInfo, strategy::AbstractTransformStrategy, model::Model
)
    ctx = Context(InitFromParams(get_vector_values(vi), nothing), strategy)
    outputs = VarInfo(VectorValueAccumulator(), LogJacobianAccumulator())
    _, outputs = evaluate!!(model, ctx, outputs)
    vi = _set_vector_values!!(vi, get_vector_values(outputs))
    return hasacc(vi, Val(:LogJacobian)) ? setlogjac!!(vi, getlogjac(outputs)) : vi
end
link!!(vi::VarInfo, model::Model) = update_transform_status!!(vi, LinkAll(), model)
invlink!!(vi::VarInfo, model::Model) = update_transform_status!!(vi, UnlinkAll(), model)
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

mutable struct VectorChunkIterator!{T<:AbstractVector}
    vec::T
    index::Int
end
function (vci::VectorChunkIterator!)(tv::TransformedValue{V,T}) where {V<:AbstractVector,T}
    len = length(tv.value)
    new_val = @view vci.vec[(vci.index):(vci.index + len - 1)]
    vci.index += len
    return TransformedValue(new_val, tv.transform)
end
function unflatten!!(vi::VarInfo, vec::AbstractVector)
    vci = VectorChunkIterator!(vec, 1)
    values = map_values!!(vci, get_vector_values(vi))
    expected_len = vci.index - 1
    length(vec) == expected_len || throw(
        DimensionMismatch(
            "expected a vector of length $(expected_len), but got length $(length(vec))"
        ),
    )
    return _set_vector_values!!(vi, values)
end

function subset(vi::VarInfo, vns)
    return _set_vector_values!!(copy(vi), subset(get_vector_values(vi), vns))
end
function Base.merge(left::VarInfo, right::VarInfo)
    values = merge(get_vector_values(left), get_vector_values(right))
    return _set_vector_values!!(copy(right), values)
end
