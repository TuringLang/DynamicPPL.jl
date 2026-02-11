struct VectorParamAccumulator{T,V<:VarNamedTuple} <: DynamicPPL.AbstractAccumulator
    vals::Vector{T}
    set_indices::Vector{Bool}
    vn_ranges::V
end

"""
    VectorParamAccumulator(ldf::DynamicPPL.LogDensityFunction)

An accumulator for collecting vectorised parameters from a model evaluation. The resulting
vector will be consistent with the `LogDensityFunction` used to construct the accumulator,
both in terms of whether parameters are transformed or not, as well as the order of the
parameters in the vector.

This accumulator allows you to re-evaluate a `LogDensityFunction` with a different
initialisation strategy and collect the vectorised parameters corresponding to that
strategy.
"""
function VectorParamAccumulator(ldf::LogDensityFunction)
    et = eltype(_get_input_vector_type(ldf))
    dim = ldf._dim
    vals = Vector{et}(undef, dim)
    set_indices = falses(dim)
    vn_ranges = ldf._varname_ranges
    return VectorParamAccumulator{et,typeof(vn_ranges)}(vals, set_indices, vn_ranges)
end

const VECTOR_ACC_NAME = :VectorParamAccumulator
DynamicPPL.accumulator_name(::Type{<:VectorParamAccumulator}) = VECTOR_ACC_NAME

function DynamicPPL.accumulate_observe!!(
    acc::VectorParamAccumulator, ::Distribution, val, ::Union{VarName,Nothing}
)
    return acc
end

function DynamicPPL.accumulate_assume!!(
    acc::VectorParamAccumulator,
    val,
    tval::AbstractTransformedValue,
    logjac,
    vn::VarName,
    dist::Distribution,
    ::Any,
)
    ral = acc.vn_ranges[vn]
    # sometimes you might get UntransformedValue... _get_vector_tval is in
    # src/accumulators/vector_values.jl.
    vectorised_tval = _get_vector_tval(val, tval, logjac, vn, dist)
    return _update_acc(acc, vectorised_tval, ral, vn)
end

function _update_acc(
    acc::VectorParamAccumulator,
    tval::Union{LinkedVectorValue,VectorValue},
    ral::RangeAndLinked,
    vn::VarName,
)
    if (
        (ral.is_linked && tval isa VectorValue) ||
        (!ral.is_linked && tval isa LinkedVectorValue)
    )
        throw(
            ArgumentError(
                "The LogDensityFunction specifies that `$vn` should be $(ral.is_linked ? "linked" : "unlinked"), but the vector values contain a $(tval isa LinkedVectorValue ? "linked" : "unlinked") value for that variable.",
            ),
        )
    end

    vec_val = DynamicPPL.get_internal_value(tval)
    len = length(vec_val)
    expected_len = length(ral.range)
    if len != expected_len
        throw(
            ArgumentError(
                "The length of the vector value provided for `$vn` is $len, but the LogDensityFunction expects it to be $expected_len based on the ranges that were extracted when the LogDensityFunction was constructed.",
            ),
        )
    end

    if any(acc.set_indices[ral.range])
        throw(
            ArgumentError(
                "Setting to the same indices in the output vector more than once. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
            ),
        )
    end
    Accessors.@set acc.vals = BangBang.setindex!!(acc.vals, vec_val, ral.range)
    acc.set_indices[ral.range] .= true

    return acc
end

function DynamicPPL.reset(acc::VectorParamAccumulator)
    acc.set_indices .= false
    return acc
end

function Base.copy(acc::VectorParamAccumulator)
    return VectorParamAccumulator(copy(acc.vals), copy(acc.set_indices), acc.vn_ranges)
end

function DynamicPPL.split(acc::VectorParamAccumulator)
    new_acc = VectorParamAccumulator(copy(acc.vals), copy(acc.set_indices), acc.vn_ranges)
    new_acc.set_indices .= false
    return new_acc
end

function DynamicPPL.combine(acc1::VectorParamAccumulator, acc2::VectorParamAccumulator)
    if acc1.vn_ranges != acc2.vn_ranges
        throw(
            ArgumentError("Cannot combine VectorParamAccumulators with different vn_ranges")
        )
    end
    if any(acc1.set_indices .& acc2.set_indices)
        throw(
            ArgumentError(
                "Cannot combine VectorParamAccumulators that have overlapping set indices. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
            ),
        )
    end
    Accessors.@set acc1.vals = BangBang.setindex!!(
        acc1.vals, acc2.vals[acc2.set_indices], acc2.set_indices
    )
    acc1.set_indices .= acc1.set_indices .| acc2.set_indices
    return acc1
end

"""
    get_vector_params(vi::DynamicPPL.AbstractVarInfo)

Extract the vectorised parameters from the `VectorParamAccumulator` stored in the
`AbstractVarInfo`. If there is no `VectorParamAccumulator` in the `AbstractVarInfo`, or if
some indices in the output vector were not set, an error will be thrown.
"""
function get_vector_params(vi::DynamicPPL.AbstractVarInfo)
    acc = DynamicPPL.getacc(vi, Val(VECTOR_ACC_NAME))
    if !all(acc.set_indices)
        throw(
            ArgumentError(
                "Some indices in the output vector were not set. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
            ),
        )
    end
    return acc.vals
end
