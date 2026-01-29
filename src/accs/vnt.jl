"""
    VNTAccumulator{AccName}(f::F, values::VarNamedTuple=VarNamedTuple()) where {AccName,F}

A generic accumulator that applies a function `f` to values seen during model execution
and stores the results in a `VarNamedTuple`.

`AccName` is the name of the accumulator, and is exposed to allow users to define and use
multiple forms of `VNTAccumulator` within the same set of accumulators. In theory, each
`VNTAccumulator` with the same function `f` should use the same accumulator name. This is
not enforced.

The function `f` should have the signature:

    f(val, tval, logjac, vn, dist) -> value_to_store

where `val`, `tval`, `logjac`, `vn`, and `dist` have their usual meanings in
accumulate_assume!! (see its docstring for more details). If a value does not need to
be accumulated, this can be signalled by returning `DoNotAccumulate()` from `f`.
"""
struct VNTAccumulator{AccName,F,VNT<:VarNamedTuple} <: AbstractAccumulator
    f::F
    values::VNT
end
function VNTAccumulator{AccName}(
    f::F, values::VarNamedTuple=VarNamedTuple()
) where {AccName,F}
    return VNTAccumulator{AccName,F,typeof(values)}(f, values)
end

function Base.copy(acc::VNTAccumulator{AccName}) where {AccName}
    return VNTAccumulator{AccName}(acc.f, copy(acc.values))
end

accumulator_name(::VNTAccumulator{AccName}) where {AccName} = AccName

function _zero(acc::VNTAccumulator{AccName}) where {AccName}
    return VNTAccumulator{AccName}(acc.f, empty(acc.values))
end
reset(acc::VNTAccumulator{AccName}) where {AccName} = _zero(acc)
split(acc::VNTAccumulator{AccName}) where {AccName} = _zero(acc)
function combine(
    acc1::VNTAccumulator{AccName}, acc2::VNTAccumulator{AccName}
) where {AccName}
    if acc1.f != acc2.f
        throw(ArgumentError("Cannot combine VNTAccumulators with different functions"))
    end
    return VNTAccumulator{AccName}(acc2.f, merge(acc1.values, acc2.values))
end

"""
    DoNotAccumulate()

Sentinel value indicating that no accumulation should be performed for a given variable.
"""
struct DoNotAccumulate end

function accumulate_assume!!(
    acc::VNTAccumulator{AccName}, val, tval, logjac, vn, dist, template
) where {AccName}
    new_val = acc.f(val, tval, logjac, vn, dist)
    return if new_val isa DoNotAccumulate
        acc
    else
        new_values = DynamicPPL.templated_setindex!!(acc.values, new_val, vn, template)
        VNTAccumulator{AccName}(acc.f, new_values)
    end
end
accumulate_observe!!(acc::VNTAccumulator, right, left, vn) = acc
