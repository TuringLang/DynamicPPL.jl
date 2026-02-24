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

"""
    DoNotAccumulate()

Sentinel value indicating that no accumulation should be performed for a given variable.
"""
struct DoNotAccumulate end

"""
    TSVNTAccumulator{AccName}(f::F, values::VarNamedTuple)

The same as `VNTAccumulator`, but with an abstract type parameter for the values.
This is required for threadsafe evaluation with VNT-based accumulators.
"""
struct TSVNTAccumulator{AccName,F} <: AbstractAccumulator
    f::F
    values::VarNamedTuple

    function TSVNTAccumulator{AccName}(f::F, values::VarNamedTuple) where {AccName,F}
        return new{AccName,F}(f, values)
    end
end
function promote_for_threadsafe_eval(
    acc::VNTAccumulator{AccName,F}, ::Type
) where {AccName,F}
    return TSVNTAccumulator{AccName}(acc.f, acc.values)
end

for acc_type in (:VNTAccumulator, :TSVNTAccumulator)
    @eval begin
        function Base.copy(acc::$acc_type{AccName}) where {AccName}
            return $acc_type{AccName}(acc.f, copy(acc.values))
        end
        accumulator_name(::$acc_type{AccName}) where {AccName} = AccName

        function update_values(
            acc::$acc_type{AccName}, new_values::VarNamedTuple
        ) where {AccName}
            return $acc_type{AccName}(acc.f, new_values)
        end

        function accumulate_assume!!(
            acc::$acc_type{AccName}, val, tval, logjac, vn, dist, template
        ) where {AccName}
            new_val = acc.f(val, tval, logjac, vn, dist)
            return if new_val isa DoNotAccumulate
                acc
            else
                new_values = DynamicPPL.templated_setindex!!(
                    acc.values, new_val, vn, template
                )
                update_values(acc, new_values)
            end
        end
        accumulate_observe!!(acc::$acc_type, right, left, vn, template) = acc

        function _zero(acc::$acc_type)
            return update_values(acc, empty(acc.values))
        end
        reset(acc::$acc_type) = _zero(acc)
        split(acc::$acc_type) = _zero(acc)
        function combine(acc1::$acc_type{AccName}, acc2::$acc_type{AccName}) where {AccName}
            if acc1.f != acc2.f
                throw(ArgumentError("Cannot combine $acc_type with different functions"))
            end
            return update_values(acc1, merge(acc1.values, acc2.values))
        end
    end
end
