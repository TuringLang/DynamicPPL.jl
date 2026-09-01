const RAW_VALUE_ACCNAME = :RawValues

# TODO(mhauru) The deepcopy here is quite unfortunate. It is needed so that the model body
# can go mutating the object without that in turn mutating the value stored in the
# accumulator, which should be as it was at `~` time. Could there be a way around this?
_safe_copy(val) = deepcopy(val)
# collect is much faster than deepcopy on views, and for our purposes is the same (returns a
# copy of data that is not aliased to the original).
# See https://github.com/TuringLang/DynamicPPL.jl/pull/1350
_safe_copy(val::SubArray) = collect(val)

"""
    RawValueAccumulator(include_colon_eq::Bool) <: AbstractAccumulator

An accumulator that keeps track of the model parameters exactly as they are seen in the
model.

The parameter `include_colon_eq` controls whether variables on the LHS of `:=` are also
included in the accumulator's separate `colon_eq_values` collection.
"""
struct RawValueAccumulator{V<:VarNamedTuple,C<:VarNamedTuple} <: AbstractAccumulator
    include_colon_eq::Bool
    values::V
    colon_eq_values::C
end

function RawValueAccumulator(include_colon_eq::Bool)
    return RawValueAccumulator(include_colon_eq, VarNamedTuple(), VarNamedTuple())
end

accumulator_name(::RawValueAccumulator) = RAW_VALUE_ACCNAME
function Base.copy(acc::RawValueAccumulator)
    return update_values(acc, copy(acc.values), copy(acc.colon_eq_values))
end
accumulate_observe!!(acc::RawValueAccumulator, right, left, vn, template) = acc
function accumulate_assume!!(
    acc::RawValueAccumulator, val, tval, logjac, vn, dist, template
)
    new_val = _safe_copy(val)
    new_values = DynamicPPL.templated_setindex!!(acc.values, new_val, vn, template)
    return update_values(acc, new_values, acc.colon_eq_values)
end
function update_values(
    acc::RawValueAccumulator, values::VarNamedTuple, colon_eq_values::VarNamedTuple
)
    return RawValueAccumulator(acc.include_colon_eq, values, colon_eq_values)
end
function update_values(
    acc::RawValueAccumulator{VarNamedTuple,VarNamedTuple},
    values::VarNamedTuple,
    colon_eq_values::VarNamedTuple,
)
    return RawValueAccumulator{VarNamedTuple,VarNamedTuple}(
        acc.include_colon_eq, values, colon_eq_values
    )
end
function reset(acc::RawValueAccumulator)
    return update_values(acc, empty(acc.values), empty(acc.colon_eq_values))
end
split(acc::RawValueAccumulator) = reset(acc)
function combine(acc1::RawValueAccumulator, acc2::RawValueAccumulator)
    if acc1.include_colon_eq != acc2.include_colon_eq
        msg = "Cannot combine RawValueAccumulators with different `include_colon_eq` values"
        throw(ArgumentError(msg))
    end
    return update_values(
        acc1,
        merge(acc1.values, acc2.values),
        merge(acc1.colon_eq_values, acc2.colon_eq_values),
    )
end
function promote_for_threadsafe_eval(acc::RawValueAccumulator, ::Type)
    return RawValueAccumulator{VarNamedTuple,VarNamedTuple}(
        acc.include_colon_eq, acc.values, acc.colon_eq_values
    )
end

# We need a separate function for the colon-eq case since that function doesn't give us tval
# and logjac, and we don't want to have to pass in dummy values for those.
function store_colon_eq!!(acc::RawValueAccumulator, vn::VarName, val, template)
    new_val = _safe_copy(val)
    new_values = DynamicPPL.templated_setindex!!(acc.colon_eq_values, new_val, vn, template)
    return update_values(acc, acc.values, new_values)
end

#################################################################

# Debug version of RawValueAcc: it does the same thing as RawValueAcc, but additionally
# errors if a value is set twice. This is used in check_model. To catch cases where `:=`
# clashes with a tilde statement, we always include the colon-eq values in the accumulator.
struct DebugGetRawValues
    repeated_vns::Set{VarName}
end
is_extracting_colon_eq_values(g::DebugGetRawValues) = true
Base.copy(d::DebugGetRawValues) = DebugGetRawValues(copy(d.repeated_vns))
function DebugRawValueAccumulator()
    return VNTAccumulator{RAW_VALUE_ACCNAME}(DebugGetRawValues(Set{VarName}()))
end

# Split accumulators need independent sets because repeated names are mutable state.
function _zero(
    acc::Union{
        VNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
        TSVNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
    },
)
    new_acc = copy(acc)
    empty!(new_acc.f.repeated_vns)
    return update_values(new_acc, empty(new_acc.values))
end

# Unfortunately we have to overload accumulate_assume!! since we need to use the
# templated_setindex_no_overwrite!! function
function accumulate_assume!!(
    acc::Union{
        VNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
        TSVNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
    },
    val,
    tval,
    logjac,
    vn,
    dist,
    template,
)
    new_val = _safe_copy(val)
    # The exception catching is probably slow, but it's ok since it only happens inside
    # check_model.
    new_vnt = try
        DynamicPPL.VarNamedTuples.templated_setindex_no_overwrite!!(
            acc.values, new_val, vn, template
        )
    catch e
        # Don't error immediately, save it for later.
        if e isa DynamicPPL.VarNamedTuples.MustNotOverwriteError
            push!(acc.f.repeated_vns, e.target_vn)
            DynamicPPL.templated_setindex!!(acc.values, new_val, vn, template)
        else
            rethrow(e)
        end
    end
    return update_values(acc, new_vnt)
end

function store_colon_eq!!(
    acc::Union{
        VNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
        TSVNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
    },
    vn::VarName,
    val,
    template,
)
    new_val = _safe_copy(val)
    new_values = DynamicPPL.VarNamedTuples.templated_setindex_no_overwrite!!(
        acc.values, new_val, vn, template
    )
    return update_values(acc, new_values)
end

function DynamicPPL.combine(
    acc1::Union{
        VNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
        TSVNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
    },
    acc2::Union{
        VNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
        TSVNTAccumulator{RAW_VALUE_ACCNAME,DebugGetRawValues},
    },
)
    union!(acc1.f.repeated_vns, acc2.f.repeated_vns)

    new_values = acc1.values
    for (vn, val) in pairs(acc2.values)
        top_sym = DynamicPPL.AbstractPPL.getsym(vn)
        template_from_acc2_values = get(
            acc2.values.data, top_sym, DynamicPPL.VarNamedTuples.NoTemplate()
        )
        new_values = try
            DynamicPPL.VarNamedTuples.templated_setindex_no_overwrite!!(
                new_values, val, vn, template_from_acc2_values
            )
        catch e
            if e isa DynamicPPL.VarNamedTuples.MustNotOverwriteError
                push!(acc1.f.repeated_vns, e.target_vn)

                # Note: if `acc1` and `acc2` have different templates
                # `templated_setindex!!` uses the structure inside `acc1`'s values.
                DynamicPPL.templated_setindex!!(
                    new_values, val, vn, template_from_acc2_values
                )
            else
                rethrow(e)
            end
        end
    end
    return update_values(acc1, new_values)
end
