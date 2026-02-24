const RAW_VALUE_ACCNAME = :RawValues

struct GetRawValues
    "A flag indicating whether variables on the LHS of := should also be included"
    include_colon_eq::Bool
end
# TODO(mhauru) The deepcopy here is quite unfortunate. It is needed so that the model body
# can go mutating the object without that in turn mutating the value stored in the
# accumulator, which should be as it was at `~` time. Could there be a way around this?
(g::GetRawValues)(val, tval, logjac, vn, dist) = deepcopy(val)
is_extracting_colon_eq_values(g::GetRawValues) = g.include_colon_eq

"""
    RawValueAccumulator(include_colon_eq)::Bool <: AbstractAccumulator

An accumulator that keeps tracks of the model parameters exactly as they are seen in the
model.

The parameter `include_colon_eq` controls whether variables on the LHS of `:=` are also
included in the accumulator. If `true`, then these variables will be included; if `false`,
they will not be included.
"""
function RawValueAccumulator(include_colon_eq::Bool)
    return VNTAccumulator{RAW_VALUE_ACCNAME}(GetRawValues(include_colon_eq))
end

# We need a separate function for the colon-eq case since that function doesn't give us tval
# and logjac, and we don't want to have to pass in dummy values for those.
function store_colon_eq!!(
    acc::Union{
        VNTAccumulator{RAW_VALUE_ACCNAME,GetRawValues},
        TSVNTAccumulator{RAW_VALUE_ACCNAME,GetRawValues},
    },
    vn::VarName,
    val,
    template,
)
    new_val = deepcopy(val)
    new_values = DynamicPPL.templated_setindex!!(acc.values, new_val, vn, template)
    return update_values(acc, new_values)
end

#################################################################

# Debug version of RawValueAcc: it does the same thing as RawValueAcc, but additionally
# errors if a value is set twice. This is used in check_model. To catch cases where `:=`
# clashes with a tilde statement, we always include the colon-eq values in the accumulator.
struct DebugGetRawValues
    repeated_vns::Set{VarName}
end
is_extracting_colon_eq_values(g::DebugGetRawValues) = true
function DebugRawValueAccumulator()
    return VNTAccumulator{RAW_VALUE_ACCNAME}(DebugGetRawValues(Set{VarName}()))
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
    new_val = deepcopy(val)
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
    new_val = deepcopy(val)
    new_values = DynamicPPL.VarNamedTuples.templated_setindex_no_overwrite!!(
        acc.values, new_val, vn, template
    )
    return update_values(acc, new_values)
end
