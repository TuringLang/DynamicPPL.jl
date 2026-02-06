const RAW_VALUE_ACCNAME = :RawValues

struct GetRawValues
    "A flag indicating whether variables on the LHS of := should also be included"
    include_colon_eq::Bool
end
# TODO(mhauru) The deepcopy here is quite unfortunate. It is needed so that the model body
# can go mutating the object without that reactively affecting the value in the accumulator,
# which should be as it was at `~` time. Could there be a way around this?
(g::GetRawValues)(val, tval, logjac, vn, dist) = deepcopy(val)

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
    acc::VNTAccumulator{RAW_VALUE_ACCNAME}, vn::VarName, val, template
)
    new_val = deepcopy(val)
    new_values = DynamicPPL.templated_setindex!!(acc.values, new_val, vn, template)
    return VNTAccumulator{RAW_VALUE_ACCNAME}(acc.f, new_values)
end
