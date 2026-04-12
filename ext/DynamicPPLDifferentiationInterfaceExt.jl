module DynamicPPLDifferentiationInterfaceExt

import DifferentiationInterface as DI
using DynamicPPL:
    DynamicPPL,
    AccumulatorTuple,
    LogDensityAt,
    Model,
    VarNamedTuple,
    AbstractTransformStrategy,
    _use_closure,
    logdensity_at
using ADTypes: ADTypes

function DynamicPPL._prepare_gradient(
    adtype::ADTypes.AbstractADType,
    x::AbstractVector{<:Real},
    model::Model,
    getlogdensity::Any,
    varname_ranges::VarNamedTuple,
    transform_strategy::AbstractTransformStrategy,
    accs::AccumulatorTuple,
)
    args = (model, getlogdensity, varname_ranges, transform_strategy, accs)
    return if _use_closure(adtype)
        DI.prepare_gradient(LogDensityAt(args...), adtype, x)
    else
        DI.prepare_gradient(logdensity_at, adtype, x, map(DI.Constant, args)...)
    end
end

function DynamicPPL._value_and_gradient(
    adtype::ADTypes.AbstractADType,
    prep,
    params::AbstractVector{<:Real},
    model::Model,
    getlogdensity::Any,
    varname_ranges::VarNamedTuple,
    transform_strategy::AbstractTransformStrategy,
    accs::AccumulatorTuple,
)
    return if _use_closure(adtype)
        DI.value_and_gradient(
            LogDensityAt(model, getlogdensity, varname_ranges, transform_strategy, accs),
            prep,
            adtype,
            params,
        )
    else
        DI.value_and_gradient(
            logdensity_at,
            prep,
            adtype,
            params,
            DI.Constant(model),
            DI.Constant(getlogdensity),
            DI.Constant(varname_ranges),
            DI.Constant(transform_strategy),
            DI.Constant(accs),
        )
    end
end

end # module
