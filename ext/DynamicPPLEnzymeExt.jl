module DynamicPPLEnzymeExt

using DynamicPPL: ADTypes, DynamicPPL
using Enzyme: Enzyme

_enzyme_gradient_mode(::ADTypes.AutoEnzyme{Nothing}) = Enzyme.ReverseWithPrimal
function _enzyme_gradient_mode(adtype::ADTypes.AutoEnzyme)
    return Enzyme.EnzymeCore.set_runtime_activity(Enzyme.ReverseWithPrimal, adtype.mode)
end

function DynamicPPL._prepare_gradient(
    ::ADTypes.AutoEnzyme,
    x::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    return (; dx=similar(x))
end

function DynamicPPL._value_and_gradient(
    adtype::ADTypes.AutoEnzyme,
    prep,
    params::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    f = DynamicPPL.LogDensityAt(
        model, getlogdensity, varname_ranges, transform_strategy, accs
    )
    dx = prep.dx
    fill!(dx, zero(eltype(dx)))
    # Const(f): LogDensityAt is not being differentiated; without Const, Enzyme errors
    # because it cannot prove the function argument is readonly.
    # We always use reverse mode to obtain the full gradient in one pass, but preserve
    # runtime-activity settings from `adtype.mode` when they were requested.
    # autodiff(ReverseWithPrimal, ...) returns ((), val); dx is mutated in-place.
    _, val = Enzyme.autodiff(
        _enzyme_gradient_mode(adtype),
        Enzyme.Const(f),
        Enzyme.Active,
        Enzyme.Duplicated(params, dx),
    )
    return val, copy(dx)
end

end # module
