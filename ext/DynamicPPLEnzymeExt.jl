module DynamicPPLEnzymeExt

using DynamicPPL: ADTypes, DynamicPPL, logdensity_at
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
    dx = prep.dx
    fill!(dx, zero(eltype(dx)))
    # Pass the plain function plus Const arguments; Enzyme is brittle with closure-like callables.
    _, val = Enzyme.autodiff(
        _enzyme_gradient_mode(adtype),
        logdensity_at,
        Enzyme.Active,
        Enzyme.Duplicated(params, dx),
        Enzyme.Const(model),
        Enzyme.Const(getlogdensity),
        Enzyme.Const(varname_ranges),
        Enzyme.Const(transform_strategy),
        Enzyme.Const(accs),
    )
    return val, copy(dx)
end

end # module
