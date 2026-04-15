module DynamicPPLEnzymeExt

using DynamicPPL: ADTypes, DynamicPPL, logdensity_at
using Enzyme: Enzyme

_enzyme_gradient_mode(::ADTypes.AutoEnzyme{Nothing}) = Enzyme.ReverseWithPrimal
function _enzyme_gradient_mode(adtype::ADTypes.AutoEnzyme)
    return Enzyme.EnzymeCore.WithPrimal(adtype.mode)
end

_cache_enzyme_gradient(::ADTypes.AutoEnzyme{Nothing}) = true
_cache_enzyme_gradient(::ADTypes.AutoEnzyme{<:Enzyme.EnzymeCore.ReverseMode}) = true
_cache_enzyme_gradient(::ADTypes.AutoEnzyme) = false

function _extract_value_and_gradient(result::NamedTuple{(:derivs, :val)})
    return result.val, first(result.derivs)
end

function DynamicPPL._prepare_gradient(
    adtype::ADTypes.AutoEnzyme,
    x::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    return _cache_enzyme_gradient(adtype) ? (; dx=similar(x)) : nothing
end

function DynamicPPL._value_and_gradient(
    adtype::ADTypes.AutoEnzyme,
    prep::NamedTuple{(:dx,)},
    params::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    isempty(params) && return logdensity_at(
        params, model, getlogdensity, varname_ranges, transform_strategy, accs
    ),
    copy(params)
    dx = prep.dx
    fill!(dx, zero(eltype(dx)))
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

function DynamicPPL._value_and_gradient(
    adtype::ADTypes.AutoEnzyme,
    ::Nothing,
    params::AbstractVector{<:Real},
    model::DynamicPPL.Model,
    getlogdensity::Any,
    varname_ranges::DynamicPPL.VarNamedTuple,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
    accs::DynamicPPL.AccumulatorTuple,
)
    isempty(params) && return logdensity_at(
        params, model, getlogdensity, varname_ranges, transform_strategy, accs
    ),
    copy(params)
    # Pass the plain function plus Const arguments; Enzyme is brittle with closure-like callables.
    val, dx = _extract_value_and_gradient(
        Enzyme.gradient(
            _enzyme_gradient_mode(adtype),
            logdensity_at,
            params,
            Enzyme.Const(model),
            Enzyme.Const(getlogdensity),
            Enzyme.Const(varname_ranges),
            Enzyme.Const(transform_strategy),
            Enzyme.Const(accs),
        ),
    )
    return val, copy(dx)
end

end # module
