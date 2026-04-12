module DynamicPPLEnzymeExt

using DynamicPPL: ADTypes, DynamicPPL
using Enzyme: Enzyme

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
    ::ADTypes.AutoEnzyme,
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
    # autodiff(ReverseWithPrimal, ...) returns ((), val); dx is mutated in-place.
    _, val = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Enzyme.Const(f),
        Enzyme.Active,
        Enzyme.Duplicated(params, dx),
    )
    return val, copy(dx)
end

end # module
