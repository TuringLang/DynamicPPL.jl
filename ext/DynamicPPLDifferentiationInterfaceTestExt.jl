module DynamicPPLDifferentiationInterfaceTestExt

using DynamicPPL:
    DynamicPPL,
    ADTypes,
    LogDensityProblems,
    Model,
    AbstractVarInfo,
    VarInfo,
    LogDensityFunction
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT

"""
    REFERENCE_ADTYPE

Reference AD backend to use for comparison. In this case, ForwardDiff.jl, since
it's the default AD backend used in Turing.jl.
"""
const REFERENCE_ADTYPE = ADTypes.AutoForwardDiff()

"""
    DynamicPPL.TestUtils.AD.make_scenario(
        model::Model,
        adtype::ADTypes.AbstractADType,
        varinfo::AbstractVarInfo=VarInfo(model),
        params::Vector{<:Real}=varinfo[:],
        reference_adtype::ADTypes.AbstractADType=REFERENCE_ADTYPE,
        expected_value_and_grad::Union{Nothing,Tuple{Real,Vector{<:Real}}}=nothing,
    )

Construct a DifferentiationInterfaceTest.Scenario for the given `model` and `adtype`.

More docs to follow.
"""
function DynamicPPL.TestUtils.AD.make_scenario(
    model::Model,
    adtype::ADTypes.AbstractADType;
    varinfo::AbstractVarInfo=VarInfo(model),
    params::Vector{<:Real}=varinfo[:],
    reference_adtype::ADTypes.AbstractADType=REFERENCE_ADTYPE,
    expected_grad::Union{Nothing,Vector{<:Real}}=nothing,
)
    params = map(identity, params)
    context = DynamicPPL.DefaultContext()
    adtype = DynamicPPL.tweak_adtype(adtype, model, varinfo, context)
    if DynamicPPL.use_closure(adtype)
        f = x -> DynamicPPL.logdensity_at(x, model, varinfo, context)
        di_contexts = ()
    else
        f = DynamicPPL.logdensity_at
        di_contexts = (DI.Constant(model), DI.Constant(varinfo), DI.Constant(context))
    end

    # Calculate ground truth to compare against
    grad_true = if expected_grad === nothing
        ldf_reference = LogDensityFunction(model; adtype=reference_adtype)
        LogDensityProblems.logdensity_and_gradient(ldf_reference, params)[2]
    else
        expected_grad
    end

    return DIT.Scenario{:gradient,:out}(
        f, params; contexts=di_contexts, res1=grad_true, name="$(model.f)"
    )
end

end
