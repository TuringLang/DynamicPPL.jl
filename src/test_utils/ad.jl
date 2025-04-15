module AD

using ADTypes: AbstractADType, AutoForwardDiff
using Chairmarks: @be
import DifferentiationInterface as DI
using DocStringExtensions
using DynamicPPL: Model, LogDensityFunction, VarInfo, AbstractVarInfo, link
using LogDensityProblems: logdensity, logdensity_and_gradient
using Random: Random, Xoshiro
using Statistics: median
using Test: @test

export ADResult, run_ad

# This function needed to work around the fact that different backends can
# return different AbstractArrays for the gradient. See
# https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/754 for more
# context.
_to_vec_f64(x::AbstractArray) = x isa Vector{Float64} ? x : collect(Float64, x)

"""
    REFERENCE_ADTYPE

Reference AD backend to use for comparison. In this case, ForwardDiff.jl, since
it's the default AD backend used in Turing.jl.
"""
const REFERENCE_ADTYPE = AutoForwardDiff()

"""
    ADResult

Data structure to store the results of the AD correctness test.
"""
struct ADResult
    "The DynamicPPL model that was tested"
    model::Model
    "The VarInfo that was used"
    varinfo::AbstractVarInfo
    "The values at which the model was evaluated"
    params::Vector{<:Real}
    "The AD backend that was tested"
    adtype::AbstractADType
    "The absolute tolerance for the value of logp"
    value_atol::Real
    "The absolute tolerance for the gradient of logp"
    grad_atol::Real
    "The expected value of logp"
    value_expected::Union{Nothing,Float64}
    "The expected gradient of logp"
    grad_expected::Union{Nothing,Vector{Float64}}
    "The value of logp (calculated using `adtype`)"
    value_actual::Union{Nothing,Real}
    "The gradient of logp (calculated using `adtype`)"
    grad_actual::Union{Nothing,Vector{Float64}}
    "If benchmarking was requested, the time taken by the AD backend to calculate the gradient of logp, divided by the time taken to evaluate logp itself"
    time_vs_primal::Union{Nothing,Float64}
end

"""
    run_ad(
        model::Model,
        adtype::ADTypes.AbstractADType;
        test=true,
        benchmark=false,
        value_atol=1e-6,
        grad_atol=1e-6,
        linked::Bool=true,
        varinfo::AbstractVarInfo=VarInfo(model),
        params::Union{Nothing,Vector{<:Real}}=nothing,
        reference_adtype::ADTypes.AbstractADType=REFERENCE_ADTYPE,
        expected_value_and_grad::Union{Nothing,Tuple{Real,Vector{<:Real}}}=nothing,
        verbose=true,
    )::ADResult

Test the correctness and/or benchmark the AD backend `adtype` for the model
`model`.

Whether to test and benchmark is controlled by the `test` and `benchmark`
keyword arguments. By default, `test` is `true` and `benchmark` is `false`.

Returns an [`ADResult`](@ref) object, which contains the results of the
test and/or benchmark.

Note that to run AD successfully you will need to import the AD backend itself.
For example, to test with `AutoReverseDiff()` you will need to run `import
ReverseDiff`.

There are two positional arguments, which absolutely must be provided:

1. `model` - The model being tested.
2. `adtype` - The AD backend being tested.

Everything else is optional, and can be categorised into several groups:

1. _How to specify the VarInfo._

   DynamicPPL contains several different types of VarInfo objects which change
   the way model evaluation occurs. If you want to use a specific type of
   VarInfo, pass it as the `varinfo` argument. Otherwise, it will default to
   using a `TypedVarInfo` generated from the model.

   It will also perform _linking_, that is, the parameters in the VarInfo will
   be transformed to unconstrained Euclidean space if they aren't already in
   that space. Note that the act of linking may change the length of the
   parameters. To disable linking, set `linked=false`.

2. _How to specify the parameters._

   For maximum control over this, generate a vector of parameters yourself and
   pass this as the `params` argument. If you don't specify this, it will be
   taken from the contents of the VarInfo.

   Note that if the VarInfo is not specified (and thus automatically generated)
   the parameters in it will have been sampled from the prior of the model. If
   you want to seed the parameter generation, the easiest way is to pass a
   `rng` argument to the VarInfo constructor (i.e. do `VarInfo(rng, model)`).

   Finally, note that these only reflect the parameters used for _evaluating_
   the gradient. If you also want to control the parameters used for
   _preparing_ the gradient, then you need to manually set these parameters in
   the VarInfo object, for example using `vi = DynamicPPL.unflatten(vi,
   prep_params)`. You could then evaluate the gradient at a different set of
   parameters using the `params` keyword argument.

3. _How to specify the results to compare against._ (Only if `test=true`.)

   Once logp and its gradient has been calculated with the specified `adtype`,
   it must be tested for correctness.

   This can be done either by specifying `reference_adtype`, in which case logp
   and its gradient will also be calculated with this reference in order to
   obtain the ground truth; or by using `expected_value_and_grad`, which is a
   tuple of `(logp, gradient)` that the calculated values must match. The
   latter is useful if you are testing multiple AD backends and want to avoid
   recalculating the ground truth multiple times.

   The default reference backend is ForwardDiff. If none of these parameters are
   specified, ForwardDiff will be used to calculate the ground truth.

4. _How to specify the tolerances._ (Only if `test=true`.)

   The tolerances for the value and gradient can be set using `value_atol` and
   `grad_atol`. These default to 1e-6.

5. _Whether to output extra logging information._

   By default, this function prints messages when it runs. To silence it, set
   `verbose=false`.
"""
function run_ad(
    model::Model,
    adtype::AbstractADType;
    test=true,
    benchmark=false,
    value_atol=1e-6,
    grad_atol=1e-6,
    linked::Bool=true,
    varinfo::AbstractVarInfo=VarInfo(model),
    params::Union{Nothing,Vector{<:Real}}=nothing,
    reference_adtype::AbstractADType=REFERENCE_ADTYPE,
    expected_value_and_grad::Union{Nothing,Tuple{Real,Vector{<:Real}}}=nothing,
    verbose=true,
)::ADResult
    if linked
        varinfo = link(varinfo, model)
    end
    if isnothing(params)
        params = varinfo[:]
    end
    params = map(identity, params)

    verbose && @info "Running AD on $(model.f) with $(adtype)\n"
    verbose && println("       params : $(params)")
    ldf = LogDensityFunction(model, varinfo; adtype=adtype)

    value, grad = logdensity_and_gradient(ldf, params)
    grad = _to_vec_f64(grad)
    verbose && println("       actual : $((value, grad))")

    if test
        # Calculate ground truth to compare against
        value_true, grad_true = if expected_value_and_grad === nothing
            ldf_reference = LogDensityFunction(model, varinfo; adtype=reference_adtype)
            logdensity_and_gradient(ldf_reference, params)
        else
            expected_value_and_grad
        end
        verbose && println("     expected : $((value_true, grad_true))")
        grad_true = _to_vec_f64(grad_true)
        # Then compare
        @test isapprox(value, value_true; atol=value_atol)
        @test isapprox(grad, grad_true; atol=grad_atol)
    else
        value_true = nothing
        grad_true = nothing
    end

    time_vs_primal = if benchmark
        primal_benchmark = @be (ldf, params) logdensity(_[1], _[2])
        grad_benchmark = @be (ldf, params) logdensity_and_gradient(_[1], _[2])
        t = median(grad_benchmark).time / median(primal_benchmark).time
        verbose && println("grad / primal : $(t)")
        t
    else
        nothing
    end

    return ADResult(
        model,
        varinfo,
        params,
        adtype,
        value_atol,
        grad_atol,
        value_true,
        grad_true,
        value,
        grad,
        time_vs_primal,
    )
end

end # module DynamicPPL.TestUtils.AD
