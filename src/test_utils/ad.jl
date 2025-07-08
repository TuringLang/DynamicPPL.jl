module AD

using ADTypes: AbstractADType, AutoForwardDiff
using Chairmarks: @be
import DifferentiationInterface as DI
using DocStringExtensions
using DynamicPPL: Model, LogDensityFunction, VarInfo, AbstractVarInfo, link
using LogDensityProblems: logdensity, logdensity_and_gradient
using Random: AbstractRNG, default_rng
using Statistics: median
using Test: @test

export ADResult, run_ad, ADIncorrectException, WithBackend, WithExpectedResult, NoTest

"""
    AbstractADCorrectnessTestSetting

Different ways of testing the correctness of an AD backend.
"""
abstract type AbstractADCorrectnessTestSetting end

"""
    WithBackend(adtype::AbstractADType=AutoForwardDiff()) <: AbstractADCorrectnessTestSetting

Test correctness by comparing it against the result obtained with `adtype`.

`adtype` defaults to ForwardDiff.jl, since it's the default AD backend used in
Turing.jl.
"""
struct WithBackend{AD<:AbstractADType} <: AbstractADCorrectnessTestSetting
    adtype::AD
end
WithBackend() = WithBackend(AutoForwardDiff())

"""
    WithExpectedResult(
        value::T,
        grad::AbstractVector{T}
    ) where {T <: AbstractFloat}
    <: AbstractADCorrectnessTestSetting

Test correctness by comparing it against a known result (e.g. one obtained
analytically, or one obtained with a different backend previously). Both the
value of the primal (i.e. the log-density) as well as its gradient must be
supplied.
"""
struct WithExpectedResult{T<:AbstractFloat} <: AbstractADCorrectnessTestSetting
    value::T
    grad::AbstractVector{T}
end

"""
    NoTest() <: AbstractADCorrectnessTestSetting

Disable correctness testing.
"""
struct NoTest <: AbstractADCorrectnessTestSetting end

"""
    ADIncorrectException{T<:AbstractFloat}

Exception thrown when an AD backend returns an incorrect value or gradient.

The type parameter `T` is the numeric type of the value and gradient.

# Fields
$(TYPEDFIELDS)
"""
struct ADIncorrectException{T<:AbstractFloat} <: Exception
    value_expected::T
    value_actual::T
    grad_expected::Vector{T}
    grad_actual::Vector{T}
end

"""
    ADResult{Tparams<:AbstractFloat,Tresult<:AbstractFloat,Ttol<:AbstractFloat}

Data structure to store the results of the AD correctness test.

The type parameter `Tparams` is the numeric type of the parameters passed in;
`Tresult` is the type of the value and the gradient; and `Ttol` is the type of the
absolute and relative tolerances used for correctness testing.

# Fields
$(TYPEDFIELDS)
"""
struct ADResult{Tparams<:AbstractFloat,Tresult<:AbstractFloat,Ttol<:AbstractFloat}
    "The DynamicPPL model that was tested"
    model::Model
    "The VarInfo that was used"
    varinfo::AbstractVarInfo
    "The values at which the model was evaluated"
    params::Vector{Tparams}
    "The AD backend that was tested"
    adtype::AbstractADType
    "Absolute tolerance used for correctness test"
    atol::Ttol
    "Relative tolerance used for correctness test"
    rtol::Ttol
    "The expected value of logp"
    value_expected::Union{Nothing,Tresult}
    "The expected gradient of logp"
    grad_expected::Union{Nothing,Vector{Tresult}}
    "The value of logp (calculated using `adtype`)"
    value_actual::Tresult
    "The gradient of logp (calculated using `adtype`)"
    grad_actual::Vector{Tresult}
    "If benchmarking was requested, the time taken by the AD backend to calculate the gradient of logp, divided by the time taken to evaluate logp itself"
    time_vs_primal::Union{Nothing,Tresult}
end

"""
    run_ad(
        model::Model,
        adtype::ADTypes.AbstractADType;
        test::Union{AbstractADCorrectnessTestSetting,Bool}=WithBackend(),
        benchmark=false,
        atol::AbstractFloat=1e-8,
        rtol::AbstractFloat=sqrt(eps()),
        varinfo::AbstractVarInfo=link(VarInfo(model), model),
        params::Union{Nothing,Vector{<:AbstractFloat}}=nothing,
        verbose=true,
    )::ADResult

### Description

Test the correctness and/or benchmark the AD backend `adtype` for the model
`model`.

Whether to test and benchmark is controlled by the `test` and `benchmark`
keyword arguments. By default, `test` is `true` and `benchmark` is `false`.

Note that to run AD successfully you will need to import the AD backend itself.
For example, to test with `AutoReverseDiff()` you will need to run `import
ReverseDiff`.

### Arguments

There are two positional arguments, which absolutely must be provided:

1. `model` - The model being tested.
2. `adtype` - The AD backend being tested.

Everything else is optional, and can be categorised into several groups:

1. _How to specify the VarInfo._

   DynamicPPL contains several different types of VarInfo objects which change
   the way model evaluation occurs. If you want to use a specific type of
   VarInfo, pass it as the `varinfo` argument. Otherwise, it will default to
   using a linked `TypedVarInfo` generated from the model. Here, _linked_
   means that the parameters in the VarInfo have been transformed to
   unconstrained Euclidean space if they aren't already in that space.

2. _How to specify the parameters._

   For maximum control over this, generate a vector of parameters yourself and
   pass this as the `params` argument. If you don't specify this, it will be
   taken from the contents of the VarInfo.

   Note that if the VarInfo is not specified (and thus automatically generated)
   the parameters in it will have been sampled from the prior of the model. If
   you want to seed the parameter generation for the VarInfo, you can pass the
   `rng` keyword argument, which will then be used to create the VarInfo.

   Finally, note that these only reflect the parameters used for _evaluating_
   the gradient. If you also want to control the parameters used for
   _preparing_ the gradient, then you need to manually set these parameters in
   the VarInfo object, for example using `vi = DynamicPPL.unflatten(vi,
   prep_params)`. You could then evaluate the gradient at a different set of
   parameters using the `params` keyword argument.

3. _How to specify the results to compare against._

   Once logp and its gradient has been calculated with the specified `adtype`,
   it can optionally be tested for correctness. The exact way this is tested 
   is specified in the `test` parameter.

   There are several options for this:

    - You can explicitly specify the correct value using
      [`WithExpectedResult()`](@ref).
    - You can compare against the result obtained with a different AD backend
      using [`WithBackend(adtype)`](@ref).
    - You can disable testing by passing [`NoTest()`](@ref).
    - The default is to compare against the result obtained with ForwardDiff,
      i.e. `WithBackend(AutoForwardDiff())`.
    - `test=false` and `test=true` are synonyms for
      `NoTest()` and `WithBackend(AutoForwardDiff())`, respectively.

4. _How to specify the tolerances._ (Only if testing is enabled.)

   Both absolute and relative tolerances can be specified using the `atol` and
   `rtol` keyword arguments respectively. The behaviour of these is similar to
   `isapprox()`, i.e. the value and gradient are considered correct if either
   atol or rtol is satisfied. The default values are `100*eps()` for `atol` and
   `sqrt(eps())` for `rtol`.

   For the most part, it is the `rtol` check that is more meaningful, because
   we cannot know the magnitude of logp and its gradient a priori. The `atol`
   value is supplied to handle the case where gradients are equal to zero.

5. _Whether to output extra logging information._

   By default, this function prints messages when it runs. To silence it, set
   `verbose=false`.

### Returns / Throws

Returns an [`ADResult`](@ref) object, which contains the results of the
test and/or benchmark.

If `test` is `true` and the AD backend returns an incorrect value or gradient, an
`ADIncorrectException` is thrown. If a different error occurs, it will be
thrown as-is.
"""
function run_ad(
    model::Model,
    adtype::AbstractADType;
    test::Union{AbstractADCorrectnessTestSetting,Bool}=WithBackend(),
    benchmark::Bool=false,
    atol::AbstractFloat=100 * eps(),
    rtol::AbstractFloat=sqrt(eps()),
    rng::AbstractRNG=default_rng(),
    varinfo::AbstractVarInfo=link(VarInfo(rng, model), model),
    params::Union{Nothing,Vector{<:AbstractFloat}}=nothing,
    verbose=true,
)::ADResult
    # Convert Boolean `test` to an AbstractADCorrectnessTestSetting
    if test isa Bool
        test = test ? WithBackend() : NoTest()
    end

    # Extract parameters
    if isnothing(params)
        params = varinfo[:]
    end
    params = map(identity, params)  # Concretise

    # Calculate log-density and gradient with the backend of interest
    verbose && @info "Running AD on $(model.f) with $(adtype)\n"
    verbose && println("       params : $(params)")
    ldf = LogDensityFunction(model, varinfo; adtype=adtype)
    value, grad = logdensity_and_gradient(ldf, params)
    # collect(): https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/754
    grad = collect(grad)
    verbose && println("       actual : $((value, grad))")

    # Test correctness
    if test isa NoTest
        value_true = nothing
        grad_true = nothing
    else
        # Get the correct result
        if test isa WithExpectedResult
            value_true = test.value
            grad_true = test.grad
        elseif test isa WithBackend
            ldf_reference = LogDensityFunction(model, varinfo; adtype=test.adtype)
            value_true, grad_true = logdensity_and_gradient(ldf_reference, params)
            # collect(): https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/754
            grad_true = collect(grad_true)
        end
        # Perform testing
        verbose && println("     expected : $((value_true, grad_true))")
        exc() = throw(ADIncorrectException(value, value_true, grad, grad_true))
        isapprox(value, value_true; atol=atol, rtol=rtol) || exc()
        isapprox(grad, grad_true; atol=atol, rtol=rtol) || exc()
    end

    # Benchmark
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
        atol,
        rtol,
        value_true,
        grad_true,
        value,
        grad,
        time_vs_primal,
    )
end

end # module DynamicPPL.TestUtils.AD
