module AD

using ADTypes: AbstractADType, AutoForwardDiff
using Chairmarks: @be
using DocStringExtensions
using DynamicPPL:
    DynamicPPL,
    Model,
    LogDensityFunction,
    AbstractTransformStrategy,
    LinkAll,
    getlogjoint_internal,
    InitFromPrior

using LinearAlgebra: norm
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
    atol::T
    rtol::T
end
function Base.showerror(io::IO, e::ADIncorrectException)
    value_passed = isapprox(e.value_expected, e.value_actual; atol=e.atol, rtol=e.rtol)
    grad_passed = isapprox(e.grad_expected, e.grad_actual; atol=e.atol, rtol=e.rtol)
    s = if !value_passed && !grad_passed
        "value and gradient"
    elseif !value_passed
        "value"
    else
        "gradient"
    end
    println(io, "ADIncorrectException: The AD backend returned an incorrect $s.")
    println(io, "  Testing was carried out with")
    println(io, "               atol : $(e.atol)")
    println(io, "               rtol : $(e.rtol)")
    # calculate what tolerances would have been needed to pass for value
    if !value_passed
        min_atol_needed_to_pass_value = abs(e.value_expected - e.value_actual)
        min_rtol_needed_to_pass_value =
            min_atol_needed_to_pass_value / max(abs(e.value_expected), abs(e.value_actual))
        println(io, "  The value check failed because:")
        println(io, "     expected value : $(e.value_expected)")
        println(io, "       actual value : $(e.value_actual)")
        println(io, "  This value correctness check would have passed if either:")
        println(io, "               atol ≥ $(min_atol_needed_to_pass_value), or")
        println(io, "               rtol ≥ $(min_rtol_needed_to_pass_value)")
    end
    if !grad_passed
        norm_expected = norm(e.grad_expected)
        norm_actual = norm(e.grad_actual)
        max_norm = max(norm_expected, norm_actual)
        norm_diff = norm(e.grad_expected - e.grad_actual)
        min_atol_needed_to_pass_grad = norm_diff
        min_rtol_needed_to_pass_grad = norm_diff / max_norm
        # min tolerances needed to pass overall
        println(io, "  The gradient check failed because:")
        println(io, "      expected grad : $(e.grad_expected)")
        println(io, "        actual grad : $(e.grad_actual)")
        println(io, "  The gradient correctness check would have passed if either:")
        println(io, "               atol ≥ $(min_atol_needed_to_pass_grad), or")
        println(io, "               rtol ≥ $(min_rtol_needed_to_pass_grad)")
    end
    return nothing
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
    "The function used to extract the log density from the model"
    getlogdensity::Function

    "The LogDensityFunction that was used"
    ldf::LogDensityFunction
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
    "If benchmarking was requested, the time taken by the AD backend to evaluate the gradient
    of logp (in seconds)"
    grad_time::Union{Nothing,Tresult}
    "If benchmarking was requested, the time taken by the AD backend to evaluate logp (in
    seconds)"
    primal_time::Union{Nothing,Tresult}
end
function Base.show(io::IO, ::MIME"text/plain", result::ADResult)
    printstyled(io, "ADResult\n"; bold=true)
    println(io, "  ├ model          : $(result.model.f)")
    println(io, "  ├ adtype         : $(result.adtype)")
    println(io, "  ├ value_actual   : $(result.value_actual)")
    println(io, "  ├ value_expected : $(result.value_expected)")
    println(io, "  ├ grad_actual    : $(result.grad_actual)")
    println(io, "  ├ grad_expected  : $(result.grad_expected)")
    if result.grad_time !== nothing && result.primal_time !== nothing
        println(io, "  ├ grad_time      : $(result.grad_time) s")
        println(io, "  ├ primal_time    : $(result.primal_time) s")
    end
    return println(io, "  └ params         : $(result.params)")
end

"""
    run_ad(
        model::Model,
        adtype::ADTypes.AbstractADType;
        test::Union{AbstractADCorrectnessTestSetting,Bool}=WithBackend(),
        benchmark=false,
        atol::AbstractFloat=1e-8,
        rtol::AbstractFloat=sqrt(eps()),
        getlogdensity::Function=getlogjoint_internal,
        rng::Random.AbstractRNG=Random.default_rng(),
        transform_strategy::AbstractTransformStrategy=LinkAll(),
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
1. `adtype` - The AD backend being tested.

Everything else is optional, and can be categorised into several groups:

1. _Whether to evaluate in linked space or not._

   If requested, DynamicPPL internally transforms the model parameters to unconstrained
   Euclidean space before evaluating the log density in that transformed space.

   You can control whether this transformation happens or not by passing the
   `transform_strategy` keyword argument. The default is `LinkAll()`, which means that all
   parameters will be transformed to unconstrained space. This is the most relevant setting
   for testing AD.

   However, if you want to evaluate in the original space, you can use `UnlinkAll()`; you
   can also specify mixed linking strategies if desired (see [the DynamicPPL
   documentation](@ref transform-strategies) for more information). 

1. _How to specify the parameters to be used for evaluation._

   For maximum control over this, generate a vector of parameters yourself and pass this as
   the `params` argument. If you don't specify this, it will be generated randomly from the
   prior of the model. If you want to seed the parameter generation, you can pass the `rng`
   keyword argument, which will then be used to generate the parameters.

1. _Which type of logp is being calculated._

   By default, `run_ad` evaluates the 'internal log joint density' of the model,
   i.e., the log joint density in the unconstrained space. Thus, for example, in

       @model f() = x ~ LogNormal()

   the internal log joint density is `logpdf(Normal(), log(x))`. This is the
   relevant log density for e.g. Hamiltonian Monte Carlo samplers and is therefore
   the most useful to test.

   If you want the log joint density in the original model parameterisation, you
   can use `getlogjoint`. Likewise, if you want only the prior or likelihood,
   you can use `getlogprior` or `getloglikelihood`, respectively.

1. _How to specify the results to compare against._

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

1. _How to specify the tolerances._ (Only if testing is enabled.)

   Both absolute and relative tolerances can be specified using the `atol` and
   `rtol` keyword arguments respectively. The behaviour of these is similar to
   `isapprox()`, i.e. the value and gradient are considered correct if either
   atol or rtol is satisfied. The default values are `100*eps()` for `atol` and
   `sqrt(eps())` for `rtol`.

   For the most part, it is the `rtol` check that is more meaningful, because
   we cannot know the magnitude of logp and its gradient a priori. The `atol`
   value is supplied to handle the case where gradients are equal to zero.

1. _Whether to benchmark._

   By default, benchmarking is disabled. To enable it, set `benchmark=true`.
   When enabled, the time taken to evaluate logp as well as its gradient is
   measured using Chairmarks.jl, and the `ADResult` object returned will
   contain `grad_time` and `primal_time` fields with the median times (in
   seconds). The `benchmark_seconds` keyword (default `1`) sets the time
   budget passed to Chairmarks for each of the two measurements; raising it
   collects more samples and yields a tighter median estimate at the cost
   of a longer run.

1. _Whether to output extra logging information._

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
    benchmark_seconds::Real=1,
    atol::AbstractFloat=100 * eps(),
    rtol::AbstractFloat=sqrt(eps()),
    getlogdensity::Function=getlogjoint_internal,
    rng::AbstractRNG=default_rng(),
    transform_strategy::AbstractTransformStrategy=LinkAll(),
    params::Union{Nothing,Vector{<:AbstractFloat}}=nothing,
    verbose=true,
)::ADResult
    # Convert Boolean `test` to an AbstractADCorrectnessTestSetting
    if test isa Bool
        test = test ? WithBackend() : NoTest()
    end

    verbose && @info "Running AD on $(model.f) with $(adtype)\n"

    # Generate initial parameters
    ldf = LogDensityFunction(model, getlogdensity, transform_strategy; adtype=adtype)
    if isnothing(params)
        params = rand(rng, ldf, InitFromPrior())
    end

    params = [p for p in params]  # Concretise
    verbose && println("       params : $(params)")

    # Calculate log-density and gradient with the backend of interest
    value, grad = logdensity_and_gradient(ldf, params)
    # collect(): some backends (e.g. Enzyme) return non-Vector gradients
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
            ldf_reference = LogDensityFunction(
                model, getlogdensity, transform_strategy; adtype=test.adtype
            )
            value_true, grad_true = logdensity_and_gradient(ldf_reference, params)
            grad_true = collect(grad_true)
        end
        # Perform testing
        verbose && println("     expected : $((value_true, grad_true))")
        exc() = throw(ADIncorrectException(value, value_true, grad, grad_true, atol, rtol))
        isapprox(value, value_true; atol=atol, rtol=rtol) || exc()
        isapprox(grad, grad_true; atol=atol, rtol=rtol) || exc()
    end

    # Benchmark
    grad_time, primal_time = if benchmark
        # Per-sample incremental GC keeps accumulated garbage from triggering a
        # full collection mid-sample, which would inflate that sample several-
        # fold. Auto-tuned `evals` (not pinned to 1) batches enough calls per
        # sample that fast log-densities clear `time_ns`'s real precision floor
        # (tens of ns on Linux/macOS) instead of reading as zero. Pattern
        # borrowed from Mooncake's bench harness:
        # https://github.com/chalk-lab/Mooncake.jl/blob/main/bench/run_benchmarks.jl
        # Per-sample `setup` deep-copies `params` so each sample starts from a
        # fresh input buffer, matching Mooncake's bench harness. (Setup runs
        # before the timed window, so the copy is excluded from measurements.)
        logdensity(ldf, params)  # Warm-up
        GC.gc(true)
        primal_benchmark = @be(
            deepcopy($params),
            logdensity($ldf, _),
            _ -> GC.gc(false),
            seconds = benchmark_seconds,
        )
        if verbose
            print("   evaluation : ")
            show(stdout, MIME("text/plain"), median(primal_benchmark))
            println()
        end
        logdensity_and_gradient(ldf, params)  # Warm-up
        GC.gc(true)
        grad_benchmark = @be(
            deepcopy($params),
            logdensity_and_gradient($ldf, _),
            _ -> GC.gc(false),
            seconds = benchmark_seconds,
        )
        if verbose
            print("     gradient : ")
            show(stdout, MIME("text/plain"), median(grad_benchmark))
            println()
        end
        median_primal = median(primal_benchmark).time
        median_grad = median(grad_benchmark).time
        r(f) = round(f; sigdigits=4)
        verbose && println("  grad / eval : $(r(median_grad / median_primal))")
        (median_grad, median_primal)
    else
        nothing, nothing
    end

    return ADResult(
        model,
        getlogdensity,
        ldf,
        params,
        adtype,
        atol,
        rtol,
        value_true,
        grad_true,
        value,
        grad,
        grad_time,
        primal_time,
    )
end

end # module DynamicPPL.TestUtils.AD
