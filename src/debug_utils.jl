module DebugUtils

using ..DynamicPPL

using Random: Random
using InteractiveUtils: InteractiveUtils
using Distributions

export check_model, has_static_constraints

"""
    DebugAccumulator <: AbstractAccumulator

An accumulator which checks calls at each tilde-statement for potential errors.

Right now this accumulator only checks for `NaN` values on the left-hand side of observe
statements, and partially `missing` values on the left-hand side of observe statements.

Other checks in `check_model` are accomplished via different accumulators.
"""
struct DebugAccumulator <: AbstractAccumulator
    "A flag indicating whether this accumulator has found any issues with the model"
    failed::Bool
end
DebugAccumulator() = DebugAccumulator(false)

const _DEBUG_ACC_NAME = :Debug
DynamicPPL.accumulator_name(::Type{<:DebugAccumulator}) = _DEBUG_ACC_NAME

_zero(::DebugAccumulator) = DebugAccumulator(false)
DynamicPPL.reset(acc::DebugAccumulator) = _zero(acc)
DynamicPPL.split(acc::DebugAccumulator) = _zero(acc)

function DynamicPPL.combine(acc1::DebugAccumulator, acc2::DebugAccumulator)
    return DebugAccumulator(acc1.failed || acc2.failed)
end

"""
    _has_partial_missings(x, dist)

Check if `x` is a container that contains partial `missing` values.
"""
_has_partial_missings(x, dist) = false
function _has_partial_missings(x::AbstractArray, ::MultivariateDistribution)
    for i in eachindex(x)
        if isassigned(x, i) && ismissing(x[i])
            return true
        end
    end
    return false
end
function _has_partial_missings(
    x::NamedTuple{names}, dists::Distributions.ProductNamedTupleDistribution
) where {names}
    for name in names
        sub_value = x[name]
        sub_dist = dists.dists[name]
        if _has_partial_missings(sub_value, sub_dist)
            return true
        end
    end
    return false
end

"""
    _has_nans(x)

Check if `x` is `NaN`, or contains any `NaN` values.
"""
_has_nans(x::NamedTuple) = any(_has_nans, x)
_has_nans(x::AbstractArray) = any(_has_nans, x)
_has_nans(x) = isnan(x)
_has_nans(::Missing) = false

function DynamicPPL.accumulate_assume!!(
    acc::DebugAccumulator, val, tval, logjac, vn::VarName, right::Distribution, template
)
    return acc
end

function DynamicPPL.accumulate_observe!!(
    acc::DebugAccumulator, right::Distribution, val, vn::Union{VarName,Nothing}
)
    if _has_partial_missings(val, right)
        msg = if vn === nothing
            "on the left-hand side of an observe statement"
        else
            "for variable $(vn) on the left-hand side of an observe statement"
        end
        full_msg =
            "Encountered a container with one or more `missing` value(s) $msg." *
            " To treat the variable on the left-hand side as a random variable, you" *
            " should specify a single `missing` rather than a vector of `missing`s." *
            " It is not currently possible to set part but not all of a distribution" *
            " to be `missing`."
        @warn full_msg
        acc.failed = true
    end
    # Check for NaN's as well
    if _has_nans(val)
        msg =
            "Encountered a NaN value on the left-hand side of an" *
            " observe statement; this may indicate that your data" *
            " contain NaN values."
        @warn msg
        acc.failed = true
    end
    return acc
end

"""
    DynamicPPL.DebugUtils.check_model(
        [rng::Random.AbstractRNG,]
        model::Model;
        error_on_failure=false,
        fail_if_discrete=false
    )

Check `model` for potential issues. Returns `true` if the model check succeeded, `false`
otherwise.

The model is only evaluated a single time, so if the model contains any indeterminism,
results may differ across runs. The `rng` argument can be used to control reproducibility if
needed.

# Issues that this function checks for

- Repeated usage of the same or overlapping VarNames

- `NaN` on the left-hand side of observe statements

- (if `fail_if_discrete` is set) Usage of discrete distributions

- Empty models emit a warning, but do not fail (since they are not incorrect *per se*)

# Keyword arguments

- `error_on_failure::Bool`: Whether to throw an error (instead of just returning `false`) if
  the model check fails.

- `fail_if_discrete::Bool`: Whether to fail (i.e., return `false` or throw an error,
   depending on `error_on_failure`) when the model contains discrete distributions. Discrete
   distributions do not have a differentiable log-density and are incompatible with
   gradient-based approaches such as HMC / NUTS or optimisation.

# Examples

## Correct model

```jldoctest
julia> using DynamicPPL.DebugUtils: check_model; using Distributions

julia> @model demo_correct() = x ~ Normal()
demo_correct (generic function with 2 methods)

julia> model = demo_correct();

julia> check_model(model)
true

julia> cond_model = model | (x = 1.0,);

julia> # Empty models will issue a warning, but not a failure
       check_model(cond_model)
┌ Warning: The model does not contain any parameters.
└ @ DynamicPPL.DebugUtils DynamicPPL.jl/src/debug_utils.jl:215
true
```

## Incorrect model

```jldoctest; setup=:(using Distributions)
julia> using DynamicPPL.DebugUtils: check_model; using Distributions

julia> @model function demo_incorrect()
           # Sampling `x` twice.
           x ~ Normal()
           x ~ Exponential()
       end
demo_incorrect (generic function with 2 methods)

julia> # Notice that VarInfo(model_incorrect) evaluates the model, but doesn't actually
       # alert us to the issue of `x` being sampled twice.
       model = demo_incorrect(); varinfo = VarInfo(model);

julia> check_model(model; error_on_failure=true)
ERROR: varname x used multiple times in model
"""
function check_model(
    rng::Random.AbstractRNG, model::Model; error_on_failure=false, fail_if_discrete=false
)
    failed = false

    # Check that a variable in the model arguments is neither conditioned nor fixed.
    conditioned_vns = keys(DynamicPPL.conditioned(model.context))
    for vn in conditioned_vns
        if DynamicPPL.inargnames(vn, model)
            @warn "Variable $(vn) is both in the model arguments and in the conditioning!\n" *
                "Please use either conditioning through the model arguments, or through " *
                "`condition` / `|`, not both."
            failed = true
        end
    end

    # Run the model and collect the data we need
    oavi = DynamicPPL.OnlyAccsVarInfo((
        DebugAccumulator(),
        PriorDistributionAccumulator(),
        DynamicPPL.DebugRawValueAccumulator(),
    ))
    init_strategy = InitFromPrior()
    _, oavi = DynamicPPL.init!!(rng, model, oavi, init_strategy, UnlinkAll())

    # If there are no raw values, then there are no parameters, so we can skip the rest of
    # the checks (but we will warn).
    if isempty(get_raw_values(oavi))
        @warn "The model does not contain any parameters."
        return true
    end

    # Check if the DebugAccumulator found any issues with the model.
    debug_acc = DynamicPPL.getacc(oavi, Val(_DEBUG_ACC_NAME))
    failed = failed || debug_acc.failed

    # Check the DebugRawValueAccumulator
    debug_raw_value_acc = DynamicPPL.getacc(oavi, Val(DynamicPPL.RAW_VALUE_ACCNAME))
    repeated_vns = debug_raw_value_acc.f.repeated_vns
    if !isempty(repeated_vns)
        for vn in repeated_vns
            @warn (
                "Assigning to the variable $(vn) led to a previous value being overwritten." *
                " This indicates that a value is being set twice (e.g. if the same variable occurs in a model twice)."
            )
        end
        failed = true
    end

    # Check for discrete distributions if requested.
    # NOTE: This uses the `ValueSupport` from the type of `dist`, which may not
    # be accurate for composite distributions (e.g. `ProductDistribution`) that
    # mix discrete and continuous components. As of Distributions.jl v0.25,
    # such mixed products are typed as `Continuous`, so a discrete component
    # inside one would not be caught here.
    if fail_if_discrete
        prior_acc = DynamicPPL.getacc(oavi, DynamicPPL.PRIOR_ACCNAME).values
        for (vn, dist) in pairs(prior_acc)
            if dist isa Distributions.DiscreteDistribution
                msg =
                    "Variable $(vn) is sampled from a discrete distribution " *
                    "($(typeof(dist).name.wrapper)). Discrete distributions are not " *
                    "differentiable, and thus not compatible with approaches that " *
                    "require gradient information, e.g. HMC / NUTS or optimisation."
                @warn msg
                failed = true
            end
        end
    end

    if failed && error_on_failure
        error("Model check failed; please see the warnings above for details.")
    end

    return !failed
end
function check_model(model::Model; error_on_failure=false, fail_if_discrete=false)
    return check_model(
        Random.default_rng(),
        model;
        error_on_failure=error_on_failure,
        fail_if_discrete=fail_if_discrete,
    )
end

"""
    has_static_constraints([rng, ]model::Model; num_evals=5)

Attempts to detect whether `model` has static constraints (i.e., the support of all variables
is the same regardless of what their values are). Returns `true` if the model has static
constraints, `false` otherwise.

Note that this is a heuristic check based on sampling from the model multiple times
and checking if the model is consistent across runs.

# Arguments

- `rng::Random.AbstractRNG`: The random number generator to use when evaluating the model.
- `model::Model`: The model to check.

# Keyword Arguments
- `num_evals::Int`: The number of evaluations to perform. Default: `5`.
"""
function has_static_constraints(rng::Random.AbstractRNG, model::Model; num_evals::Int=5)
    prior_vnts = map(1:num_evals) do _
        accs = DynamicPPL.OnlyAccsVarInfo(PriorDistributionAccumulator())
        _, accs = DynamicPPL.init!!(rng, model, accs, InitFromPrior(), UnlinkAll())
        return only(DynamicPPL.getaccs(accs)).values
    end
    all_vns = mapreduce(keys, vcat, prior_vnts)
    for vn in all_vns
        # Check that the bijector for `vn` is the same across all runs. (Note that
        # the distribution can vary, as long as the bijector doesn't change)
        bijectors = map(vnts -> DynamicPPL.link_transform(vnts[vn]), prior_vnts)
        if !isempty(bijectors) && any(b -> b != bijectors[1], bijectors)
            return false
        end
    end
    return true
end
function has_static_constraints(model::Model; num_evals::Int=5)
    return has_static_constraints(Random.default_rng(), model; num_evals=num_evals)
end

"""
    gen_evaluator_call_with_types(model[, varinfo])

Generate the evaluator call and the types of the arguments.

# Arguments
- `model::Model`: The model whose evaluator is of interest.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Returns
A 2-tuple with the following elements:
- `f`: This is either `model.f` or `Core.kwcall`, depending on whether
    the model has keyword arguments.
- `argtypes::Type{<:Tuple}`: The types of the arguments for the evaluator.
"""
function gen_evaluator_call_with_types(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model)
)
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo)
    return if isempty(kwargs)
        (model.f, Base.typesof(args...))
    else
        (Core.kwcall, Tuple{typeof(kwargs),Core.Typeof(model.f),map(Core.Typeof, args)...})
    end
end

"""
    model_warntype(model[, varinfo]; optimize=true)

Check the type stability of the model's evaluator, warning about any potential issues.

This simply calls `@code_warntype` on the model's evaluator, filling in internal arguments where needed.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Keyword Arguments
- `optimize::Bool`: Whether to generate optimized code. Default: `false`.
"""
function model_warntype(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model), optimize::Bool=false
)
    ftype, argtypes = gen_evaluator_call_with_types(model, varinfo)
    return InteractiveUtils.code_warntype(ftype, argtypes; optimize=optimize)
end

"""
    model_typed(model[, varinfo]; optimize=true)

Return the type inference for the model's evaluator.

This simply calls `@code_typed` on the model's evaluator, filling in internal arguments where needed.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Keyword Arguments
- `optimize::Bool`: Whether to generate optimized code. Default: `true`.
"""
function model_typed(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model), optimize::Bool=true
)
    ftype, argtypes = gen_evaluator_call_with_types(model, varinfo)
    return only(InteractiveUtils.code_typed(ftype, argtypes; optimize=optimize))
end

end
