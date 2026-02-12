module DebugUtils

using ..DynamicPPL

using Random: Random
using InteractiveUtils: InteractiveUtils

using DocStringExtensions: TYPEDFIELDS
using Distributions

export check_model, has_static_constraints

"""
    DebugAccumulator <: AbstractAccumulator

An accumulator which collects enough information to potentially catch errors in the model.

# Fields
$(TYPEDFIELDS)
"""
struct DebugAccumulator <: AbstractAccumulator
    "mapping from varnames to the number of times they have been seen"
    varnames_seen::OrderedDict{VarName,Int}
    "whether to throw an error if we encounter errors in the model"
    error_on_failure::Bool
    "whether to check for discrete distributions (incompatible with differentiation)"
    check_discrete::Bool
end

function DebugAccumulator(error_on_failure=false, check_discrete=false)
    return DebugAccumulator(OrderedDict{VarName,Int}(), error_on_failure, check_discrete)
end

const _DEBUG_ACC_NAME = :Debug
DynamicPPL.accumulator_name(::Type{<:DebugAccumulator}) = _DEBUG_ACC_NAME

function Base.:(==)(acc1::DebugAccumulator, acc2::DebugAccumulator)
    return (
        acc1.varnames_seen == acc2.varnames_seen &&
        acc1.error_on_failure == acc2.error_on_failure &&
        acc1.check_discrete == acc2.check_discrete
    )
end

function _zero(acc::DebugAccumulator)
    return DebugAccumulator(
        OrderedDict{VarName,Int}(), acc.error_on_failure, acc.check_discrete
    )
end
DynamicPPL.reset(acc::DebugAccumulator) = _zero(acc)
DynamicPPL.split(acc::DebugAccumulator) = _zero(acc)
function DynamicPPL.combine(acc1::DebugAccumulator, acc2::DebugAccumulator)
    return DebugAccumulator(
        merge(acc1.varnames_seen, acc2.varnames_seen),
        acc1.error_on_failure || acc2.error_on_failure,
        acc1.check_discrete || acc2.check_discrete,
    )
end

function record_varname!(acc::DebugAccumulator, varname::VarName, dist)
    if haskey(acc.varnames_seen, varname)
        if acc.error_on_failure
            error("varname $varname used multiple times in model")
        else
            @warn "varname $varname used multiple times in model"
        end
        acc.varnames_seen[varname] += 1
    else
        # We need to check:
        # 1. Does this `varname` subsume any of the other keys.
        # 2. Does any of the other keys subsume `varname`.
        vns = collect(keys(acc.varnames_seen))
        # Is `varname` subsumed by any of the other keys?
        idx_parent = findfirst(Base.Fix2(subsumes, varname), vns)
        if idx_parent !== nothing
            varname_parent = vns[idx_parent]
            if acc.error_on_failure
                error(
                    "varname $(varname_parent) used multiple times in model (subsumes $varname)",
                )
            else
                @warn "varname $(varname_parent) used multiple times in model (subsumes $varname)"
            end
            # Update count of parent.
            acc.varnames_seen[varname_parent] += 1
        else
            # Does `varname` subsume any of the other keys?
            idx_child = findfirst(Base.Fix1(subsumes, varname), vns)
            if idx_child !== nothing
                varname_child = vns[idx_child]
                if acc.error_on_failure
                    error(
                        "varname $(varname_child) used multiple times in model (subsumed by $varname)",
                    )
                else
                    @warn "varname $(varname_child) used multiple times in model (subsumed by $varname)"
                end

                # Update count of child.
                acc.varnames_seen[varname_child] += 1
            end
        end

        acc.varnames_seen[varname] = 1
    end
end

_has_missings(x) = ismissing(x)
function _has_missings(x::AbstractArray)
    # Can't just use `any` because `x` might contain `undef`.
    for i in eachindex(x)
        if isassigned(x, i) && _has_missings(x[i])
            return true
        end
    end
    return false
end

_has_nans(x::NamedTuple) = any(_has_nans, x)
_has_nans(x::AbstractArray) = any(_has_nans, x)
_has_nans(x) = isnan(x)
_has_nans(::Missing) = false

function DynamicPPL.accumulate_assume!!(
    acc::DebugAccumulator, val, tval, logjac, vn::VarName, right::Distribution, template
)
    record_varname!(acc, vn, right)
    # Check for discrete distributions if requested.
    # NOTE: This uses the `ValueSupport` from the type of `right`, which may not
    # be accurate for composite distributions (e.g. `ProductDistribution`) that
    # mix discrete and continuous components. As of Distributions.jl v0.25,
    # such mixed products are typed as `Continuous`, so a discrete component
    # inside one would not be caught here. This is arguably a bug in
    # Distributions.jl rather than something to work around here.
    if acc.check_discrete
        if right isa Distributions.DiscreteDistribution
            msg =
                "Variable $(vn) is sampled from a discrete distribution " *
                "($(typeof(right).name.wrapper)). Discrete distributions are not " *
                "differentiable, and thus not compatible with approaches that " *
                "require gradient information, e.g. HMC / NUTS or optimisation."
            if acc.error_on_failure
                error(msg)
            else
                @warn msg
            end
        end
    end
    return acc
end

function DynamicPPL.accumulate_observe!!(
    acc::DebugAccumulator, right::Distribution, val, vn::Union{VarName,Nothing}
)
    if _has_missings(val)
        # If `val` itself is a missing, that's a bug because that should cause
        # us to go down the assume path.
        val === missing && error(
            "Encountered `missing` value on the left-hand side of an observe" *
            " statement. This should not happen. Please open an issue at" *
            " https://github.com/TuringLang/DynamicPPL.jl.",
        )
        # Otherwise it's an array with some missing values.
        msg =
            "Encountered a container with one or more `missing` value(s) on the" *
            " left-hand side of an observe statement. To treat the variable on" *
            " the left-hand side as a random variable, you should specify a single" *
            " `missing` rather than a vector of `missing`s. It is not possible to" *
            " set part but not all of a distribution to be `missing`."
        if acc.error_on_failure
            error(msg)
        else
            @warn msg
        end
    end
    # Check for NaN's as well
    if _has_nans(val)
        msg =
            "Encountered a NaN value on the left-hand side of an" *
            " observe statement; this may indicate that your data" *
            " contain NaN values."
        if acc.error_on_failure
            error(msg)
        else
            @warn msg
        end
    end
    return acc
end

function check_varnames_seen(varnames_seen::AbstractDict{VarName,Int})
    if isempty(varnames_seen)
        @warn "The model does not contain any parameters."
        return true
    end

    issuccess = true
    for (varname, count) in varnames_seen
        if count == 0
            @warn "varname $varname was never seen"
            issuccess = false
        elseif count > 1
            @warn "varname $varname was seen $count times; it should only be seen once!"
            issuccess = false
        end
    end

    return issuccess
end

# A check we run on the model before evaluating it.
function check_model_pre_evaluation(model::Model)
    issuccess = true
    # If something is in the model arguments, then it should NOT be in `condition`,
    # nor should there be any symbol present in `condition` that has the same symbol.
    conditioned_vns = keys(DynamicPPL.conditioned(model.context))
    for vn in conditioned_vns
        if DynamicPPL.inargnames(vn, model)
            @warn "Variable $(vn) is both in the model arguments and in the conditioning!\n" *
                "Please use either conditioning through the model arguments, or through " *
                "`condition` / `|`, not both."

            issuccess = false
        end
    end

    return issuccess
end

function check_model_post_evaluation(acc::DebugAccumulator)
    return check_varnames_seen(acc.varnames_seen)
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
└ @ DynamicPPL.DebugUtils DynamicPPL.jl/src/debug_utils.jl:342
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
    # Perform checks before evaluating the model.
    issuccess = check_model_pre_evaluation(model)

    # TODO(penelopeysm): Implement merge, etc. for DebugAccumulator, and then perform a
    # check on the merged accumulator, rather than checking it in the accumulate_assume
    # calls. That way we can also correctly support multi-threaded evaluation.
    oavi = DynamicPPL.OnlyAccsVarInfo((
        DebugAccumulator(error_on_failure, fail_if_discrete),
    ))
    init_strategy = InitFromPrior()
    _, oavi = DynamicPPL.init!!(rng, model, oavi, init_strategy, UnlinkAll())

    # Perform checks after evaluating the model.
    debug_acc = DynamicPPL.getacc(oavi, Val(_DEBUG_ACC_NAME))
    issuccess = issuccess && check_model_post_evaluation(debug_acc)

    if !issuccess && error_on_failure
        error("model check failed")
    end

    return issuccess
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
