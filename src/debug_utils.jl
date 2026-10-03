module DebugUtils

using ..DynamicPPL

using Bijectors: Bijectors
using Random: Random
using InteractiveUtils: InteractiveUtils
using Distributions

export check_model, has_static_constraints

# Accumulators see distributions and values, not submodel calls (or fixed LHS variables).
# Track reached models only during check_model through its initialisation strategy.
struct BindingCheckStrategy{S<:AbstractInitStrategy} <: AbstractInitStrategy
    strategy::S
    models::Vector{Model}
    namespaces::Set{Symbol}
    lock::ReentrantLock
end
function DynamicPPL.init(rng, vn, dist, strategy::BindingCheckStrategy)
    return DynamicPPL.init(rng, vn, dist, strategy.strategy)
end
function DynamicPPL.get_param_eltype(strategy::BindingCheckStrategy)
    return DynamicPPL.get_param_eltype(strategy.strategy)
end

function DynamicPPL.tilde_assume!!(
    parent::Model,
    ctx::Context{<:Random.AbstractRNG,<:BindingCheckStrategy},
    submodel::DynamicPPL.Submodel{M,AutoPrefix},
    vn::VarName,
    template,
    vi::AbstractVarInfo,
) where {M<:Model,AutoPrefix}
    checking = ctx.strategy
    if AutoPrefix || DynamicPPL.getprefix(submodel.model) !== nothing
        # This child and its descendants have their own namespace.
        namespace = AutoPrefix ? vn : DynamicPPL.getprefix(submodel.model)
        lock(checking.lock) do
            push!(checking.namespaces, DynamicPPL.AbstractPPL.getsym(namespace))
        end
        context = Context(ctx.rng, checking.strategy, ctx.transform_strategy)
        return DynamicPPL.tilde_assume!!(parent, context, submodel, vn, template, vi)
    end
    lock(checking.lock) do
        push!(checking.models, submodel.model)
    end
    # Retain the checking strategy for unprefixed descendants, while delegating the
    # actual evaluation to the ordinary submodel method.
    return invoke(
        DynamicPPL.tilde_assume!!,
        Tuple{Model,Context,typeof(submodel),VarName,Any,AbstractVarInfo},
        parent,
        ctx,
        submodel,
        vn,
        template,
        vi,
    )
end

function _warn_unused_binding_names(model, ctx::BindingCheckStrategy)
    names = copy(ctx.namespaces)
    for reached in ctx.models
        lhs = DynamicPPL._lhs_names(DynamicPPL._binding_metadata(reached))
        # Handwritten models do not promise complete LHS metadata.
        lhs === nothing && return nothing
        union!(names, lhs)
    end
    for name in keys(DynamicPPL._submodel_values(model, nothing).data)
        if name ∉ names
            @warn "Binding `$name` has no LHS top symbol in this model or any reached unprefixed submodel. It may be unused; an unprefixed submodel in an untaken branch could still use it."
        end
    end
    return nothing
end

"""
    DebugAccumulator <: AbstractAccumulator

An accumulator which checks calls at each tilde-statement for potential errors.

This accumulator checks for `NaN` values on the left-hand side of observe statements.

Other checks in `check_model` are accomplished via different accumulators.
"""
struct DebugAccumulator <: AbstractAccumulator
    "A flag indicating whether this accumulator has found any issues with the model"
    failed::Bool
end
DebugAccumulator() = DebugAccumulator(false)
Base.copy(acc::DebugAccumulator) = acc

const _DEBUG_ACC_NAME = :Debug
DynamicPPL.accumulator_name(::Type{<:DebugAccumulator}) = _DEBUG_ACC_NAME

_zero(::DebugAccumulator) = DebugAccumulator(false)
DynamicPPL.reset(acc::DebugAccumulator) = _zero(acc)
DynamicPPL.split(acc::DebugAccumulator) = _zero(acc)

function DynamicPPL.combine(acc1::DebugAccumulator, acc2::DebugAccumulator)
    return DebugAccumulator(acc1.failed || acc2.failed)
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
    acc::DebugAccumulator, right::Distribution, val, vn::Union{VarName,Nothing}, template
)
    failed = acc.failed
    if _has_nans(val)
        msg =
            "Encountered a NaN value on the left-hand side of an" *
            " observe statement; this may indicate that your data" *
            " contain NaN values."
        @warn msg
        failed = true
    end
    return DebugAccumulator(failed)
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

The main check evaluates `model` once, so indeterminism may produce different results across
runs. If ForwardDiff is loaded, a second best-effort evaluation checks for latent variables
derived from model inputs. Use `rng` to control reproducibility if needed.

# Issues that this function checks for

- Repeated usage of the same or overlapping VarNames

- `NaN` on the left-hand side of observe statements

- Input-derived values overwritten by latent tilde statements, if ForwardDiff is loaded

- (if `fail_if_discrete` is set) Usage of discrete distributions

- Empty models emit a warning, but do not fail (since they are not incorrect *per se*)

- Bindings without an LHS top symbol in the model or reached unprefixed submodels warn,
  but do not fail: an untaken submodel branch could still use them.

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
┌ Warning: Assigning to the variable x led to a previous value being overwritten. This indicates that a value is being set twice (e.g. if the same variable occurs in a model twice).
└ @ DynamicPPL.DebugUtils DynamicPPL.jl/src/debug_utils.jl:237
ERROR: Model check failed; please see the warnings above for details.
```
"""
function check_model(
    rng::Random.AbstractRNG, model::Model; error_on_failure=false, fail_if_discrete=false
)
    failed = false

    # Run the model and collect the data we need
    vi = DynamicPPL.VarInfo((
        DebugAccumulator(),
        PriorDistributionAccumulator(),
        DynamicPPL.DebugRawValueAccumulator(),
    ))
    checking = BindingCheckStrategy(
        InitFromPrior(), Model[model], Set{Symbol}(), ReentrantLock()
    )
    _, vi = DynamicPPL.init!!(rng, model, vi, checking, UnlinkAll())
    _warn_unused_binding_names(model, checking)

    params = get_raw_values(vi)
    # This adds one evaluation per `check_model` call, not per ordinary model evaluation.
    # Turing's `sample(model, NUTS(), 1000)` checks once before sampling; MCMC steps
    # do not trigger it. `check_model=false` disables all model checks.
    provenance_ext = Base.get_extension(DynamicPPL, :DynamicPPLInputProvenanceExt)
    if !isempty(params) && provenance_ext !== nothing
        provenance_ext.check_input_provenance(rng, model, params)
    end

    # If there are no raw values, then there are no parameters. We just warn in this case.
    # (But don't `return` early, because there might be other things that are wrong with the
    # model.)
    if isempty(params)
        @warn "The model does not contain any parameters."
    end

    # Check if the DebugAccumulator found any issues with the model.
    debug_acc = DynamicPPL.getacc(vi, Val(_DEBUG_ACC_NAME))
    if debug_acc.failed
        failed = true
    end

    # Check the DebugRawValueAccumulator
    debug_raw_value_acc = DynamicPPL.getacc(vi, Val(DynamicPPL.RAW_VALUE_ACCNAME))
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
        prior_acc = DynamicPPL.getacc(vi, Val(DynamicPPL.PRIOR_ACCNAME)).values
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
        accs = DynamicPPL.VarInfo(PriorDistributionAccumulator())
        _, accs = DynamicPPL.init!!(rng, model, accs, InitFromPrior(), UnlinkAll())
        return only(DynamicPPL.getaccs(accs)).values
    end
    all_vns = mapreduce(keys, vcat, prior_vnts)
    for vn in all_vns
        # Check that the bijector for `vn` is the same across all runs. (Note that
        # the distribution can vary, as long as the bijector doesn't change)
        bijectors = map(
            vnts -> Bijectors.VectorBijectors.from_linked_vec(vnts[vn]), prior_vnts
        )
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
    gen_evaluator_call_with_types(model[, varinfo]; context)

Generate the evaluator call and the types of the arguments.

# Arguments
- `model::Model`: The model whose evaluator is of interest.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Keyword Arguments
- `context::Context`: The evaluation context. Defaults to the values supplied in `varinfo`,
  unlinked, or `InitFromPrior()` when `varinfo` has no values.

# Returns
A 2-tuple with the following elements:
- `f`: The model body function, or `Core.kwcall` if it takes keyword arguments.
    Models with argument LHS variables use prepared arguments for their body function.
- `argtypes::Type{<:Tuple}`: The types of the arguments for the evaluator.
"""
function gen_evaluator_call_with_types(
    model::Model,
    varinfo::AbstractVarInfo=VarInfo(model);
    context::Context=Context(
        if !DynamicPPL.hasacc(varinfo, Val(DynamicPPL.VECTORVAL_ACCNAME)) ||
            isempty(varinfo)
            InitFromPrior()
        else
            InitFromParams(get_values(varinfo), nothing)
        end,
        UnlinkAll(),
    ),
)
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, context, varinfo)
    f, args, kwargs = DynamicPPL._model_evaluator(model.f, args, kwargs)
    return if isempty(kwargs)
        (f, Base.typesof(args...))
    else
        (Core.kwcall, Tuple{typeof(kwargs),Core.Typeof(f),map(Core.Typeof, args)...})
    end
end

"""
    model_warntype(model[, varinfo, optimize=false]; context)

Check the type stability of the model's evaluator, warning about any potential issues.

This simply calls `@code_warntype` on the model's evaluator, filling in internal arguments where needed.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Keyword Arguments
- `context::Context`: The evaluation context. Defaults to the values supplied in `varinfo`,
  unlinked, or `InitFromPrior()` when `varinfo` has no values.
"""
function model_warntype(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model), optimize::Bool=false; kwargs...
)
    ftype, argtypes = gen_evaluator_call_with_types(model, varinfo; kwargs...)
    return InteractiveUtils.code_warntype(ftype, argtypes; optimize=optimize)
end

"""
    model_typed(model[, varinfo, optimize=true]; context)

Return the type inference for the model's evaluator.

This simply calls `@code_typed` on the model's evaluator, filling in internal arguments where needed.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Keyword Arguments
- `context::Context`: The evaluation context. Defaults to the values supplied in `varinfo`,
  unlinked, or `InitFromPrior()` when `varinfo` has no values.
"""
function model_typed(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model), optimize::Bool=true; kwargs...
)
    ftype, argtypes = gen_evaluator_call_with_types(model, varinfo; kwargs...)
    return only(InteractiveUtils.code_typed(ftype, argtypes; optimize=optimize))
end

end
