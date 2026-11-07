module DebugUtils

using ..DynamicPPL

using Bijectors: Bijectors
using Random: Random
using InteractiveUtils: InteractiveUtils
using Distributions

export check_model, has_static_constraints

# Fixed bindings bypass accumulators, so the checker records reads before tilde dispatch.
struct BindingCheckContext{C<:AbstractContext} <: DynamicPPL.AbstractParentContext
    context::C
    models::Vector{Tuple{Model,Vector{VarName}}}
    removals::Vector{Tuple{Union{VarName,Nothing},DataType,Bool}}
    removal_ids::Tuple{Vector{Int},Vector{Int}}
    removal_sources::IdDict{Model,Tuple{Vector{Int},Vector{Int}}}
    reads::Tuple
    lock::ReentrantLock
end
function BindingCheckContext(context)
    return BindingCheckContext(
        context,
        Tuple{Model,Vector{VarName}}[],
        Tuple{Union{VarName,Nothing},DataType,Bool}[],
        (Int[], Int[]),
        IdDict{Model,Tuple{Vector{Int},Vector{Int}}}(),
        (VarName[],),
        ReentrantLock(),
    )
end
DynamicPPL.childcontext(ctx::BindingCheckContext) = ctx.context
function DynamicPPL.setchildcontext(ctx::BindingCheckContext, child::AbstractContext)
    return DynamicPPL.ConstructionBase.setproperties(ctx, (; context=child))
end
function _register_removals!(ctx, model, prefix)
    return get!(ctx.removal_sources, model) do
        return map((DynamicPPL.ConditionRole, DynamicPPL.FixRole)) do role
            ids = Int[]
            for marker in DynamicPPL._removals(role, model)
                name = if marker.name === nothing
                    nothing
                else
                    DynamicPPL.maybe_prefix(marker.name, prefix)
                end
                push!(ctx.removals, (name, role, marker.matched))
                push!(ids, length(ctx.removals))
            end
            ids
        end
    end
end
function DynamicPPL._binding_check_child(
    ctx::BindingCheckContext, parent, child, prefix, template
)
    ids = lock(ctx.lock) do
        own_ids = _register_removals!(ctx, child, DynamicPPL._model_prefix(child))
        table = child.values
        arguments = merge(child.args, child.defaults)
        map(
            (DynamicPPL.ConditionRole, DynamicPPL.FixRole), own_ids, ctx.removal_ids
        ) do role, own, inherited
            markers = DynamicPPL._child_removals(
                DynamicPPL._removals(role, parent), prefix, template
            )
            for (id, marker) in zip(inherited, markers)
                if marker.active && DynamicPPL._removal_matches(table, role, marker.name)
                    name, role, _ = ctx.removals[id]
                    ctx.removals[id] = (name, role, true)
                end
                table = DynamicPPL._apply_removals(table, role, (marker,), arguments)
            end
            vcat(own, inherited)
        end
    end
    return DynamicPPL.ConstructionBase.setproperties(
        ctx, (; removal_ids=ids, reads=(ctx.reads..., VarName[]))
    )
end
function DynamicPPL._binding_check_model(ctx::BindingCheckContext, model)
    lock(ctx.lock) do
        push!(ctx.models, (model, last(ctx.reads)))
    end
    return nothing
end
function DynamicPPL._binding_check_read(ctx::BindingCheckContext, model, vn, template)
    vn = DynamicPPL._concretize_prefix(vn, template; canonical=true)
    vn = DynamicPPL.maybe_prefix(vn, DynamicPPL._model_prefix(model))
    lock(ctx.lock) do
        for reads in ctx.reads
            push!(reads, vn)
        end
    end
    return nothing
end
function _warn_unused_removals(ctx)
    for (name, role, used) in ctx.removals
        (name === nothing || used) && continue
        layer = role === DynamicPPL.ConditionRole ? "observations" : "fixed bindings"
        @warn "Removal of $layer at `$name` is unused by any reached model."
    end
    return nothing
end
function _warn_unread_bindings(ctx)
    unread = Set{VarName}()
    for (model, reads) in ctx.models
        roots = model.values.roots.data
        names = Tuple(
            n for n in keys(roots) if
            roots[n].value isa Union{VarNamedTuple,DynamicPPL.VarNamedTuples.PartialArray}
        )
        table = DynamicPPL.BindingTable(VarNamedTuple(NamedTuple{names}(roots)))
        prefix = DynamicPPL._model_prefix(model)
        local_reads = Tuple(
            prefix === nothing ? vn : DynamicPPL.AbstractPPL.unprefix(vn, prefix) for
            vn in reads if prefix === nothing || subsumes(prefix, vn)
        )
        if !isempty(local_reads)
            for role in (DynamicPPL.ConditionRole, DynamicPPL.FixRole)
                table = DynamicPPL._remove_table(table, role, local_reads; validate=false)
            end
        end
        for (name, root) in pairs(table.roots.data)
            entries = Tuple[]
            DynamicPPL._binding_entries!(entries, root.value, VarName{name}(), root.storage)
            for (vn, value, _) in entries
                value isa DynamicPPL.BoundValue || continue
                push!(unread, DynamicPPL.maybe_prefix(vn, prefix))
            end
        end
    end
    for vn in sort!(collect(unread); by=string)
        @warn "Partial binding `$vn` is not read by any executed tilde."
    end
    return nothing
end
function _warn_unused_templates(ctx)
    entries, lhs = Set{VarName}(), Set{VarName}()
    for (model, _) in ctx.models
        metadata = DynamicPPL._binding_metadata(model)
        DynamicPPL._lhs_names(metadata) === nothing && continue
        prefix = DynamicPPL._model_prefix(model)
        for name in DynamicPPL._binding_template_names(metadata)
            push!(entries, DynamicPPL.maybe_prefix(name, prefix))
        end
        for (name, submodel) in DynamicPPL._lhs_addresses(metadata)
            submodel || push!(lhs, DynamicPPL.maybe_prefix(name, prefix))
        end
    end
    for name in sort!(collect(entries); by=string)
        if !any(root -> subsumes(root, name) || subsumes(name, root), lhs)
            @warn "Binding template entry `$name` has no LHS variable in this model or any reached submodel. It may be unused; a submodel in an untaken branch could still use it."
        end
    end
    return nothing
end

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
    acc::DebugAccumulator, right::Distribution, val, vn::Union{VarName,Nothing}, template
)
    failed = acc.failed
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
        failed = true
    end
    # Check for NaN's as well
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

- Unused named removals, unread partial bindings and unused binding template entries emit
  warnings without failing the check. An untaken branch may still use them.

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
    init_strategy = InitFromPrior()
    binding_context = BindingCheckContext(model.context)
    ids = _register_removals!(binding_context, model, DynamicPPL._model_prefix(model))
    binding_context = DynamicPPL.ConstructionBase.setproperties(
        binding_context, (; removal_ids=ids)
    )
    checked_model = DynamicPPL.contextualize(model, binding_context)
    _, vi = DynamicPPL.init!!(rng, checked_model, vi, init_strategy, UnlinkAll())
    _warn_unused_removals(binding_context)
    _warn_unread_bindings(binding_context)
    _warn_unused_templates(binding_context)

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
    gen_evaluator_call_with_types(model[, varinfo])

Generate the evaluator call and the types of the arguments.

# Arguments
- `model::Model`: The model whose evaluator is of interest.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Returns
A 2-tuple with the following elements:
- `f`: The model body function, or `Core.kwcall` when keyword arguments are passed.
    For `@model` evaluators with argument LHS variables, the inner body receives
    prepared arguments positionally.
- `argtypes::Type{<:Tuple}`: The types of the arguments for the evaluator.
"""
function gen_evaluator_call_with_types(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model)
)
    strategy =
        if !DynamicPPL.hasacc(varinfo, Val(DynamicPPL.VECTORVAL_ACCNAME)) ||
            isempty(varinfo)
            InitFromPrior()
        else
            InitFromParams(get_values(varinfo), nothing)
        end
    context = InitContext(strategy, UnlinkAll())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(
        setleafcontext(model, context), varinfo
    )
    f, args, kwargs = DynamicPPL._model_evaluator(model.f, args, kwargs)
    return if isempty(kwargs)
        (f, Base.typesof(args...))
    else
        (Core.kwcall, Tuple{typeof(kwargs),Core.Typeof(f),map(Core.Typeof, args)...})
    end
end

"""
    model_warntype(model[, varinfo, optimize=false])

Check the type stability of the model's evaluator, warning about any potential issues.

This simply calls `@code_warntype` on the model's evaluator, filling in internal arguments where needed.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.
"""
function model_warntype(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model), optimize::Bool=false
)
    ftype, argtypes = gen_evaluator_call_with_types(model, varinfo)
    return InteractiveUtils.code_warntype(ftype, argtypes; optimize=optimize)
end

"""
    model_typed(model[, varinfo, optimize=true])

Return the type inference for the model's evaluator.

This simply calls `@code_typed` on the model's evaluator, filling in internal arguments where needed.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.
"""
function model_typed(
    model::Model, varinfo::AbstractVarInfo=VarInfo(model), optimize::Bool=true
)
    ftype, argtypes = gen_evaluator_call_with_types(model, varinfo)
    return only(InteractiveUtils.code_typed(ftype, argtypes; optimize=optimize))
end

end
