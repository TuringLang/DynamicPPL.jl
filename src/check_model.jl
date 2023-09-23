"""
    DebugContext <: AbstractContext

A context used for checking validity of a model.

This context is used by [`check_model`](@ref) to check that a model is valid.
"""
struct DebugContext{M<:Model,C<:AbstractContext} <: AbstractContext
    model::M
    context::C
    varnames_seen::OrderedDict{VarName,Int}
    tildes_seen::Vector{Any}
    error_on_failure::Bool
    record_varinfo::Bool
end

function DebugContext(
    model::Model,
    context::AbstractContext=DefaultContext();
    varnames_seen=OrderedDict{VarName,Int}(),
    tildes_seen=Vector{Any}(),
    error_on_failure=false,
    record_varinfo=false,
)
    return DebugContext(
        model,
        context,
        varnames_seen,
        tildes_seen,
        error_on_failure,
        record_varinfo
    )
end

NodeTrait(::DebugContext) = IsParent()
childcontext(context::DebugContext) = context.context
setchildcontext(context::DebugContext, child) = Setfield.@set context.context = child

function record_varname!(context::DebugContext, varname::VarName, dist)
    if haskey(context.varnames_seen, varname)
        if context.error_on_failure
            error("varname $varname used multiple times in model")
        else
            @warn "varname $varname used multiple times in model"
        end
        context.varnames_seen[varname] += 1
    else
        context.varnames_seen[varname] = 1
    end
end

# tilde
_has_missings(x) = ismissing(x)
_has_missings(x::AbstractArray) = any(ismissing, x)

function record_pre_tilde_assume!(context::DebugContext, vn, dist, varinfo)
    record_varname!(context, vn, dist)
    return nothing
end

function record_post_tilde_assume!(context::DebugContext, vn, dist, value, logp, varinfo)
    record = (varname=vn, right=dist, value=value, logp=logp)
    if context.record_varinfo
        record = merge(record, (varinfo=deepcopy(varinfo),))
    end
    push!(context.tildes_seen, record)
    return nothing
end

function record_pre_tilde_observe!(context::DebugContext, vn, dist, varinfo) end

function record_post_tilde_observe!(context::DebugContext, vn, dist, logp, varinfo)
end

# dot-tilde

function record_pre_dot_tilde_assume!(context::DebugContext, vn, left, right, varinfo)
    # TODO: Can we do without the memory allocation here?
    splat(record_varname!).(tuple.(broadcast_safe(context), vn, broadcast_safe(right)))

    # Check for `missing`s; these should not end up here.
    if _has_missings(value)
        error(
            "Variable $(vn) has missing has missing value(s)!\n" *
            "This is not supported for syntax dotted syntax, such as " *
            "`@. x ~ dist` or `x .~ dist`"
        )
    end

    # Check that `left` does not contain any ``
    return nothing
end

function record_post_dot_tilde_assume!(
    context::DebugContext, vns, left, right, value, logp, varinfo
)
    record = (varname=vns, left=left, right=right, value=value, logp=logp)
    if context.record_varinfo
        record = merge(record, (varinfo=deepcopy(varinfo),))
    end
    push!(context.tildes_seen, record)

    return nothing
end

function record_pre_dot_tilde_observe!(context::DebugContext, left, right, vi)
    record = (left=left, right=right)
    push!(context.tildes_seen, record)

    return nothing
end

function record_post_dot_tilde_observe!(context::DebugContext, left, right, logp, vi)
    return nothing
end

# Tilde-implementations
# tilde
function tilde_assume(context::DebugContext, right, vn, vi)
    record_pre_tilde_assume!(context, vn, right, vi)
    value, logp, vi = tilde_assume(childcontext(context), right, vn, vi)
    record_post_tilde_assume!(context, vn, right, value, logp, vi)
    return value, logp, vi
end
function tilde_assume(rng, context::DebugContext, sampler, right, vn, vi)
    record_pre_tilde_assume!(context, vn, right, vi)
    value, logp, vi = tilde_assume(rng, childcontext(context), sampler, right, vn, vi)
    record_post_tilde_assume!(context, vn, right, value, logp, vi)
    return value, logp, vi
end

function tilde_observe(context::DebugContext, right, left, vi)
    record_pre_tilde_observe!(context, left, right, vi)
    logp, vi = tilde_observe(childcontext(context), right, left, vi)
    record_post_tilde_observe!(context, left, right, logp, vi)
    return logp, vi
end

# dot-tilde
function dot_tilde_assume(context::DebugContext, right, left, vn, vi)
    record_pre_dot_tilde_assume!(context, vn, left, right, vi)
    value, logp, vi = dot_tilde_assume(
        childcontext(context), right, left, vn, vi
    )
    record_post_dot_tilde_assume!(context, vn, left, right, value, logp, vi)
    return value, logp, vi
end

function dot_tilde_assume(rng, context::DebugContext, sampler, right, left, vn, vi)
    record_pre_dot_tilde_assume!(context, vn, left, right, vi)
    value, logp, vi = dot_tilde_assume(
        rng, childcontext(context), sampler, right, left, vn, vi
    )
    record_post_dot_tilde_assume!(context, vn, left, right, value, logp, vi)
    return value, logp, vi
end

function dot_tilde_observe(context::DebugContext, right, left, vi)
    record_pre_dot_tilde_observe!(context, left, right, vi)
    logp, vi = dot_tilde_observe(childcontext(context), right, left, vi)
    record_post_dot_tilde_observe!(context, left, right, logp, vi)
    return logp, vi
end

# A check we run on the model before evaluating it.
function _check_model_pre_evaluation(context::DebugContext, model::Model)
    
end


"""
    check_model(model::Model[, varinfo]; context=DefaultContext(), error_on_failure=false)

Check that `model` is valid, warning about any potential issues.

This will check the model for the following issues:
1. Repeated usage of the same varname in a model.
2. Incorrectly treating a variable as random rather than fixed, and vice versa.

# Arguments
- `model::Model`: The model to check.
- `varinfo::VarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.

# Keyword Arguments
- `context::AbstractContext`: The context to use when evaluating the model. Default: [`DefaultContext`](@ref).
- `error_on_failure::Bool`: Whether to throw an error if the model check fails. Default: `false`.

# Returns
- `trace::Vector{Any}`: The trace of the model.
- `issuccess::Bool`: Whether the model check succeeded.
"""
function check_model(
    model::Model,
    varinfo=VarInfo(model);
    context=DefaultContext(),
    error_on_failure=false,
    kwargs...
)
    # Execute the model with the debug context.
    debug_context = DebugContext(model, context; error_on_failure=error_on_failure, kwargs...)
    retval, varinfo_result = DynamicPPL.evaluate!!(model, varinfo, debug_context)
    # Verify that the number of times we've seen each varname is sensible.
    issuccess = check_varnames_seen(debug_context.varnames_seen)

    if !issuccess && error_on_failure
        error("model check failed")
    end

    trace = debug_context.tildes_seen
    return issuccess, (trace=trace, varnames_seen=debug_context.varnames_seen)
end

function check_varnames_seen(varnames_seen::AbstractDict{VarName,Int})
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

# Special behaviors.
# 1. Chcek that we're not sampling the same variable twice.

# 2. Heuristic checks to see if we're sampling something that the
# user intended to be fixed or conditioned.
