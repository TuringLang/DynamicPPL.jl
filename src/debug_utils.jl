module DebugUtils

using ..DynamicPPL
using ..DynamicPPL: broadcast_safe, AbstractContext, childcontext

using Setfield: Setfield

using DocStringExtensions
using Distributions

export check_model, check_model_and_trace, DebugContext

# Statements
abstract type Stmt end

function Base.show(io::IO, statements::Vector{Stmt})
    for stmt in statements
        println(io, stmt)
    end
end

const RESULT_SYMBOL = "⟼"

add_io_context(io::IO) = IOContext(io, :compact => true, :limit => true)

show_varname(io::IO, varname::VarName) = print(io, varname)
function show_varname(io::IO, varname::Array{<:VarName,N}) where {N}
    # TODO: Can we remove the `VarName` at the beginning of the show completely?
    return show(IOContext(io, :typeinfo => VarName), convert(Array{VarName,N}, varname))
end

function show_right(io::IO, d::Distribution)
    pnames = fieldnames(typeof(d))
    uml, namevals = Distributions._use_multline_show(d, pnames)
    return Distributions.show_oneline(io, d, namevals)
end

function show_right(io::IO, d::Distributions.ReshapedDistribution)
    print(io, "reshape(")
    show_right(io, d.dist)
    return print(io, ")")
end

function show_right(io::IO, d::Distributions.Product)
    print(io, "product(")
    for (i, dist) in enumerate(d.v)
        if i > 1
            print(io, ", ")
        end
        show_right(io, dist)
    end
    return print(io, ")")
end

show_right(io::IO, d) = show(io, d)

Base.@kwdef struct AssumeStmt <: Stmt
    varname
    right
    value
    logp
    varinfo = nothing
end

function Base.show(io::IO, stmt::AssumeStmt)
    io = add_io_context(io)
    print(io, " assume: ")
    show_varname(io, stmt.varname)
    print(io, " ~ ")
    show_right(io, stmt.right)
    print(io, " ")
    print(io, RESULT_SYMBOL)
    print(io, " ")
    print(io, stmt.value)
    print(io, " (logprob = ")
    print(io, stmt.logp)
    return print(io, ")")
end

Base.@kwdef struct ObserveStmt <: Stmt
    left
    right
    logp
    varinfo = nothing
end

function Base.show(io::IO, stmt::ObserveStmt)
    io = add_io_context(io)
    print(io, "observe: ")
    show_right(io, stmt.left)
    print(io, " ~ ")
    show_right(io, stmt.right)
    print(io, " (logprob = ")
    print(io, stmt.logp)
    return print(io, ")")
end

Base.@kwdef struct DotAssumeStmt <: Stmt
    varname
    left
    right
    value
    logp
    varinfo = nothing
end

function Base.show(io::IO, stmt::DotAssumeStmt)
    io = add_io_context(io)
    print(io, " assume: ")
    show_varname(io, stmt.varname)
    print(io, " = ")
    print(io, stmt.left)
    print(io, " .~ ")
    show_right(io, stmt.right)
    print(io, " ")
    print(io, RESULT_SYMBOL)
    print(io, " ")
    print(io, stmt.value)
    print(io, " (logprob = ")
    print(io, stmt.logp)
    return print(io, ")")
end

Base.@kwdef struct DotObserveStmt <: Stmt
    left
    right
    logp
    varinfo = nothing
end

function Base.show(io::IO, stmt::DotObserveStmt)
    io = add_io_context(io)
    print(io, "observe: ")
    print(io, stmt.left)
    print(io, " .~ ")
    show_right(io, stmt.right)
    print(io, " ")
    print(io, RESULT_SYMBOL)
    print(io, " (logprob = ")
    print(io, stmt.logp)
    return print(io, ")")
end

"""
    DebugContext <: AbstractContext

A context used for checking validity of a model.

This context is used by [`check_model`](@ref) to check that a model is valid.

# Fields
$(FIELDS)
"""
struct DebugContext{M<:Model,C<:AbstractContext} <: AbstractContext
    "model that is being run"
    model::M
    "context used for running the model"
    context::C
    "mapping from varnames to the number of times they have been seen"
    varnames_seen::OrderedDict{VarName,Int}
    "tilde statements that have been executed"
    statements::Vector{Stmt}
    "whether to throw an error if we encounter warnings"
    error_on_failure::Bool
    "whether to record the tilde statements"
    record_statements::Bool
    "whether to record the varinfo in every tilde statement"
    record_varinfo::Bool
end

function DebugContext(
    model::Model,
    context::AbstractContext=DefaultContext();
    varnames_seen=OrderedDict{VarName,Int}(),
    statements=Vector{Stmt}(),
    error_on_failure=false,
    record_statements=true,
    record_varinfo=false,
)
    return DebugContext(
        model,
        context,
        varnames_seen,
        statements,
        error_on_failure,
        record_statements,
        record_varinfo,
    )
end

DynamicPPL.NodeTrait(::DebugContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::DebugContext) = context.context
function DynamicPPL.setchildcontext(context::DebugContext, child)
    Setfield.@set context.context = child
end

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

# assume
function record_pre_tilde_assume!(context::DebugContext, vn, dist, varinfo)
    record_varname!(context, vn, dist)
    return nothing
end

function record_post_tilde_assume!(context::DebugContext, vn, dist, value, logp, varinfo)
    stmt = AssumeStmt(;
        varname=vn,
        right=dist,
        value=value,
        logp=logp,
        varinfo=context.record_varinfo ? varinfo : nothing,
    )
    if context.record_statements
        push!(context.statements, stmt)
    end
    return nothing
end

function DynamicPPL.tilde_assume(context::DebugContext, right, vn, vi)
    record_pre_tilde_assume!(context, vn, right, vi)
    value, logp, vi = DynamicPPL.tilde_assume(childcontext(context), right, vn, vi)
    record_post_tilde_assume!(context, vn, right, value, logp, vi)
    return value, logp, vi
end
function DynamicPPL.tilde_assume(rng, context::DebugContext, sampler, right, vn, vi)
    record_pre_tilde_assume!(context, vn, right, vi)
    value, logp, vi = DynamicPPL.tilde_assume(
        rng, childcontext(context), sampler, right, vn, vi
    )
    record_post_tilde_assume!(context, vn, right, value, logp, vi)
    return value, logp, vi
end

# observe
function record_pre_tilde_observe!(context::DebugContext, left, dist, varinfo)
    # Check for `missing`s; these should not end up here.
    if _has_missings(left)
        error(
            "Encountered missing value(s) in observe!\n" *
            "Remember that using `missing` to de-condition a variable is only " *
            "supported for univariate distributions, not for $dist",
        )
    end
end

function record_post_tilde_observe!(context::DebugContext, left, right, logp, varinfo)
    stmt = ObserveStmt(;
        left=left,
        right=right,
        logp=logp,
        varinfo=context.record_varinfo ? varinfo : nothing,
    )
    if context.record_statements
        push!(context.statements, stmt)
    end
    return nothing
end

function DynamicPPL.tilde_observe(context::DebugContext, right, left, vi)
    record_pre_tilde_observe!(context, left, right, vi)
    logp, vi = DynamicPPL.tilde_observe(childcontext(context), right, left, vi)
    record_post_tilde_observe!(context, left, right, logp, vi)
    return logp, vi
end
function DynamicPPL.tilde_observe(context::DebugContext, sampler, right, left, vi)
    record_pre_tilde_observe!(context, left, right, vi)
    logp, vi = DynamicPPL.tilde_observe(childcontext(context), sampler, right, left, vi)
    record_post_tilde_observe!(context, left, right, logp, vi)
    return logp, vi
end

# dot-assume
function record_pre_dot_tilde_assume!(context::DebugContext, vn, left, right, varinfo)
    # Check for `missing`s; these should not end up here.
    if _has_missings(left)
        error(
            "Variable $(vn) has missing has missing value(s)!\n" *
            "Usage of `missing` is not supported for dotted syntax, such as " *
            "`@. x ~ dist` or `x .~ dist`",
        )
    end

    # TODO: Can we do without the memory allocation here?
    record_varname!.(broadcast_safe(context), vn, broadcast_safe(right))

    # Check that `left` does not contain any ``
    return nothing
end

function record_post_dot_tilde_assume!(
    context::DebugContext, vns, left, right, value, logp, varinfo
)
    stmt = DotAssumeStmt(;
        varname=vns,
        left=left,
        right=right,
        value=value,
        logp=logp,
        varinfo=context.record_varinfo ? deepcopy(varinfo) : nothing,
    )
    if context.record_statements
        push!(context.statements, stmt)
    end

    return nothing
end

function DynamicPPL.dot_tilde_assume(context::DebugContext, right, left, vn, vi)
    record_pre_dot_tilde_assume!(context, vn, left, right, vi)
    value, logp, vi = DynamicPPL.dot_tilde_assume(
        childcontext(context), right, left, vn, vi
    )
    record_post_dot_tilde_assume!(context, vn, left, right, value, logp, vi)
    return value, logp, vi
end

function DynamicPPL.dot_tilde_assume(
    rng, context::DebugContext, sampler, right, left, vn, vi
)
    record_pre_dot_tilde_assume!(context, vn, left, right, vi)
    value, logp, vi = DynamicPPL.dot_tilde_assume(
        rng, childcontext(context), sampler, right, left, vn, vi
    )
    record_post_dot_tilde_assume!(context, vn, left, right, value, logp, vi)
    return value, logp, vi
end

# dot-observe
function record_pre_dot_tilde_observe!(context::DebugContext, left, right, vi)
    # Check for `missing`s; these should not end up here.
    if _has_missings(left)
        # TODO: Once `observe` statements receive `vn`, refer to this in the
        # error message.
        error(
            "Encountered missing value(s) in observe!\n" *
            "Usage of `missing` is not supported for dotted syntax, such as " *
            "`@. x ~ dist` or `x .~ dist`",
        )
    end
end

function record_post_dot_tilde_observe!(context::DebugContext, left, right, logp, vi)
    stmt = DotObserveStmt(;
        left=left,
        right=right,
        logp=logp,
        varinfo=context.record_varinfo ? deepcopy(vi) : nothing,
    )
    if context.record_statements
        push!(context.statements, stmt)
    end
    return nothing
end
function DynamicPPL.dot_tilde_observe(context::DebugContext, right, left, vi)
    record_pre_dot_tilde_observe!(context, left, right, vi)
    logp, vi = DynamicPPL.dot_tilde_observe(childcontext(context), right, left, vi)
    record_post_dot_tilde_observe!(context, left, right, logp, vi)
    return logp, vi
end

_conditioned_varnames(d::AbstractDict) = keys(d)
_conditioned_varnames(d) = map(sym -> VarName{sym}(), keys(d))
function conditioned_varnames(context)
    conditioned_values = DynamicPPL.conditioned(context)
    return _conditioned_varnames(conditioned_values)
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

# A check we run on the model before evaluating it.
function check_model_pre_evaluation(context::DebugContext, model::Model)
    issuccess = true
    # If something is in the model arguments, then it should NOT be in `condition`,
    # nor should there be any symbol present in `condition` that has the same symbol.
    for vn in conditioned_varnames(model.context)
        if DynamicPPL.inargnames(vn, model)
            @warn "Variable $(vn) is both in the model arguments and in the conditioning!\n" *
                "Please use either conditioning through the model arguments, or through " *
                "`condition` / `|`, not both."

            issuccess = false
        end
    end

    return issuccess
end

function check_model_post_evaluation(context::DebugContext, model::Model)
    return check_varnames_seen(context.varnames_seen)
end

"""
    check_model_and_trace(model::Model; kwargs...)

Check that `model` is valid, warning about any potential issues.

This will check the model for the following issues:
1. Repeated usage of the same varname in a model.
2. Incorrectly treating a variable as random rather than fixed, and vice versa.

# Arguments
- `model::Model`: The model to check.

# Keyword Arguments
- `varinfo::VarInfo`: The varinfo to use when evaluating the model. Default: `VarInfo(model)`.
- `context::AbstractContext`: The context to use when evaluating the model. Default: [`DefaultContext`](@ref).
- `error_on_failure::Bool`: Whether to throw an error if the model check fails. Default: `false`.

# Returns
- `issuccess::Bool`: Whether the model check succeeded.
- `trace::Vector{Stmt}`: The trace of statements executed during the model check.
"""
function check_model_and_trace(
    model::Model;
    varinfo=VarInfo(),
    context=SamplingContext(),
    error_on_failure=false,
    kwargs...,
)
    # Execute the model with the debug context.
    debug_context = DebugContext(
        model, context; error_on_failure=error_on_failure, kwargs...
    )

    # Perform checks before evaluating the model.
    issuccess = check_model_pre_evaluation(debug_context, model)

    # Force single-threaded execution.
    retval, varinfo_result = DynamicPPL.evaluate_threadunsafe!!(
        model, varinfo, debug_context
    )

    # Perform checks after evaluating the model.
    issuccess &= check_model_post_evaluation(debug_context, model)

    if !issuccess && error_on_failure
        error("model check failed")
    end

    trace = debug_context.statements
    return issuccess, trace
end

"""
    check_model(model::Model; kwargs...)

Check that `model` is valid, warning about any potential issues.

See [`check_model_and_trace`](@ref) for more details on supported keword arguments
and details of which types of checks are performed.

# Returns
- `issuccess::Bool`: Whether the model check succeeded.
"""
check_model(model::Model; kwargs...) = first(check_model_and_trace(model; kwargs...))

end
