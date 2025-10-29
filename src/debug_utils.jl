module DebugUtils

using ..DynamicPPL

using Random: Random
using Accessors: Accessors
using InteractiveUtils: InteractiveUtils

using DocStringExtensions
using Distributions

export check_model, check_model_and_trace, has_static_constraints

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
    # Attempt to make the type concrete in case the symbol is shared.
    return _show_varname(io, [vn for vn in varname])
end
function _show_varname(io::IO, varname::Array{<:VarName,N}) where {N}
    # Print the first and last element of the array.
    print(io, "[")
    show_varname(io, varname[1])
    print(io, ", ..., ")
    show_varname(io, varname[end])
    print(io, "]")
    # And the size.
    print(io, " ", size(varname))

    return nothing
end
function _show_varname(io::IO, varname::Array{<:VarName{sym},N}) where {N,sym}
    print(io, sym, "[...]", " ", size(varname))
    return nothing
end

function show_right(io::IO, d::Distribution)
    pnames = fieldnames(typeof(d))
    _, namevals = Distributions._use_multline_show(d, pnames)
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
    return nothing
end

Base.@kwdef struct ObserveStmt <: Stmt
    varname
    right
    value
end

function Base.show(io::IO, stmt::ObserveStmt)
    io = add_io_context(io)
    print(io, " observe: ")
    if stmt.varname === nothing
        print(io, stmt.value)
    else
        show_varname(io, stmt.varname)
        print(io, " (= ")
        print(io, stmt.value)
        print(io, ")")
    end
    print(io, " ~ ")
    show_right(io, stmt.right)
    return nothing
end

# Some utility methods for extracting information from a trace.
"""
    varnames_in_trace(trace)

Return all the varnames present in the trace.
"""
varnames_in_trace(trace::AbstractVector) = mapreduce(varnames_in_stmt, vcat, trace)

varnames_in_stmt(stmt::AssumeStmt) = [stmt.varname]
varnames_in_stmt(::ObserveStmt) = []

function distributions_in_trace(trace::AbstractVector)
    return mapreduce(distributions_in_stmt, vcat, trace)
end

distributions_in_stmt(stmt::AssumeStmt) = [stmt.right]
distributions_in_stmt(stmt::ObserveStmt) = [stmt.right]

"""
    DebugAccumulator <: AbstractAccumulator

An accumulator which captures tilde-statements inside a model and attempts to catch
errors in the model.

# Fields
$(TYPEDFIELDS)
"""
struct DebugAccumulator <: AbstractAccumulator
    "mapping from varnames to the number of times they have been seen"
    varnames_seen::OrderedDict{VarName,Int}
    "tilde statements that have been executed"
    statements::Vector{Stmt}
    "whether to throw an error if we encounter errors in the model"
    error_on_failure::Bool
end

function DebugAccumulator(error_on_failure=false)
    return DebugAccumulator(OrderedDict{VarName,Int}(), Vector{Stmt}(), error_on_failure)
end

const _DEBUG_ACC_NAME = :Debug
DynamicPPL.accumulator_name(::Type{<:DebugAccumulator}) = _DEBUG_ACC_NAME

function Base.:(==)(acc1::DebugAccumulator, acc2::DebugAccumulator)
    return (
        acc1.varnames_seen == acc2.varnames_seen &&
        acc1.statements == acc2.statements &&
        acc1.error_on_failure == acc2.error_on_failure
    )
end

function _zero(acc::DebugAccumulator)
    return DebugAccumulator(
        OrderedDict{VarName,Int}(), Vector{Stmt}(), acc.error_on_failure
    )
end
DynamicPPL.reset(acc::DebugAccumulator) = _zero(acc)
DynamicPPL.split(acc::DebugAccumulator) = _zero(acc)
function DynamicPPL.combine(acc1::DebugAccumulator, acc2::DebugAccumulator)
    return DebugAccumulator(
        merge(acc1.varnames_seen, acc2.varnames_seen),
        vcat(acc1.statements, acc2.statements),
        acc1.error_on_failure || acc2.error_on_failure,
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
    acc::DebugAccumulator, val, _logjac, vn::VarName, right::Distribution
)
    record_varname!(acc, vn, right)
    stmt = AssumeStmt(; varname=vn, right=right, value=val)
    push!(acc.statements, stmt)
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
    stmt = ObserveStmt(; varname=vn, right=right, value=val)
    push!(acc.statements, stmt)
    return acc
end

_conditioned_varnames(d::AbstractDict) = keys(d)
_conditioned_varnames(d) = map(sym -> VarName{sym}(), keys(d))
function conditioned_varnames(context)
    conditioned_values = DynamicPPL.conditioned(context)
    return _conditioned_varnames(conditioned_values)
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

function check_model_post_evaluation(acc::DebugAccumulator)
    return check_varnames_seen(acc.varnames_seen)
end

"""
    check_model_and_trace(model::Model, varinfo::AbstractVarInfo; error_on_failure=false)

Check that evaluating `model` with the given `varinfo` is valid, warning about any potential
issues.

This will check the model for the following issues:

1. Repeated usage of the same varname in a model.
2. `NaN` on the left-hand side of observe statements.

# Arguments
- `model::Model`: The model to check.
- `varinfo::AbstractVarInfo`: The varinfo to use when evaluating the model.

# Keyword Argument
- `error_on_failure::Bool`: Whether to throw an error if the model check fails. Default: `false`.

# Returns
- `issuccess::Bool`: Whether the model check succeeded.
- `trace::Vector{Stmt}`: The trace of statements executed during the model check.

# Examples
## Correct model

```jldoctest check-model-and-tracecheck-model-and-trace; setup=:(using Distributions)
julia> using StableRNGs

julia> rng = StableRNG(42);

julia> @model demo_correct() = x ~ Normal()
demo_correct (generic function with 2 methods)

julia> model = demo_correct(); varinfo = VarInfo(rng, model);

julia> issuccess, trace = check_model_and_trace(model, varinfo);

julia> issuccess
true

julia> print(trace)
 assume: x ~ Normal{Float64}(μ=0.0, σ=1.0) ⟼ -0.670252

julia> cond_model = model | (x = 1.0,);

julia> issuccess, trace = check_model_and_trace(cond_model, VarInfo(cond_model));
┌ Warning: The model does not contain any parameters.
└ @ DynamicPPL.DebugUtils DynamicPPL.jl/src/debug_utils.jl:342

julia> issuccess
true

julia> print(trace)
 observe: x (= 1.0) ~ Normal{Float64}(μ=0.0, σ=1.0)
```

## Incorrect model

```jldoctest check-model-and-tracecheck-model-and-trace; setup=:(using Distributions)
julia> @model function demo_incorrect()
           # (×) Sampling `x` twice will lead to incorrect log-probabilities!
           x ~ Normal()
           x ~ Exponential()
       end
demo_incorrect (generic function with 2 methods)

julia> # Notice that VarInfo(model_incorrect) evaluates the model, but doesn't actually
       # alert us to the issue of `x` being sampled twice.
       model = demo_incorrect(); varinfo = VarInfo(model);

julia> issuccess, trace = check_model_and_trace(model, varinfo; error_on_failure=true);
ERROR: varname x used multiple times in model
```
"""
function check_model_and_trace(
    model::Model, varinfo::AbstractVarInfo; error_on_failure=false
)
    # Add debug accumulator to the VarInfo.
    varinfo = DynamicPPL.setaccs!!(deepcopy(varinfo), (DebugAccumulator(error_on_failure),))

    # Perform checks before evaluating the model.
    issuccess = check_model_pre_evaluation(model)

    # Force single-threaded execution.
    _, varinfo = DynamicPPL.evaluate_threadunsafe!!(model, varinfo)

    # Perform checks after evaluating the model.
    debug_acc = DynamicPPL.getacc(varinfo, Val(_DEBUG_ACC_NAME))
    issuccess = issuccess && check_model_post_evaluation(debug_acc)

    if !issuccess && error_on_failure
        error("model check failed")
    end

    trace = debug_acc.statements
    return issuccess, trace
end

"""
    check_model(model::Model, varinfo::AbstractVarInfo; error_on_failure=false)

Check that `model` is valid, warning about any potential issues (or erroring if
`error_on_failure` is `true`).

# Returns
- `issuccess::Bool`: Whether the model check succeeded.
"""
check_model(model::Model, varinfo::AbstractVarInfo; error_on_failure=false) =
    first(check_model_and_trace(model, varinfo; error_on_failure=error_on_failure))

# Convenience method used to check if all elements in a list are the same.
function all_the_same(xs)
    issuccess = true
    for i in 2:length(xs)
        if xs[1] != xs[i]
            issuccess = false
            break
        end
    end

    return issuccess
end

"""
    has_static_constraints([rng, ]model::Model; num_evals=5, error_on_failure=false)

Return `true` if the model has static constraints, `false` otherwise.

Note that this is a heuristic check based on sampling from the model multiple times
and checking if the model is consistent across runs.

# Arguments
- `rng::Random.AbstractRNG`: The random number generator to use when evaluating the model.
- `model::Model`: The model to check.

# Keyword Arguments
- `num_evals::Int`: The number of evaluations to perform. Default: `5`.
- `error_on_failure::Bool`: Whether to throw an error if any of the `num_evals` model
  checks fail. Default: `false`.
"""
function has_static_constraints(
    rng::Random.AbstractRNG, model::Model; num_evals::Int=5, error_on_failure::Bool=false
)
    new_model = DynamicPPL.contextualize(model, InitContext(rng))
    results = map(1:num_evals) do _
        check_model_and_trace(new_model, VarInfo(); error_on_failure=error_on_failure)
    end

    # Extract the distributions and the corresponding bijectors for each run.
    traces = map(last, results)
    dists_per_trace = map(distributions_in_trace, traces)
    transforms = map(dists_per_trace) do dists
        map(DynamicPPL.link_transform, dists)
    end

    # Check if the distributions are the same across all runs.
    return all_the_same(transforms)
end
function has_static_constraints(
    model::Model; num_evals::Int=5, error_on_failure::Bool=false
)
    return has_static_constraints(
        Random.default_rng(), model; num_evals=num_evals, error_on_failure=error_on_failure
    )
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
