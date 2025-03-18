"""
    TrackedValue{T}

A struct that wraps something on the right-hand side of `:=`. This is needed
because the DynamicPPL compiler actually converts `lhs := rhs` to `lhs ~
TrackedValue(rhs)` (so that we can hit the `tilde_assume` method below). Having
the rhs wrapped in a TrackedValue makes sure that the logpdf of the rhs is not
computed (as it wouldn't make sense).
"""
struct TrackedValue{T}
    value::T
end

is_tracked_value(::TrackedValue) = true
is_tracked_value(::Any) = false

check_tilde_rhs(x::TrackedValue) = x

"""
    ValuesAsInModelContext

A context that is used by [`values_as_in_model`](@ref) to obtain values
of the model parameters as they are in the model.

This is particularly useful when working in unconstrained space, but one
wants to extract the realization of a model in a constrained space.

# Fields
$(TYPEDFIELDS)
"""
struct ValuesAsInModelContext{C<:AbstractContext} <: AbstractContext
    "values that are extracted from the model"
    values::OrderedDict
    "whether to extract variables on the LHS of :="
    include_colon_eq::Bool
    "varnames to be tracked; `nothing` means track all varnames"
    tracked_varnames::Union{Nothing,Array{<:VarName}}
    "child context"
    context::C
end
function ValuesAsInModelContext(
    include_colon_eq::Bool,
    tracked_varnames::Union{Nothing,Array{<:VarName}},
    context::AbstractContext,
)
    return ValuesAsInModelContext(
        OrderedDict(), include_colon_eq, tracked_varnames, context
    )
end

NodeTrait(::ValuesAsInModelContext) = IsParent()
childcontext(context::ValuesAsInModelContext) = context.context
function setchildcontext(context::ValuesAsInModelContext, child)
    return ValuesAsInModelContext(
        context.values, context.include_colon_eq, context.tracked_varnames, child
    )
end

is_extracting_values(context::ValuesAsInModelContext) = context.include_colon_eq
function is_extracting_values(context::AbstractContext)
    return is_extracting_values(NodeTrait(context), context)
end
is_extracting_values(::IsParent, ::AbstractContext) = false
is_extracting_values(::IsLeaf, ::AbstractContext) = false

function Base.push!(context::ValuesAsInModelContext, vn::VarName, value)
    return setindex!(context.values, copy(value), prefix(context, vn))
end

function broadcast_push!(context::ValuesAsInModelContext, vns, values)
    return push!.((context,), vns, values)
end

# This will be hit if we're broadcasting an `AbstractMatrix` over a `MultivariateDistribution`.
function broadcast_push!(
    context::ValuesAsInModelContext, vns::AbstractVector, values::AbstractMatrix
)
    for (vn, col) in zip(vns, eachcol(values))
        push!(context, vn, col)
    end
end

# `tilde_asssume`
function tilde_assume(context::ValuesAsInModelContext, right, vn, vi)
    is_tracked_value_right = is_tracked_value(right)
    if is_tracked_value_right
        value = right.value
        logp = zero(getlogp(vi))
    else
        value, logp, vi = tilde_assume(childcontext(context), right, vn, vi)
    end
    # Save the value.
    if is_tracked_value_right ||
        isnothing(context.tracked_varnames) ||
        any(tracked_vn -> subsumes(tracked_vn, vn), context.tracked_varnames)
        push!(context, vn, value)
    end
    # Pass on.
    return value, logp, vi
end
function tilde_assume(
    rng::Random.AbstractRNG, context::ValuesAsInModelContext, sampler, right, vn, vi
)
    is_tracked_value_right = is_tracked_value(right)
    if is_tracked_value_right
        value = right.value
        logp = zero(getlogp(vi))
    else
        value, logp, vi = tilde_assume(rng, childcontext(context), sampler, right, vn, vi)
    end
    # Save the value.
    if is_tracked_value_right ||
        isnothing(context.tracked_varnames) ||
        any(tracked_vn -> subsumes(tracked_vn, vn), context.tracked_varnames)
        push!(context, vn, value)
    end
    # Pass on.
    return value, logp, vi
end

"""
    values_as_in_model(model::Model, include_colon_eq::Bool, varinfo::AbstractVarInfo[, context::AbstractContext])

Get the values of `varinfo` as they would be seen in the model.

More specifically, this method attempts to extract the realization _as seen in
the model_. For example, `x[1] ~ truncated(Normal(); lower=0)` will result in a
realization that is compatible with `truncated(Normal(); lower=0)` -- i.e. one
where the value of `x[1]` is positive -- regardless of whether `varinfo` is
working in unconstrained space.

Hence this method is a "safe" way of obtaining realizations in constrained
space at the cost of additional model evaluations.

# Arguments
- `model::Model`: model to extract realizations from.
- `include_colon_eq::Bool`: whether to also include variables on the LHS of `:=`.
- `varinfo::AbstractVarInfo`: variable information to use for the extraction.
- `context::AbstractContext`: base context to use for the extraction. Defaults
   to `DynamicPPL.DefaultContext()`.

# Examples

## When `VarInfo` fails

The following demonstrates a common pitfall when working with [`VarInfo`](@ref)
and constrained variables.

```jldoctest
julia> using Distributions, StableRNGs

julia> rng = StableRNG(42);

julia> @model function model_changing_support()
           x ~ Bernoulli(0.5)
           y ~ x == 1 ? Uniform(0, 1) : Uniform(11, 12)
       end;

julia> model = model_changing_support();

julia> # Construct initial type-stable `VarInfo`.
       varinfo = VarInfo(rng, model);

julia> # Link it so it works in unconstrained space.
       varinfo_linked = DynamicPPL.link(varinfo, model);

julia> # Perform computations in unconstrained space, e.g. changing the values of `θ`.
       # Flip `x` so we hit the other support of `y`.
       θ = [!varinfo[@varname(x)], rand(rng)];

julia> # Update the `VarInfo` with the new values.
       varinfo_linked = DynamicPPL.unflatten(varinfo_linked, θ);

julia> # Determine the expected support of `y`.
       lb, ub = θ[1] == 1 ? (0, 1) : (11, 12)
(0, 1)

julia> # Approach 1: Convert back to constrained space using `invlink` and extract.
       varinfo_invlinked = DynamicPPL.invlink(varinfo_linked, model);

julia> # (×) Fails! Because `VarInfo` _saves_ the original distributions
       # used in the very first model evaluation, hence the support of `y`
       # is not updated even though `x` has changed.
       lb ≤ first(varinfo_invlinked[@varname(y)]) ≤ ub
false

julia> # Approach 2: Extract realizations using `values_as_in_model`.
       # (✓) `values_as_in_model` will re-run the model and extract
       # the correct realization of `y` given the new values of `x`.
       lb ≤ values_as_in_model(model, true, varinfo_linked)[@varname(y)] ≤ ub
true
```
"""
function values_as_in_model(
    model::Model,
    include_colon_eq::Bool,
    varinfo::AbstractVarInfo,
    tracked_varnames=model.tracked_varnames,
    context::AbstractContext=DefaultContext(),
)
    @show tracked_varnames
    tracked_varnames = isnothing(tracked_varnames) ? nothing : collect(tracked_varnames)
    context = ValuesAsInModelContext(include_colon_eq, tracked_varnames, context)
    evaluate!!(model, varinfo, context)
    return context.values
end
