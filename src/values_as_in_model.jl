
"""
    ValuesAsInModelContext

A context that is used by [`values_as_in_model`](@ref) to obtain values
of the model parameters as they are in the model.

This is particularly useful when working in unconstrained space, but one
wants to extract the realization of a model in a constrained space.

# Fields
$(TYPEDFIELDS)
"""
struct ValuesAsInModelContext{T,C<:AbstractContext} <: AbstractContext
    "values that are extracted from the model"
    values::T
    "child context"
    context::C
end

ValuesAsInModelContext(values) = ValuesAsInModelContext(values, DefaultContext())
function ValuesAsInModelContext(context::AbstractContext)
    return ValuesAsInModelContext(OrderedDict(), context)
end

NodeTrait(::ValuesAsInModelContext) = IsParent()
childcontext(context::ValuesAsInModelContext) = context.context
function setchildcontext(context::ValuesAsInModelContext, child)
    return ValuesAsInModelContext(context.values, child)
end

function Base.push!(context::ValuesAsInModelContext, vn::VarName, value)
    return setindex!(context.values, copy(value), vn)
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
    value, logp, vi = tilde_assume(childcontext(context), right, vn, vi)
    # Save the value.
    push!(context, vn, value)
    # Save the value.
    # Pass on.
    return value, logp, vi
end
function tilde_assume(
    rng::Random.AbstractRNG, context::ValuesAsInModelContext, sampler, right, vn, vi
)
    value, logp, vi = tilde_assume(rng, childcontext(context), sampler, right, vn, vi)
    # Save the value.
    push!(context, vn, value)
    # Pass on.
    return value, logp, vi
end

# `dot_tilde_assume`
function dot_tilde_assume(context::ValuesAsInModelContext, right, left, vn, vi)
    value, logp, vi = dot_tilde_assume(childcontext(context), right, left, vn, vi)

    # Save the value.
    _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
    broadcast_push!(context, _vns, value)

    return value, logp, vi
end
function dot_tilde_assume(
    rng::Random.AbstractRNG, context::ValuesAsInModelContext, sampler, right, left, vn, vi
)
    value, logp, vi = dot_tilde_assume(
        rng, childcontext(context), sampler, right, left, vn, vi
    )
    # Save the value.
    _right, _left, _vns = unwrap_right_left_vns(right, left, vn)
    broadcast_push!(context, _vns, value)

    return value, logp, vi
end

"""
    values_as_in_model(model::Model[, varinfo::AbstractVarInfo, context::AbstractContext])
    values_as_in_model(rng::Random.AbstractRNG, model::Model[, varinfo::AbstractVarInfo, context::AbstractContext])

Get the values of `varinfo` as they would be seen in the model.

If no `varinfo` is provided, then this is effectively the same as
[`Base.rand(rng::Random.AbstractRNG, model::Model)`](@ref).

More specifically, this method attempts to extract the realization _as seen in the model_.
For example, `x[1] ~ truncated(Normal(); lower=0)` will result in a realization compatible
with `truncated(Normal(); lower=0)` regardless of whether `varinfo` is working in unconstrained
space.

Hence this method is a "safe" way of obtaining realizations in constrained space at the cost
of additional model evaluations.

# Arguments
- `model::Model`: model to extract realizations from.
- `varinfo::AbstractVarInfo`: variable information to use for the extraction.
- `context::AbstractContext`: context to use for the extraction. If `rng` is specified, then `context`
    will be wrapped in a [`SamplingContext`](@ref) with the provided `rng`.

# Examples

## When `VarInfo` fails

The following demonstrates a common pitfall when working with [`VarInfo`](@ref) and constrained variables.

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
       lb ≤ varinfo_invlinked[@varname(y)] ≤ ub
false

julia> # Approach 2: Extract realizations using `values_as_in_model`.
       # (✓) `values_as_in_model` will re-run the model and extract
       # the correct realization of `y` given the new values of `x`.
       lb ≤ values_as_in_model(model, varinfo_linked)[@varname(y)] ≤ ub
true
```
"""
function values_as_in_model(
    model::Model,
    varinfo::AbstractVarInfo=VarInfo(),
    context::AbstractContext=DefaultContext(),
)
    context = ValuesAsInModelContext(context)
    evaluate!!(model, varinfo, context)
    return context.values
end
function values_as_in_model(
    rng::Random.AbstractRNG,
    model::Model,
    varinfo::AbstractVarInfo=VarInfo(),
    context::AbstractContext=DefaultContext(),
)
    return values_as_in_model(model, varinfo, SamplingContext(rng, context))
end
