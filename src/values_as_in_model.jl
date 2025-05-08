"""
    ValuesAsInModelAccumulator <: AbstractAccumulator

An accumulator that is used by [`values_as_in_model`](@ref) to obtain values
of the model parameters as they are in the model.

This is particularly useful when working in unconstrained space, but one
wants to extract the realization of a model in a constrained space.

# Fields
$(TYPEDFIELDS)
"""
struct ValuesAsInModelAccumulator <: AbstractAccumulator
    "values that are extracted from the model"
    values::OrderedDict
    "whether to extract variables on the LHS of :="
    include_colon_eq::Bool
end
function ValuesAsInModelAccumulator(include_colon_eq)
    return ValuesAsInModelAccumulator(OrderedDict(), include_colon_eq)
end

accumulator_name(::Type{<:ValuesAsInModelAccumulator}) = :ValuesAsInModel

function split(acc::ValuesAsInModelAccumulator)
    return ValuesAsInModelAccumulator(empty(acc.values), acc.include_colon_eq)
end
function combine(acc1::ValuesAsInModelAccumulator, acc2::ValuesAsInModelAccumulator)
    if acc1.include_colon_eq != acc2.include_colon_eq
        msg = "Cannot combine accumulators with different include_colon_eq values."
        throw(ArgumentError(msg))
    end
    return ValuesAsInModelAccumulator(
        merge(acc1.values, acc2.values), acc1.include_colon_eq
    )
end

function Base.push!(acc::ValuesAsInModelAccumulator, vn::VarName, val)
    setindex!(acc.values, deepcopy(val), vn)
    return acc
end

function is_extracting_values(vi::AbstractVarInfo)
    return hasacc(vi, Val(:ValuesAsInModel)) &&
           getacc(vi, Val(:ValuesAsInModel)).include_colon_eq
end

function accumulate_assume!!(acc::ValuesAsInModelAccumulator, val, logjac, vn, right)
    return push!(acc, vn, val)
end

accumulate_observe!!(acc::ValuesAsInModelAccumulator, right, left, vn) = acc

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
- `context::AbstractContext`: evaluation context to use in the extraction. Defaults
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
    context::AbstractContext=DefaultContext(),
)
    accs = getaccs(varinfo)
    varinfo = setaccs!!(deepcopy(varinfo), (ValuesAsInModelAccumulator(include_colon_eq),))
    varinfo = last(evaluate!!(model, varinfo, context))
    return getacc(varinfo, Val(:ValuesAsInModel)).values
end
