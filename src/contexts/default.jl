"""
    struct DefaultContext <: AbstractContext end

`DefaultContext`, as the name suggests, is the default context used when instantiating a
model.

```jldoctest
julia> @model f() = x ~ Normal();

julia> model = f(); model.context
DefaultContext()
```

As an evaluation context, the behaviour of `DefaultContext` is to require all variables to be
present in the `AbstractVarInfo` used for evaluation. Thus, semantically, evaluating a model
with `DefaultContext` means 'calculating the log-probability associated with the variables
in the `AbstractVarInfo`'.
"""
struct DefaultContext <: AbstractContext end
NodeTrait(::DefaultContext) = IsLeaf()

"""
    DynamicPPL.tilde_assume!!(
        ::DefaultContext,
        prefix::Union{VarName,Nothing},
        right::Distribution,
        vn::VarName,
        vi::AbstractVarInfo
    )

Handle assumed variables. For `DefaultContext`, this function extracts the value associated
with `vn` from `vi`, If `vi` does not contain an appropriate value then this will error.
"""
function tilde_assume!!(
    ::DefaultContext,
    ::Union{VarName,Nothing},
    right::Distribution,
    vn::VarName,
    vi::AbstractVarInfo,
)
    y = getindex_internal(vi, vn)
    f = from_maybe_linked_internal_transform(vi, vn, right)
    x, inv_logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume!!(vi, x, -inv_logjac, vn, right)
    return x, vi
end

"""
    DynamicPPL.tilde_observe!!(
        ::DefaultContext,
        right::Distribution,
        left,
        vn::Union{VarName,Nothing},
        vi::AbstractVarInfo,
    )

Handle observed variables. This just accumulates the log-likelihood for `left`.
"""
function tilde_observe!!(
    ::DefaultContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    vi = accumulate_observe!!(vi, right, left, vn)
    return left, vi
end
