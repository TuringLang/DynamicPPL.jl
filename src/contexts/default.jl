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

"""
    DynamicPPL.tilde_assume!!(
        ::DefaultContext,
        right::Distribution,
        vn::VarName,
        template::Any,
        vi::AbstractVarInfo
    )

Handle assumed variables. For `DefaultContext`, this function extracts the value associated
with `vn` from `vi`, If `vi` does not contain an appropriate value then this will error.
"""
function tilde_assume!!(
    ::DefaultContext, right::Distribution, vn::VarName, template::Any, vi::AbstractVarInfo
)
    # TODO(penelopeysm): Conceptually, this is the same as InitContext, except that:
    #  1. init(...) is not called; instead we read the value from vi.
    #  2. apply_transform_strategy(...) is not called; instead we infer from vi whether the
    #     value is supposed to be linked or not.
    # This can definitely be unified in the future.
    tval = get_transformed_value(vi, vn)
    trf = if tval isa LinkedVectorValue
        # Note that we can't rely on the stored transform being correct (e.g. if new values
        # were placed in `vi` via `unflatten!!`, so we regenerate the transforms.
        from_linked_vec_transform(right)
    elseif tval isa VectorValue
        from_vec_transform(right)
    else
        error("Expected transformed value to be a VectorValue or LinkedVectorValue")
    end
    x, inv_logjac = with_logabsdet_jacobian(trf, get_internal_value(tval))
    vi = accumulate_assume!!(vi, x, tval, -inv_logjac, vn, right, template)
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
