const FIXED_TRANSFORM_ACCNAME = :FixedTransformAccumulator

function _get_fixed_transform(
    val, tv::TransformedValue{V,DynamicLink}, logjac, vn, dist
) where {V}
    return FixedTransform(Bijectors.VectorBijectors.from_linked_vec(dist))
end
function _get_fixed_transform(
    val, tv::TransformedValue{V,Unlink}, logjac, vn, dist
) where {V}
    return FixedTransform(Bijectors.VectorBijectors.from_vec(dist))
end
function _get_fixed_transform(
    val, tv::TransformedValue{V,<:FixedTransform}, logjac, vn, dist
) where {V}
    return tv.transform
end

"""
    FixedTransformAccumulator()

An accumulator that calculates and stores the 'fixed' transforms for all variables in a model.

Normally, when running a model with a transform strategy such as `LinkAll`, the transforms are
calculated *during* model execution and not cached. This ensures that the transforms are up-to-date
with the current variable values, which can matter in cases such as

```julia
x ~ Normal()
y ~ truncated(Normal(); lower=x)
```

or

```julia
x ~ Normal()
y ~ (x > 0 ? Normal() : Exponential())
```

where the transforms for `y` depend on the value of `x`.

"""
FixedTransformAccumulator() = VNTAccumulator{FIXED_TRANSFORM_ACCNAME}(_get_fixed_transform)

"""
    get_fixed_transforms(vi::DynamicPPL.AbstractVarInfo)

Extract the transforms stored in the [`FixedTransformAccumulator`](@ref) of an
AbstractVarInfo. Errors if the AbstractVarInfo does not have a `FixedTransformAccumulator`.
"""
function get_fixed_transforms(vi::DynamicPPL.AbstractVarInfo)
    return DynamicPPL.getacc(vi, Val(FIXED_TRANSFORM_ACCNAME)).values
end

"""
    get_fixed_transforms(
        model::DynamicPPL.Model,
        transform_strategy::AbstractTransformStrategy
    )

Extract the fixed transforms for all variables in a model by running the model with the
given transform strategy.

Note that, even though this method evaluates the model once, this method does *not* accept
an RNG argument to control that evaluation. This is because the fixed transforms are
supposed to be *fixed*, i.e., they should not depend on random choices made during model
execution!
"""
function get_fixed_transforms(
    model::DynamicPPL.Model, transform_strategy::AbstractTransformStrategy
)
    rng = Random.default_rng()
    accs = OnlyAccsVarInfo(FixedTransformAccumulator())
    _, accs = init!!(rng, model, accs, InitFromPrior(), transform_strategy)
    return get_fixed_transforms(accs)
end
