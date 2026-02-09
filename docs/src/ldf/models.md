# [Using LDFs in model evaluation](@id ldf-model)

As mentioned on the previous page, a `LogDensityFunction` contains more information than just the mapping from vectors to log-densities.
It also contains enough information to generate an initialisation strategy, and on top of that it directly wraps a transform strategy.

This means that, once you have constructed a `LogDensityFunction`, you can use it in conjunction with a parameter vector to evaluate models using the `init!!` framework established previously.

Let's start by generating the same `LogDensityFunction` as on the previous page:

```@example 1
using DynamicPPL, Distributions

@model function f()
    x ~ Normal()
    y ~ Beta(2, 2)
    return nothing
end
model = f()

accs = OnlyAccsVarInfo(VectorValueAccumulator())
_, accs = init!!(model, accs, InitFromPrior(), LinkAll())
vector_values = get_vector_values(accs)

ldf = LogDensityFunction(model, getlogjoint_internal, vector_values)
```

## Evaluating models using vectorised parameters

You can regenerate the initialisation strategy using the constructor `InitFromVector(vect, ldf)`:

```@example 1
init_strategy = InitFromVector([3.0, 4.0], ldf)
```

and access the transform strategy via

```@example 1
ldf.transform_strategy
```

With these two pieces of information, we can add whatever accumulators we like to the mix, and then evaluate using `init!!` as shown before.
For example, let's say we'd like to know what raw values the vector `[3.0, 4.0]` corresponds to.

```@example 1
accs = OnlyAccsVarInfo(RawValueAccumulator(false))
_, accs = init!!(model, accs, init_strategy, ldf.transform_strategy)
get_raw_values(accs)
```

Unsurprisingly, the raw (i.e., untransformed) value of `x` is 3.0, and `y` is equal to:

```@example 1
using StatsFuns: logistic
logistic(4.0)
```

!!! note "What happened to `unflatten!!`?"
    
    If you are familiar with older versions of DynamicPPL, you may realise that this workflow is similar to the old version of calling `unflatten(varinfo, vector)`, and then re-evaluating the model.
    
    [`unflatten!!`](@ref) still exists (albeit with a double exclamation mark now), and it still does the same thing as before, but we **strongly recommend against using it**.
    The reason is because `unflatten!!` leaves the VarInfo in an inconsistent state, since the parameters are updated but the log-density and transformations are not.
    Using this invalid VarInfo without reevaluating the model can lead to incorrect results.
    
    It is much safer to use `InitFromVector`, which encapsulates all the information needed to rerun the model, but does not pretend to also contain other information that is not actually updated.

## Obtaining new vectorised parameters

We can also do the reverse process here, which is to obtain new vectorised parameters that are consistent with this `LogDensityFunction`.

!!! danger
    
    Not yet implemented. The solution is to use a VectorValueAccumulator and then loop through the ranges stored in the LDF (don't just use internal_values_as_vector on the VVA, since there's no guarantee that the order of the vector values in the VVA is the same as the order of the vector values in the LDF!).
