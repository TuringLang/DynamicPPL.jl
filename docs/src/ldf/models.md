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

You can regenerate the initialisation strategy using the constructor

!!! danger
    
    Not yet implemented.

```@repl 1
# init_strategy = InitFromParams([3.0, 4.0], ldf)

init_strategy = InitFromPrior()
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

Unsurprisingly, `x` is 3.0, and `y` is `logistic(4.0)`.

## Obtaining new vectorised parameters

We can also do the reverse process here, which is to obtain new vectorised parameters that are consistent with this `LogDensityFunction`.

!!! danger
    
    Not yet implemented. The solution is to use a VectorValueAccumulator and then loop through the ranges stored in the LDF (don't just use internal_values_as_vector on the VVA, since there's no guarantee that the order of the vector values in the VVA is the same as the order of the vector values in the LDF!).
