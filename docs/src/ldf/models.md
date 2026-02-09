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

Suppose that we want to evaluate the `LogDensityFunction` at the new raw values `x = 5.0` and `y = 0.6`.
The corresponding vectorised parameters should be

```@example 1
using StatsFuns: logit
[5.0, logit(0.6)]
```

(Of course, in general you can't just write these down by hand, since it will depend on the model.)

The first thing to do is to re-evaluate the model with the initialisation strategy you are interested in, and collect a new set of vector values.

Note that we *must* use the same transform strategy as the one contained in the `LogDensityFunction`, since we want to collect vector values that are consistent with the `LogDensityFunction`.

```@example 1
init_strategy = InitFromParams(VarNamedTuple(; x=5.0, y=0.6))
transform_strategy = ldf.transform_strategy

accs = OnlyAccsVarInfo(VectorValueAccumulator())
_, accs = init!!(model, accs, init_strategy, transform_strategy)
vector_values = get_vector_values(accs)
```

You can of course also bundle any extra accumulators you like into the above if you are interested not only in the vectorised parameters but also (e.g.) the log-density.
This allows you to obtain all the information you need with only one model evaluation.

Once you have a new `VarNamedTuple` of vector values, you can use the function [`to_vector_input`](@ref):

```@example 1
new_vector = to_vector_input(vector_values, ldf)
```

!!! note "What happened to `varinfo[:]`?"
    
    Just like before, if you are familiar with older versions of DynamicPPL, you may realise that this workflow is similar to the old version of calling `varinfo[:]` to obtain a set of vectorised parameters.
    
    This still exists (although we prefer that you use [`internal_values_as_vector(varinfo)`](@ref internal_values_as_vector) instead, since that name is more descriptive).
    Although `internal_values_as_vector` is not as unsafe as `unflatten!!`, it still does not perform any checks to ensure that the vectorised parameters are consistent with the `LogDensityFunction`.
    For example, you could extract a length-2 vector of parameters from a `VarInfo`, and it would work with any `LogDensityFunction` that also expects a length-2 vector, even if the actual transformations and model are completely different.
    
    The approach above based on accumulators does preserve the necessary information, and `to_vector_input` will carry out the necessary checks to ensure correctness.
    It is no less performant than before (apart from the time needed to run said checks!).
