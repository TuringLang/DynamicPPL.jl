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

!!! note "Extracting the model"
    
    If you only have access to `ldf` and not the original `model`, you can just extract it via `ldf.model`.
    See the [`LogDensityFunction`](@ref) docstring for more details.

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

Notice that you can include all the necessary log-density accumulators in the above:

```@example 1
accs = OnlyAccsVarInfo(
    LogPriorAccumulator(), LogLikelihoodAccumulator(), LogJacobianAccumulator()
)
_, accs = init!!(model, accs, init_strategy, ldf.transform_strategy)
accs
```

and from this you can, in one fell swoop, obtain the log-prior, log-likelihood, and log-Jacobian corresponding to the vector `[3.0, 4.0]`.

In fact, this is exactly what calling `LogDensityProblems.logdensity(ldf, [3.0, 4.0]` does, except that it also collapses this information into a single number based on which log-density getter you specified when creating the `LogDensityFunction`.
For example, in the above `ldf`, we used `getlogjoint_internal`, which is equal to `logprior + loglikelihood - logjacobian`:

```@example 1
using LogDensityProblems

LogDensityProblems.logdensity(ldf, [3.0, 4.0])
```

This represents a loss of information since we no longer know the individual components of the log-density.
`LogDensityFunction` itself must do this to obey the LogDensityProblems interface, but as a user of DynamicPPL you need not be limited to it; you can use `InitFromVector` to obtain all the information you need.

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

Naturally, we need to re-evaluate the model with the initialisation strategy we are interested in.
We can use a special accumulator, `VectorParamAccumulator`, to get the newly vectorised set of parameters.
This accumulator must take the `LogDensityFunction` as an argument, since it uses the information stored in there to ensure that the values it collects are consistent with the `LogDensityFunction`.

Note that we *must* use the same transform strategy as the one contained in the `LogDensityFunction`, or else we will be generating vectorised parameters that are inconsistent with the `LogDensityFunction` (an error will be thrown if that happens).

```@example 1
init_strategy = InitFromParams(VarNamedTuple(; x=5.0, y=0.6))
transform_strategy = ldf.transform_strategy

accs = OnlyAccsVarInfo(VectorParamAccumulator(ldf))
_, accs = init!!(model, accs, init_strategy, transform_strategy)
vec = get_vector_params(accs)
```

You can of course also bundle any extra accumulators you like into the above if you are interested not only in the vectorised parameters but also (e.g.) the log-density.
This allows you to obtain all the information you need with only one model evaluation.

!!! note
    
    Note that `VectorParamAccumulator` is not the same as `VectorValueAccumulator`.
    The former collects *a single vector of parameters*, whereas the latter collects a VarNamedTuple of vectorised parameters for each variable.
    
    If you used the latter instead, e.g.,
    
    ```@example 1
    accs = OnlyAccsVarInfo(VectorValueAccumulator())
    _, accs = init!!(model, accs, init_strategy, transform_strategy)
    vec_vals = get_vector_values(accs)
    ```
    
    you can convert `vec_vals::VarNamedTuple` to a single vector using
    
    ```@example 1
    to_vector_input(vec_vals, ldf)
    ```
    
    which is equivalent.
    However, this is slower since it has to generate an intermediate VarNamedTuple.

!!! note "What happened to `varinfo[:]`?"
    
    Just like before, if you are familiar with older versions of DynamicPPL, you may realise that this workflow is similar to the old version of calling `varinfo[:]` to obtain a set of vectorised parameters.
    
    This still exists (although we prefer that you use [`internal_values_as_vector(varinfo)`](@ref internal_values_as_vector) instead, since that name is more descriptive).
    Although `internal_values_as_vector` is not as unsafe as `unflatten!!`, it still does not perform any checks to ensure that the vectorised parameters are consistent with the `LogDensityFunction`.
    For example, you could extract a length-2 vector of parameters from a `VarInfo`, and it would work with any `LogDensityFunction` that also expects a length-2 vector, even if the actual transformations and model are completely different.
    
    The approach above based on `VectorParamAccumulator` does preserve the necessary information, and `to_vector_input` will carry out the necessary checks to ensure correctness.
    It is no less performant than before (apart from the time needed to run said checks!).
