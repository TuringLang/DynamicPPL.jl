# [Overview](@id ldf)

Much of the Julia numerical computing ecosystem is built on top of an assumption that model parameters are provided as an `AbstractVector{<:Real}`.
This is a very general format that is agnostic towards the details of the model, and is also compatible with automatic differentiation (although it is worth pointing out that modern AD packages, such as Mooncake and Enzyme, are able to differentiate through more complex data structures).

This poses a challenge to DynamicPPL: how can we 'translate' between a model, e.g.,

```@example 1
using DynamicPPL, Distributions

@model function f()
    x ~ Normal()
    y ~ Beta(2, 2)
    return nothing
end
```

where we supply _named_ parameters, e.g. using `VarNamedTuple(x = 1.0, y = 0.5)`, and the vectorised format where parameters are _unnamed_, e.g. `[1.0, 0.5]`?

On top of that, it is often very useful to supply transformed parameters such that _any_ vector in ℝⁿ is a valid input to the algorithm.
We therefore also need a way to encode a particular transform strategy into this translation layer.

## `LogDensityFunction`, conceptually

DynamicPPL provides a struct, `LogDensityFunction`, which acts as this translation layer.
At its core, it is a wrapper around a model, a transform strategy, plus a mapping of `VarName`s to _ranges_.
For example, in the above example, we could store the information as follows:

```@example 1
model = f()
transform_strategy = LinkAll()
ranges = @vnt begin
    x := 1:1
    y := 2:2
end
```

Given all of this, plus a vector (say `[3.0, 4.0]`), we can figure out how to rerun our model:

  - Since the transform strategy is `LinkAll()`, we know that both values are transformed. (That is the transform strategy.)
  - We use the ranges to figure out that the transformed value of `x` is `[3.0]`, and the transformed value of `y` is `[4.0]`. (This essentially specifies an initialisation strategy.)
  - We can then use any accumulators we like to obtain the information we are interested in.

The actual implementation of `LogDensityFunction` is a bit more complex than this, but is otherwise not too different.

!!! note "Why store ranges instead of indices?"
    
    One might ask why we store ranges like `1:1` instead of just the index `1` itself.
    A major reason is because ranges generalise better to multivariate and other distributions, and Bijectors.jl contains an interface for transforming vectorised parameters into raw values.
    Storing indices for univariate distributions would require special-casing them, and would also make type stability more difficult.

## Creating a `LogDensityFunction`

The constructor for `LogDensityFunction` is

```julia
LogDensityFunction(model, logdensityfunc, vector_values; adtype)
```

`model` is of course the model itself, but the other arguments deserve more explanation.

  - `logdensityfunc` is a function that takes an `OnlyAccsVarInfo` and returns some real number.
    For example, it could be `getlogjoint_internal` (most of the time that is what you will want!).
    This is the argument that makes `LogDensityFunction` actually obey the interface that e.g. optimisers expect.

  - `vector_values` is a `VarNamedTuple` that contains `VectorValue`s and `LinkedVectorValue`s.
    DynamicPPL will automatically infer from this the transform strategy as well as the ranges for each `VarName`.
    
    The easiest way to supply one is to use a `VectorValueAccumulator`:
    
    ```@example 1
    accs = OnlyAccsVarInfo(VectorValueAccumulator())
    _, accs = init!!(model, accs, InitFromPrior(), LinkAll())
    vector_values = get_vector_values(accs)
    
    ldf = LogDensityFunction(model, getlogjoint_internal, vector_values)
    ```
  - If you need to also obtain gradients with respect to the vectorised parameters, you can pass `adtype::ADTypes.AbstractADType` as a keyword argument.
    If gradients are not needed, you can omit this.

## The LogDensityProblems.jl interface

Once you have constructed a `LogDensityFunction`, it will obey [the LogDensityProblems.jl interface](https://www.tamaspapp.eu/LogDensityProblems.jl/stable/).
The most important part of this is

```@example 1
using LogDensityProblems

LogDensityProblems.logdensity(ldf, [3.0, 4.0])
```

We can verify quickly that this is what we expect by running the model manually:

```@example 1
using StatsFuns: logistic

x = 3.0
y = logistic(4.0)
params = VarNamedTuple(; x=x, y=y)
_, accs = init!!(model, OnlyAccsVarInfo(), InitFromParams(params), LinkAll())
accs
```

and we see that the log-prior minus the log-Jacobian is indeed the same as what `LogDensityProblems.logdensity` returns.

If you created the `LogDensityFunction` with an `adtype`, you can also obtain gradients via

```julia
LogDensityProblems.logdensity_and_gradient(ldf, [3.0, 4.0])
```

(although that is not demonstrated here since it requires an AD backend to be loaded).

Other functions such as `LogDensityProblems.capabilities` and `LogDensityProblems.dimension` will also work as expected with `LogDensityFunction`.

## Is `LogDensityFunction` less powerful than model evaluation?

Given that the core purpose of `LogDensityFunction` is to evaluate a function mapping from vectors to log-densities, it might seem that it contains less information than the original model itself.

However, this is not true!
As alluded to above, the `LogDensityFunction` is a convenient wrapper that knows how to generate an initialisation strategy, and also contains a transform strategy.

On the next page we'll see how we can access these ourselves and use them to re-evaluate the model with the vectorised parameters.
