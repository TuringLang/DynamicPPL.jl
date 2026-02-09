# [Initialisation strategies](@id init-strategies)

In DynamicPPL, *initialisation strategies* are used to determine the parameters used to evaluate a model.
Every time an *assume* tilde-statement is seen (i.e., a random variable), the initialisation strategy is used to generate a value for that variable.

!!! note
    
    One might perhaps more appropriately call them parameter generation strategies.
    Even the name *initialisation* is a bit of a historical misnomer (the original intent was that they would be used to populate an empty VarInfo with some values).
    However, over time, it has become clear that these are general enough to describe essentially any way of choosing parameters to evaluate a model with.

DynamicPPL provides four initialisation strategies out of the box.
For many purposes you should be able to get away with only using these.

  - [`InitFromPrior`](@ref): samples from the prior distribution.
  - [`InitFromParams`](@ref): reads from a set of given parameters. The parameters may be supplied in many different forms, but a `VarNamedTuple` is the most robust.
  - [`InitFromUniform`](@ref): samples from a uniform distribution in linked space.
  - [`InitFromVector`](@ref): reads from a set of vectorised parameters. This also needs a `LogDensityFunction` to provide the necessary information about how the vectorised parameters map to the model variables.

```@docs
InitFromPrior
InitFromParams
InitFromUniform
InitFromVector
```

However, sometimes you will need to implement your own initialisation strategy.
The subsequent sections will demonstrate how this can be done.

## The required interface

Each initialisation strategy must subtype `AbstractInitStrategy`, and implement `DynamicPPL.init(rng, vn, dist, strategy)`, which must return an `AbstractTransformedValue`.

```@docs; canonical=false
AbstractInitStrategy
init
DynamicPPL.AbstractTransformedValue
```

## An example

Consider the following model:

```@example 1
using DynamicPPL, Distributions, Random

@model function f()
    x ~ Normal()
    return x
end
model = f()
```

Suppose we are writing a Metropolis–Hastings sampler, and we want to perform a random walk where the next proposed value of `x` depends on the previous value of `x`.
Given a previous value `x_prev` we can define a custom initialisation strategy as follows:

```@example 1
struct InitRandomWalk <: DynamicPPL.AbstractInitStrategy
    x_prev::Float64
    step_size::Float64
end

function DynamicPPL.init(rng, vn::VarName, ::Distribution, strategy::InitRandomWalk)
    new_x = rand(rng, Normal(strategy.x_prev, strategy.step_size))
    # Insert some printing to see when this is called.
    @info "init() is returning: $new_x"
    return DynamicPPL.UntransformedValue(new_x)
end
```

Given a previous value of `x`

```@example 1
x_prev = 4.0
nothing # hide
```

we can then make a proposal for `x` as follows:

```@example 1
new_x, new_vi = DynamicPPL.init!!(
    model, VarInfo(), InitRandomWalk(x_prev, 0.5), UnlinkAll()
)
nothing # hide
```

When evaluating the model, the value for `x` will be exactly that new value we proposed.
We can see this from the return value:

```@example 1
new_x
```

Furthermore, we can read off the associated log-probability from the newly returned VarInfo:

```@example 1
DynamicPPL.getlogjoint(new_vi) ≈ logpdf(Normal(), new_x)
```

From this log-probability, we can compute the acceptance ratio for the Metropolis–Hastings step, and thereby create a valid MCMC sampler.

In this case, we have defined an initialisation strategy that is random (and thus uses the `rng` argument for reproducibility).
However, initialisation strategies can also be fully deterministic, in which case the `rng` argument is not needed.
For example, [`DynamicPPL.InitFromParams`](@ref) reads from a set of parameters which are known ahead of time.

## The returned `AbstractTransformedValue`

As mentioned above, the `init` function must return an `AbstractTransformedValue`.
The subtype of `AbstractTransformedValue` used does not affect the result of the model evaluation, but it may have performance implications.
**In particular, the returned subtype does not determine whether the log-Jacobian term is accumulated or not: that is determined by a separate [_transform strategy_](@ref transform-strategies).**

What this means is that initialisation strategies should always choose the laziest possible subtype of `AbstractTransformedValue`.

For example, in the above example, we used `UntransformedValue`, which is the simplest possible choice.
If a linked value is required by a later step inside `tilde_assume!!` (either the transformation or accumulation steps), it is the responsibility of that step to perform the linking.

Conversely, [`DynamicPPL.InitFromUniform`](@ref) samples inside linked space.
Instead of performing the inverse link transform and returning an `UntransformedValue`, it directly returns a `LinkedVectorValue`: this means that if a linked value is required by a later step, it is not necessary to link it again.
Even if no linked value is required, this lazy approach does not hurt performance, as it just defers the inverse linking to the later step.

In both cases, only one linking operation is performed (at most).
