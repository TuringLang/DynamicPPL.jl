# How data flows through a model

Having discussed initialisation strategies and accumulators, we can now put all the pieces together to show how data enters a model, is used to perform computations, and how the results are extracted.

**The summary is: initialisation strategies are responsible for telling the model what values to use for its parameters, whereas accumulators act as containers for aggregated outputs.**

Thus, there is a clear separation between the *inputs* to the model, and the *outputs* of the model.

!!! note
    
    While `VarInfo` and `DefaultContext` still exist, this is mostly a historical remnant. `DefaultContext` means that the inputs should come from the values of the provided `VarInfo`, and the outputs are stored in the accumulators of the provided `VarInfo`. However, this can easily be refactored such that the values are provided directly as an initialisation strategy. See [this issue](https://github.com/TuringLang/DynamicPPL.jl/issues/1184) for more details.

There are three stages to every tilde-statement:

 1. Initialisation: get an `AbstractTransformedValue` from the initialisation strategy.

 2. Computation: figure out the untransformed (raw) value; compute the log-Jacobian if necessary.
 3. Accumulation: pass all the relevant information to the accumulators, which individually decide what to do with it.

In fact this (more or less) directly translates into three lines of code: see e.g. the method for `tilde_assume!!` in `src/onlyaccs.jl`, which (as of the time of writing) looks like:

```julia
function DynamicPPL.tilde_assume!!(ctx::InitContext, dist, vn, template, vi)
    # 1. Initialisation
    tval = DynamicPPL.init(ctx.rng, vn, dist, ctx.strategy)

    # 2. Computation
    # (Though see also the warning in the computation section below.)
    x, inv_logjac = Bijectors.with_logabsdet_jacobian(
        DynamicPPL.get_transform(tval), DynamicPPL.get_internal_value(tval)
    )

    # 3. Accumulation
    vi = DynamicPPL.accumulate_assume!!(vi, x, tval, -inv_logjac, vn, dist, template)
    return x, vi
end
```

For `tilde_observe!!`, the code is very similar, but even easier: the value can be read directly from the data provided to the model, so there is no need for an initialisation step.
Since the value is already untransformed, we can skip the second step.
Finally, accumulators must behave differently: e.g. incrementing the likelihood instead of the prior.
That is accomplished by calling `accumulate_observe!!` instead of `accumulate_assume!!`.

In the following sections, we stick to the three sections of `tilde_assume!!`.

## Initialisation

```julia
tval = DynamicPPL.init(ctx.rng, vn, dist, ctx.strategy)
```

The initialisation step is handled by the `init` function, which dispatches on the initialisation strategy.
For example, if `ctx.strategy` is `InitFromPrior()`, then `init()` samples a value from the distribution `dist`.

!!! note
    
    For `DefaultContext`, this is replaced by looking for the value stored inside `vi`. As described above, this can be refactored in the near future.

This step, in general, does not return just the raw value (like `rand(dist)`).
It returns an [`DynamicPPL.AbstractTransformedValue`](@ref), which represents a value that _may_ have been transformed.
In the case of `InitFromPrior()`, the value is of course not transformed; we return a [`DynamicPPL.UntransformedValue`](@ref) wrapping the sampled value.

However, consider the case where we are using parameters stored inside a `VarInfo`: the value may have been stored either as a vectorised form, or as a linked vectorised form.
In this case, `init()` will return either a [`DynamicPPL.VectorValue`](@ref) or a [`DynamicPPL.LinkedVectorValue`](@ref).

The reason why we return this wrapped value is because sometimes we don't want to eagerly perform the transformation.
Consider the case where we have an accumulator that attempts to store linked values (this is done precisely when linking a VarInfo: the linked values are stored in an accumulator, which then becomes the basis of the linked VarInfo).
In this case, if we eagerly perform the inverse link transformation, we would have to link it again inside the accumulator, which is inefficient!

The `AbstractTransformedValue` is passed straight through and is used by both the computation and accumulation steps.

## Computation

```julia
x, inv_logjac = Bijectors.with_logabsdet_jacobian(
    DynamicPPL.get_transform(tval), DynamicPPL.get_internal_value(tval)
)
```

At *some* point, we do need to perform the transformation to get the actual raw value.
This is because DynamicPPL promises in the model that the variables on the left-hand side of the tilde are actual raw values.

```julia
@model function f()
    x ~ dist
    # Here, `x` _must_ be the actual raw value.
    @show x
end
```

Thus, regardless of what we are accumulating, we will have to unwrap the transformed value provided by `init()`.
We also need to account for the log-Jacobian of the transformation, if any.

!!! note
    
    In principle, if the log-Jacobian is not of interest to any of the accumulators, we _could_ skip computing it here.
    However, that is not easy to determine in practice.
    We also cannot defer the log-Jacobian computation to the accumulator, since if multiple accumulators need the log-Jacobian, we would end up computing it multiple times.
    The current situation of computing it once here is the most sensible compromise (for now).
    
    One could envision a future where accumulators declare upfront (via their type) whether they need the log-Jacobian or not. We could then skip computing it if no accumulator needs it.

!!! warning
    
    If you look at the source code for that method, it is more complicated than the above!
    Have we lied?
    It turns out that there is a subtlety here: the transformation obtained from `DynamicPPL.get_transform(tval)` may in fact be incorrect.
    
    Consider the case where a transform is dependent on the value itself (e.g., a variable whose support depends on another variable).
    In this case, setting new values into a VarInfo (via `unflatten!!`) may cause the cached transformations to be invalid.
    Where possible, it is better to re-obtain the transformation from `dist`, which is always up-to-date since it is obtained from model execution.

## Accumulation

```julia
vi = DynamicPPL.accumulate_assume!!(vi, x, tval, -inv_logjac, vn, dist, template)
```

This step is where most of the interesting action happens.

[...]
