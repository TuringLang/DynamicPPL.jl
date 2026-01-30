# How data flows through a model

This page aims to show how data can enter a model, how we use it to perform computations, and how the results are extracted.

As a high-level summary:

  - **initialisation strategies** are responsible for generating parameter values
  - **transform strategies** are responsible for telling the model how to interpret those values (i.e., what log-Jacobian needs to be computed)
  - **accumulators** are responsible for aggregating the outputs of the model (e.g. log probabilities, transformed values, etc.)

In this model, there is a clear separation between the *inputs* to the model, and the *outputs* of the model.

!!! note "DefaultContext"
    
    While `VarInfo` and `DefaultContext` still exist, this is mostly a historical remnant.
    `DefaultContext` means that the inputs should come from the values of the provided `VarInfo`, and the outputs are stored in the accumulators of the provided `VarInfo`.
    However, this can easily be refactored such that the values are provided directly as an initialisation strategy.
    See [this issue](https://github.com/TuringLang/DynamicPPL.jl/issues/1184) for more details.

There are three stages to every tilde-statement:

 1. Initialisation: get an `AbstractTransformedValue` from the initialisation strategy.
 2. Computation: figure out the untransformed (raw) value and the transformed value (where necessary); compute the relevant log-Jacobian.
 3. Accumulation: pass all the relevant information to the accumulators, which individually decide what to do with it.

In fact this (more or less) directly translates into three lines of code: see e.g. the method for `tilde_assume!!` in `src/contexts/init.jl`, which (as of the time of writing) looks like:

```julia
function DynamicPPL.tilde_assume!!(ctx::InitContext, dist, vn, template, vi)
    # 1. Initialisation
    init_tval = DynamicPPL.init(ctx.rng, vn, dist, ctx.strategy)

    # 2. Computation
    x, tval, logjac = apply_transform_strategy(ctx.transform_strategy, init_tval, vn, dist)

    # 3. Accumulation
    vi = DynamicPPL.setindex_with_dist!!(vi, tval, dist, vn, template)
    vi = DynamicPPL.accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
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
init_tval = DynamicPPL.init(ctx.rng, vn, dist, ctx.strategy)
```

The initialisation step is handled by the `init` function, which dispatches on the initialisation strategy.
For example, if `ctx.strategy` is `InitFromPrior()`, then `init()` samples a value from the distribution `dist`.

!!! note "DefaultContext"
    
    For `DefaultContext`, initialisation is handled by looking for the value stored inside `vi`.

As discussed in the [Initialisation strategies](./init.md) page, this step, in general, does not return just the raw value (like `rand(dist)`).
It returns an [`DynamicPPL.AbstractTransformedValue`](@ref), which represents a value that _may_ have been transformed.
In the case of `InitFromPrior()`, the value is of course not transformed; we return a [`DynamicPPL.UntransformedValue`](@ref) wrapping the sampled value.

However, consider the case where we are using parameters stored inside a `VarInfo`: the value may have been stored either as a vectorised form, or as a linked vectorised form.
In this case, `init()` will return either a [`DynamicPPL.VectorValue`](@ref) or a [`DynamicPPL.LinkedVectorValue`](@ref).

The reason why we return this wrapped value is because we want to avoid having to perform transformations multiple times.
Each step is responsible for only performing the transformations it needs to.
At this stage, there has not yet been any need for the raw value, so we do not perform any transformations yet.
Thus, the `AbstractTransformedValue` is passed straight through and is used by both the computation and accumulation steps.

!!! note "The return type of init() doesn't matter"
    
    The _type_ of `AbstractTransformedValue` returned by `init()`, in general, has no impact on whether the value is considered to be transformed or not.
    That is determined solely by the transform strategy (see below).
    This separation allows us to perform the minimum amount of transformations necessary inside `init()`.
    If we were to eagerly transform the value inside `init()`, we could easily end up performing the same transformation multiple times across the different steps.

## Computation

```julia
x, tval, logjac = apply_transform_strategy(ctx.transform_strategy, init_tval, vn, dist)
```

There are three return values in this step, and they correspond to the three things that this step needs to do.
They are all interconnected, which is why they are computed together inside `apply_transform_strategy()`: by doing so we can ensure that `with_logabsdet_jacobian` is only called a maximum of once per tilde-statement.

 1. **Get the raw (untransformed) value `x`**
    
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

 2. **Get the (possibly transformed) value `tval`**
    
    In addition to the raw value, if the transform strategy indicates that we should treat `vn` as being in transformed space, we also need to compute the transformed value.
    This is because some accumulators may need to work with the transformed value instead of the raw value.
    
    (If there is a full VarInfo being used, the transformed value will also have to be set inside the VarInfo.)
 3. **Compute the log-Jacobian `logjac`**
    
    `logjac` is accumulated according to the transform specified by the transform strategy.
    The convention in DynamicPPL is that the log-Jacobian is always computed with respect to the forward transformation.

It is worth emphasising that whether a value is transformed or not is determined by the *transform strategy* provided to the model (i.e., `ctx.link_strategy`), not the initialisation strategy (`ctx.strategy`).
The reason for this is to allow a separation between the source of the values (initialisation) and how those values are to be interpreted (transform strategy).

This allows us to, for example, generate values from the (unlinked) prior but also calculate their log-density in transformed space and accumulate transformed values by combining `InitFromPrior()` with `LinkAll()`.
It also allows us to read values from an existing `VarInfo` but interpret them as being in a different space by combining `InitFromParams()` with a different transform strategy: this corresponds exactly to the act of 'linking' a VarInfo.

!!! note "DefaultContext"
    
    For DefaultContext, whether or not the variable is transformed will depend on the `VarInfo` used for evaluation. If the variable is stored as transformed in the `VarInfo`, then it will be treated as transformed here.
    Notice that both the initialisation strategy as well as the transform strategy are effectively determined by the `VarInfo` in this case.
    The separation described above is not possible when using `DefaultContext`.
    
    The move away from `DefaultContext` and towards `InitContext` is motivated by the desire to separate these two concerns, and to enable a more modular and declarative way of specifying how a model is to be evaluated.

!!! note "Log-Jacobian computation"
    
    In principle, if the log-Jacobian is not of interest to any of the accumulators, we _could_ skip computing it here.
    However, that is not easy to determine in practice.
    We also cannot defer the log-Jacobian computation to the accumulator, since it is often more efficient to compute it at the same time as the transformation (i.e., using `with_logabsdet_jacobian`).
    The current situation of computing it once here is the most sensible compromise (for now).
    
    One could envision a future where accumulators declare upfront (via their type) whether they need the log-Jacobian or not. We could then skip computing it if no accumulator needs it.

## Accumulation

```julia
vi = DynamicPPL.setindex_with_dist!!(vi, tval, dist, vn, template)
vi = DynamicPPL.accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
```

!!! note
    
    The first line, `setindex_with_dist!!`, is only necessary when using a full `VarInfo`.
    It essentially stores the value `tval` inside the `VarInfo`, but makes sure to store a vectorised form (i.e., if `tval` is an `UntransformedValue`, it will be converted to a `VectorValue` before being stored).
    This is entirely equivalent to using a `VectorValueAccumulator` to store the values; it's just that when using a full `VarInfo` that accumulator is 'built-in' as `vi.values`.
    
    Since conceptually this is the same as an accumulator, we will not discuss it further here.

Here, we pass all of the information we have gathered so far for this tilde-statement to the accumulators.
`accumulate_assume!!(vi::AbstractVarInfo, ...)` will loop over all accumulators stored inside `vi`, and call each of their individual `accumulate_assume!!` methods.
This method is responsible for deciding how to combine the information provided.

Accumulators are described in much more detail on the [Accumulators](./accumulators.md) page; please read that for more information!
