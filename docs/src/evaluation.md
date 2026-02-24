# Model evaluation

Once you have defined a DynamicPPL model, let's say,

```@example 1
using DynamicPPL, Distributions

@model function f()
    x ~ Normal()
    y ~ Beta(2, 2)
    return (x=x, y=y)
end

model = f();
```

you will want to be able to evaluate it in some way.

Much like how a typical Julia function specifies some computation that involves variables and operations, the model definition defines a generative process, its random variables, and the relationships between them.
However, it still leaves open many questions.
For example,

  - what values of `x` and `y` should be used?

  - should those values be somehow transformed, e.g., do we want to constrain `y` to be in its original interval `(0, 1)`, or do we want to treat it as an unconstrained variable in `ℝ` (which possibly requires a Jacobian term to correct for the probability density)?
  - what information do we want to know about the model? Do we want to know the values of `x` and `y`, the log-probability of the model, ...?

DynamicPPL offers a powerful and modular evaluation framework which lets you control each of these aspects individually.

The following table offers a high-level summary of each of these different parts.
Each of these are described in more detail on the linked pages; this page shows some examples of how they can be composed.

| Concept                 | Subtype                             | Purpose                                                                             |
|:----------------------- |:----------------------------------- |:----------------------------------------------------------------------------------- |
| Initialisation strategy | [`AbstractInitStrategy`](@ref)      | Specifies how parameter values are generated                                        |
| Transform strategy      | [`AbstractTransformStrategy`](@ref) | Specifies how parameter values are transformed and how the log-Jacobian is computed |
| Accumulators            | [`AbstractAccumulator`](@ref)       | Specifies how the outputs of the model are aggregated                               |

To evaluate a model with these three components, you can use the method [`DynamicPPL.init!!`](@ref):

```julia
retval, accs = DynamicPPL.init!!(
    [rng::Random.AbstractRNG]model::DynamicPPL.Model,
    accs::DynamicPPL.OnlyAccsVarInfo,
    init_strategy::DynamicPPL.AbstractInitStrategy,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
)
```

which returns a tuple of the model's return value (the NamedTuple `(x=x, y=y)` in the example above) and the accumulators after evaluation.

!!! note "OnlyAccsVarInfo"
    
    `OnlyAccsVarInfo` is a thin wrapper around a set of accumulators.
    You can construct it using `OnlyAccsVarInfo(acc1, acc2, ...)`, where `acc1`, `acc2`, ... are the accumulators that you want to use during evaluation.
    
    The main reason why `OnlyAccsVarInfo` exists is that it acts as a bridge between older code that expects a `VarInfo` and the new evaluation framework that is described above.
    `OnlyAccsVarInfo` contains only accumulators (as its name suggests), but implements a subset of the `AbstractVarInfo` interface, which allows it to be used in places where a `VarInfo` is expected.
    
    In the future it is likely that this will be removed, and you can directly pass the tuple of accumulators itself without having to wrap it.

!!! note "What's happening to VarInfo?"
    
    If you have been using DynamicPPL in the past, you may be familiar with the general idea of `VarInfo`.
    (If that sounds completely foreign to you, you can ignore this!)
    
    While this still exists, we **strongly** encourage you to not work with it directly.
    The reason for this is because `VarInfo` conflates all three of the concepts described above into a single stateful object, which makes evaluation difficult to control and to reason about.
    In the long term we would like to remove `VarInfo` entirely.
    
    If you are using DynamicPPL internals and are unsure how to adapt your old code that uses `VarInfo` to the new evaluation framework, please check out the [Migration guide](./migration.md), or [open an issue on DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl/issues).
    We are happy to help!

## Accumulators

We will talk about accumulators first, since we will need to use them to demonstrate the other concepts.

Accumulators are used to collect information during the evaluation of a model.
Each accumulator has a different function: there is a [`LogPriorAccumulator`](@ref) for accumulating the log-probability of the prior, a [`LogLikelihoodAccumulator`](@ref) for accumulating the log-probability of the likelihood, a [`RawValueAccumulator`](@ref) for collecting raw (i.e. untransformed) parameter values, and so on.

The beauty of accumulators is that they are completely separate from one another; that means that you can mix and match them as needed, and avoid computing any information that you don't need.
For example, if you don't need to know the likelihood, you can drop the `LogLikelihoodAccumulator`, which will avoid unnecessary calls to `logpdf(dist, x)` for any observed `x`.

You can specify which accumulators you want to use by passing them as arguments to `OnlyAccsVarInfo`.
If no arguments are passed, a set of default accumulators (log-prior, log-likelihood, and log-Jacobian) are used.

```@example 1
# Here, we set up an `OnlyAccsVarInfo` that only contains one accumulator.
accs = OnlyAccsVarInfo(LogPriorAccumulator())

# When calling init!!, we need to specify all three components. For now, just
# focus on the accumulators, and we'll talk about the other two components later.
init_strategy = InitFromPrior()
transform_strategy = UnlinkAll()

retval, accs = DynamicPPL.init!!(model, accs, init_strategy, transform_strategy)
accs
```

There are a number of functions that you can call on an `OnlyAccsVarInfo` to extract the information.
The most low-level one is `getacc`, which given an accumulator name (a `Symbol`) returns a specific accumulator; see the [accumulator docs](@ref accumulators-overview) for more details on this function.

```@example 1
getacc(accs, Val(:LogPrior)).logp
```

It is often more convenient though to work with higher-level functions which directly extract the information that you need.
For example, `getlogprior` will extract the log-prior from the `LogPriorAccumulator` (if one exists):

```@example 1
getlogprior(accs)
```

The [page on existing accumulators](@ref existing-accumulators) describes the ones that are provided in DynamicPPL.
Many of these will come with higher-level convenience functions: currently we define (and export) [`getlogprior`](@ref), [`getloglikelihood`](@ref), [`getlogjac`](@ref), [`getlogjoint`](@ref), [`getlogprior_internal`](@ref), [`getlogjoint_internal`](@ref), [`get_raw_values`](@ref), and [`get_vector_values`](@ref).

DynamicPPL also allows you to add your own custom accumulators, which can be used to extract (or process) information obtained during model evaluation.
This often means that you can avoid running the model multiple times just to extract different pieces of information.

## Initialisation strategies

When evaluating a model, we need to assign values to the random variables in the model.
An *initialisation strategy* specifies how these values are generated.

As a very simple example, let's say we want to generate values for `x` and `y` by sampling
from the prior.
DynamicPPL provides [`InitFromPrior()`](@ref) for this purpose:

```@example 1
accs = OnlyAccsVarInfo()
init_strategy = InitFromPrior()
transform_strategy = UnlinkAll()

retval, accs = DynamicPPL.init!!(model, accs, init_strategy, transform_strategy)
retval
```

In the return value, we see that both `x` and `y` have been drawn from the prior.
This is an inherently random process; if you run the above code multiple times, you will get
different values for `x` and `y` each time.
Initialisation strategies that involve randomness can be controlled by passing an `rng` object as the first argument to `DynamicPPL.init!!`:

```@example 1
using Random

retval1 = first(init!!(Xoshiro(468), model, accs, init_strategy, transform_strategy))
retval2 = first(init!!(Xoshiro(468), model, accs, init_strategy, transform_strategy))
retval1 == retval2
```

Apart from `InitFromPrior()`, the main initialisation strategy that you are likely to use is [`InitFromParams()`](@ref), where you can manually specify the values of the parameters that you are interested in.

```@example 1
# See the VarNamedTuple docs for examples.
params = @vnt begin
    x := 1.0
    y := 0.5
end

init_strategy = InitFromParams(params)
retval, accs = DynamicPPL.init!!(model, accs, init_strategy, transform_strategy)

retval
```

How do we know that the values of `x` and `y` that we specified in `params` are actually being used?
We can determine this by inspecting the data inside the accumulators.
Because both `x` and `y` are random variables (i.e., not conditioned data), their log-probabilities fall under the prior.
(Note that specifying `InitFromParams` is not the same as conditioning the model on those values!)

```@example 1
getlogprior(accs)
```

We can compare this to what we would get if we were to manually evaluate the log-probability:

```@example 1
logpdf(Normal(), 1.0) + logpdf(Beta(2, 2), 0.5)
```

## Transform strategies

Let's finally turn our attention to the transform strategy argument.
In the example above, we used `UnlinkAll()`, which means that the model is to be evaluated in 'unlinked' space: in DynamicPPL this refers to the original space of the parameters, without any transformations.

Often it is necessary to evaluate the model in a different space.
For example, we might be using an optimisation algorithm to find the maximum likelihood estimate.
In such cases it is often more convenient to work in unconstrained Euclidean space, where we pass in a value `transformed_y` which can be any real number, and the actual value of `y` in the model is obtained by `raw_y = logistic(transformed_y)`, which maps real numbers to the interval `(0, 1)`.

```@example 1
using StatsFuns: logistic, logit

transformed_y = 3.0
raw_y = logistic(transformed_y)
```

The use of transformations also means that we need to be careful about computing log-probabilities, because the probability associated with `transformed_y` is *not* equivalent to

```@example 1
logpdf(Beta(2, 2), raw_y)
```

but rather

```@example 1
using ChangesOfVariables: with_logabsdet_jacobian

logpdf(Beta(2, 2), raw_y) + last(with_logabsdet_jacobian(logistic, transformed_y))
```

where the Jacobian term accounts for the change of variables.
(If you aren't familiar with this concept, the [main Turing docs have an introduction on it](https://turinglang.org/docs/developers/transforms/distributions/).)

The transform strategy allows you to specify which variables are to be transformed to Euclidean space, which in turn determines whether the Jacobian term is accumulated or not.

**Importantly, the transform strategy is separate from the initialisation strategy**: this means that the initialisation strategy can provide values in untransformed space, and the transform strategy can 'reinterpret' them as being in transformed space, and then apply the necessary transformations and Jacobian corrections.

For example:

```@example 1
params = @vnt begin
    # These are always in untransformed space.
    x := 1.0
    y := 0.5
end
init_strategy = InitFromParams(params)

# This transform strategy specifies that all variables should be linked.
transform_strategy = LinkAll()

_, accs = DynamicPPL.init!!(model, accs, init_strategy, transform_strategy)
accs
```

We see that the prior term is unchanged from the `UnlinkAll()` evaluation before.
However, in constrast, the `LogJacobianAccumulator` is no longer empty; it contains the log-Jacobian term for the *forward* transform (to unconstrained space).
Since `x` is already unconstrained, this term is zero for `x`, but for `y` it is non-zero, and it is equal to

```@example 1
# `logit` is the *forward* transform from (0, 1) to ℝ.
last(with_logabsdet_jacobian(logit, 0.5))
```

That means that the log-probability in the transformed space is given by

```@example 1
getlogprior(accs) - getlogjac(accs)
```

You might ask: given that we specified parameters in untransformed space, how do we then retrieve the parameters in transformed space?
The answer to this is to use an accumulator (no surprises there!) that collects the transformed values.
Specifically, a `VectorValueAccumulator` collects vectorised forms of the parameters, which may either be [`VectorValue`](@ref)s or [`LinkedVectorValue`](@ref)s.

```@example 1
accs = OnlyAccsVarInfo(VectorValueAccumulator())
_, accs = DynamicPPL.init!!(model, accs, init_strategy, transform_strategy)
accs
```

Of course, in an actual application you should probably use all the accumulators at the same time so that you only run the model once.

If you need to extract a concatenated vector of parameters from this, e.g. to pass to an optimisation algorithm, you can use

```@example 1
get_vector_values(accs)
```

If you are thinking of doing something like this, you *probably* also want to use [`LogDensityFunction`](@ref ldf) instead, and should skip ahead to that page.

## Further reading

The rest of the DynamicPPL documentation goes into these three components in much more detail.
We also show you there how you can create your own custom initialisation strategies, transform strategies, and accumulators, so that you can extend the evaluation framework to suit your own needs.
