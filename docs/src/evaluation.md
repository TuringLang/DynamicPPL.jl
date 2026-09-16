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
    accs::DynamicPPL.VarInfo,
    init_strategy::DynamicPPL.AbstractInitStrategy,
    transform_strategy::DynamicPPL.AbstractTransformStrategy,
)
```

which returns a tuple of the model's return value (the NamedTuple `(x=x, y=y)` in the example above) and the accumulators after evaluation.

The equivalent explicit-context call is:

```@example 1
using Random: Xoshiro

context = Context(Xoshiro(1), InitFromPrior(), UnlinkAll())
retval, accs = evaluate!!(model, context, VarInfo());
```

## [Evaluation inputs and outputs](@id evaluation-inputs-outputs)

Evaluation separates the inputs that determine a model run from the outputs it records,
as proposed in [#1469](https://github.com/TuringLang/DynamicPPL.jl/issues/1469).

| Object    | Responsibility                                                        |
|:--------- |:--------------------------------------------------------------------- |
| `Model`   | Model function, arguments, and conditioned or fixed data              |
| `Context` | RNG, initialisation strategy, and requested transform strategy        |
| `VarInfo` | Output accumulators, with no separate parameter or transform storage  |
| `retval`  | The model body's ordinary Julia return value, distinct from its trace |

For a latent statement such as `x ~ Normal()`, the context's initialisation strategy
supplies `x`. Its transform strategy determines the transformed value and Jacobian.
Accumulators receive those results to compute densities or record values; evaluation
does not read previously recorded parameters from `VarInfo`.

For a conditioned observation such as `y ~ Normal(x, 1)`, the model supplies `y` and
the likelihood accumulator scores it. Literal observations such as `0 ~ Normal(x, 1)`
use the same observation path. Fixed values are not scored, and tracked assignments
such as `z := x + y` are recorded when requested. None of these operations uses the
context to select a latent value.

The context belongs to the evaluation, not to `Model`, and is passed to nested submodels.
Inside a model body, `__context__` refers to this context; use `rand(__context__.rng, ...)`
for explicit random draws controlled by the evaluation's RNG. `init!!` constructs a `Context`
and calls `evaluate!!`; custom value selection belongs in an initialisation strategy,
not a custom context type.

`VarInfo(acc1, acc2, ...)` selects the outputs to collect. `VarInfo()` collects only
log prior, log likelihood, and log Jacobian; it does not record parameter values.
Every evaluation resets its accumulators, so a value recorded in one run disappears
if its site is skipped in the next. Always retain the returned `VarInfo`: `!!` operations
may replace their input.

To reuse outputs as inputs, extract the recorded values and construct a new context
explicitly. For example, sample the model above, then evaluate it at the same parameters:

```@example 1
rng = Xoshiro(1)
context = Context(rng, InitFromPrior(), LinkAll())
retval, recorded = evaluate!!(model, context, VarInfo(RawValueAccumulator(false)))

params = get_raw_values(recorded)
context = Context(rng, InitFromParams(params, nothing), UnlinkAll())
repeated, scores = evaluate!!(model, context, VarInfo())

@assert repeated == retval
@assert iszero(getlogjac(scores))
getlogjoint(scores)
```

Here `nothing` disables fallback sampling: a missing parameter is an error, not a request
to consult previous outputs. The second context explicitly requests unlinked outputs,
independently of the first run's transform strategy. To reuse vectorised values instead,
record a [`VectorValueAccumulator`](@ref) and pass `get_vector_values(recorded)` to
`InitFromParams`; see [Storing vectorised and raw values](accs/values.md).

This separation specifies data flow, not purity: evaluation can advance the RNG, and
ordinary Julia mutations in a model body still take effect.
For density evaluation, `~` sites read supplied parameters rather than sampling;
see [Randomness in density evaluation](@ref ldf-rng).

## Accumulators

We will talk about accumulators first, since we will need to use them to demonstrate the other concepts.

Accumulators are used to collect information during the evaluation of a model.
Each accumulator has a different function: there is a [`LogPriorAccumulator`](@ref) for accumulating the log-probability of the prior, a [`LogLikelihoodAccumulator`](@ref) for accumulating the log-probability of the likelihood, a [`RawValueAccumulator`](@ref) for collecting raw (i.e. untransformed) parameter values, and so on.

The beauty of accumulators is that they are completely separate from one another; that means that you can mix and match them as needed, and avoid computing any information that you don't need.
For example, if you don't need to know the likelihood, you can drop the `LogLikelihoodAccumulator`, which will avoid unnecessary calls to `logpdf(dist, x)` for any observed `x`.

You can specify which accumulators you want to use by passing them as arguments to `VarInfo`.
If no arguments are passed, a set of default accumulators (log-prior, log-likelihood, and log-Jacobian) are used.

```@example 1
# Here, we set up a `VarInfo` that only contains one accumulator.
accs = VarInfo(LogPriorAccumulator())

# When calling init!!, we need to specify all three components. For now, just
# focus on the accumulators, and we'll talk about the other two components later.
init_strategy = InitFromPrior()
transform_strategy = UnlinkAll()

retval, accs = DynamicPPL.init!!(model, accs, init_strategy, transform_strategy)
accs
```

There are a number of functions that you can call on a `VarInfo` to extract the information.
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
accs = VarInfo()
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
Specifically, a `VectorValueAccumulator` collects vectorised forms of the parameters: that is, `TransformedValue{V,T}` where `V<:AbstractVector`.

```@example 1
accs = VarInfo(VectorValueAccumulator())
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
