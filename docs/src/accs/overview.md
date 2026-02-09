# [Overview](@id accumulators-overview)

Accumulators are objects in DynamicPPL which collect the results of computations in each tilde-statement.

Consider a tilde-statement `x ~ Beta(2, 2)`.
There are several things going on in here (we will discuss this in full detail in the [Model evaluation](../evaluation.md) page).
Loosely speaking:

  - We need to get a value for `x` that is consistent with the distribution `Beta(2, 2)`;
  - That value may or may not be transformed to linked space;
  - We need to calculate the log-density of that value.

On top of that, we may want to store other information, such as:

  - The `VarName` itself, i.e. `@varname(x)`;
  - The distribution;

and so on.

These pieces of information are collected by accumulators.
Each different accumulator is responsible for a different piece of information: thus, for example, log-likelihoods and log-priors are collected in separate accumulators.
By choosing the right set of accumulators, we can control what information is collected during model evaluation.
This allows us to perform exactly the necessary amount of computation we need to do.

Furthermore, accumulators are completely independent: inside the `accumulate_...!!` method of an accumulator, it is not possible to access the state of other accumulators.
This creates a modular design, where new accumulators can be defined and added without worrying about potential interactions with existing ones.

## The `AbstractAccumulator` API

Accumulators must subtype [`DynamicPPL.AbstractAccumulator`](@ref), whose docstring explains the required interface:

```@docs; canonical=false
AbstractAccumulator
```

The reason each accumulator must have a different name is because (for type stability reasons) accumulators are stored in a `NamedTuple`, and Julia requires that the names of fields in a `NamedTuple` be unique.

`reset` is called when starting a new model evaluation, to ensure that previously accumulated results do not affect the current evaluation.

The central two methods though are `accumulate_assume!!` and `accumulate_observe!!`.
These two methods are called whenever a tilde-statement is encountered during model evaluation.
As can be seen from their signatures, they receive all the information that we have discussed above about tilde-statements.

Formally, accumulators can be seen as [a monoidal structure](https://en.wikipedia.org/wiki/Monoid#Monoids_in_computer_science) where:

  - `reset` specifies the identity (indeed for some accumulators in DynamicPPL we define a function called `zero`);
  - `accumulate_assume!!` and `accumulate_observe!!` collectively specify the binary operation;
  - the accumulated result is a fold over all tilde-statements in the model.

Note that there is no guarantee that the binary operation be commutative (although in practice for many accumulators it is).

## An example

As an example, let's consider an accumulator that associates `VarNames` with their log-densities.
We will store these in an `OrderedDict{VarName,Tuple{Bool,Float64}}` where the `Bool` indicates whether the variable was observed (`true`) or assumed (`false`), and the `Float64` is the log-density.

!!! note
    
    This is very similar to the `pointwise_logdensities` functionality in DynamicPPL.

We define the accumulator as follows.
As one may have guessed from the above, much of the definition is really just boilerplate to make things run smoothly.
The interesting behaviour is in the two `accumulate_...!!` methods, where we compute the log-density and store it in the `OrderedDict`.

```@example 1
using DynamicPPL, OrderedCollections, Distributions

struct VarNameLogpAccumulator <: DynamicPPL.AbstractAccumulator
    logps::OrderedDict{VarName,Tuple{Bool,Float64}}

    VarNameLogpAccumulator() = new(OrderedDict{VarName,Tuple{Bool,Float64}}())
end

# Note, it's a good idea to define the symbol as a top-level `const`, since it's often
# necessary to refer to it elsewhere. We'll see this happen later.
const VARNAMELOGP_NAME = :VarNameLogp
DynamicPPL.accumulator_name(::VarNameLogpAccumulator) = VARNAMELOGP_NAME
DynamicPPL.reset(acc::VarNameLogpAccumulator) = VarNameLogpAccumulator()
DynamicPPL.copy(acc::VarNameLogpAccumulator) = VarNameLogpAccumulator(copy(acc.logps))

function DynamicPPL.accumulate_assume!!(
    acc::VarNameLogpAccumulator,
    val,
    tval,
    logjac,
    vn::VarName,
    dist::Distribution,
    template,
)
    acc.logps[vn] = (false, logpdf(dist, val))
    return acc
end

function DynamicPPL.accumulate_observe!!(acc::VarNameLogpAccumulator, dist, val, vn)
    acc.logps[vn] = (true, logpdf(dist, val))
    return acc
end
```

In this implementation, our `accumulate_...!!` methods actually mutate the accumulator in place.
This is not mandatory; you can return a new accumulator if you prefer an immutable style.

To use this accumulator in a model evaluation, we need to wrap it in an `OnlyAccsVarInfo`.

!!! note "OnlyAccsVarInfo"
    
    As described on the [Model evaluation](../evaluation.md) page, an `OnlyAccsVarInfo` is just a wrapper around a set of accumulators.
    Don't worry about the name; it's mostly a historical artifact.

We can either do this by creating it from scratch, or by modifying an existing one (see the next section for details).
In this example, we'll create an `OnlyAccsVarInfo` from scratch using only our new accumulator.
Once we've evaluated the model, we can access the accumulated log-densities by reading it back from the accumulator:

```@example 1
@model function f(y)
    x ~ Normal()
    return y ~ Normal(x)
end
model = f(2.0)

accs = DynamicPPL.OnlyAccsVarInfo(VarNameLogpAccumulator())
_, accs = DynamicPPL.init!!(model, accs, InitFromParams((; x=1.0)), UnlinkAll())

# This is why we used a const.
output_acc = DynamicPPL.getacc(accs, Val(VARNAMELOGP_NAME))
```

Since we specified that `x` should be initialised to `1.0`, we should have that

```@example 1
output_acc.logps[@varname(x)] == (false, logpdf(Normal(), 1.0))
```

and

```@example 1
output_acc.logps[@varname(y)] == (true, logpdf(Normal(1.0), 2.0))
```

(Notice that because here we used an `OrderedDict`, the accumulation process is not commutative.)

## Working with `OnlyAccsVarInfo`

As shown above, users can choose exactly which accumulators to use during model evaluation in order to control what information is collected.
This section explains how to specify which accumulators to use, and how to extract the information again afterwards.

When creating an `OnlyAccsVarInfo`, if no accumulators are specified, there is a set of _default accumulators_ that are used:

```@example 1
OnlyAccsVarInfo()
```

As alluded to above, one can control this by passing the accumulators to the `OnlyAccsVarInfo` constructor:

```@example 1
accs = OnlyAccsVarInfo(LogLikelihoodAccumulator())
```

If you then use this `OnlyAccsVarInfo` to evaluate a model, only the specified accumulators will be used.

```@example 1
@model function demo_likelihood()
    x ~ Normal()
    return 1.0 ~ Normal(x)
end
model = demo_likelihood()
_, new_accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
new_accs
```

Instead of creating an `OnlyAccsVarInfo` from scratch, one can also add accumulators to an existing `OnlyAccsVarInfo` using [`setacc!!`](@ref) and [`setaccs!!`](@ref).
Note that these functions have very similar names!
`setacc!!` adds a single accumulator, while `setaccs!!` replaces the entire set of accumulators with a new set.

```@example 1
accs = OnlyAccsVarInfo()
# Add a new accumulator
accs = setacc!!(accs, RawValueAccumulator(false))
```

In the case of `setacc!!`, the accumulator will be _added_ to the existing set if it has a name (as defined by [`accumulator_name`](@ref)) that is not already present.
If an accumulator with the same name is already present, it will be _replaced_!

This means that you have to be careful to not accidentally create a name clash with an existing accumulator.
If you are defining your own accumulator, it is a good idea to define its name with a prefix that is unique to your package.

However, it does *also* mean that you can customise the behaviour of existing accumulators.
For example, if you intentionally define an accumulator with the same name as an existing one, you can use `setacc!!` to replace it.
Turing.jl uses this mechanism in SMC inference to replace the default log-likelihood accumulator with one that triggers particle reweighting each time the likelihood is incremented (i.e., each time a new observation is made).

`setaccs!!` is simpler; it replaces whatever came before it.
It takes a tuple of accumulators as input.

```@example 1
# This will create a set of default accumulators.
accs = OnlyAccsVarInfo()
```

```@example 1
# Replace all accumulators with just a RawValueAccumulator
new_acc = RawValueAccumulator(false)
accs = setaccs!!(accs, (new_acc,))
```

Once you have evaluated a model with your desired accumulators, you can extract the accumulated results using [`getacc`](@ref).

```@example 1
_, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())

# `getacc` returns the accumulator itself; `.values` accesses the actual values inside it.
raw_values = getacc(accs, Val(accumulator_name(new_acc))).values
```

There are convenience functions [`getlogprior`](@ref), [`getloglikelihood`](@ref), [`getlogprior`](@ref), and [`get_raw_values`](@ref) that extract the corresponding accumulators' wrapped values directly from the `OnlyAccsVarInfo`.
Note that these functions will throw an error if the corresponding accumulator is not present in the `OnlyAccsVarInfo`.

```@example 1
get_raw_values(accs)
```

[The next page](./existing.md) gives an overview of the existing accumulators in DynamicPPL; these provide common functionality which you can make use of, without having to write your own accumulators from scratch.
