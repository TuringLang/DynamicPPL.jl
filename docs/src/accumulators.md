# Accumulators

Accumulators are objects in DynamicPPL which collect the results of computations in each tilde-statement.

Consider a tilde-statement `x ~ Beta(2, 2)`.
There are several things going on in here (we will discuss this in full detail in the [Model evaluation](flow.md) page).
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

To use this accumulator in a model evaluation, we need to add it into a VarInfo.
We can either do this by creating a `VarInfo` from scratch, or by calling `DynamicPPL.setacc!!` or `DynamicPPL.setaccs!!` on an existing `VarInfo`.

!!! note
    
    These two functions have very similar names, so be careful to use the right one! `setacc!!` adds a single accumulator to a VarInfo, whereas `setaccs!!` replaces all the VarInfo's accumulators with a new set.

In this example, we'll create a `VarInfo` from scratch using only our new accumulator.
To minimise the computational overhead, we use an `OnlyAccsVarInfo`, which is a slimmed down version of a `VarInfo` that only contains accumulators.
(This is a minor detail; don't worry about it if you aren't familiar with `VarInfo` types.)
Once we've evaluated the model, we can access the accumulated log-densities by reading it back from the accumulator:

```@example 1
@model function f(y)
    x ~ Normal()
    return y ~ Normal(x)
end
model = f(2.0)

vi = DynamicPPL.OnlyAccsVarInfo((VarNameLogpAccumulator(),))
_, vi = DynamicPPL.init!!(model, vi, InitFromParams((; x=1.0)))

# This is why we used a const.
output_acc = DynamicPPL.getacc(vi, Val(VARNAMELOGP_NAME))
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

## Thread-safe accumulation

DynamicPPL contains a 'thread-safe model evaluation mode', which can be accessed by calling [`DynamicPPL.setthreadsafe`](@ref) on a model.

```@example 1
@model function g(y)
    x ~ Normal()
    Threads.@threads for i in eachindex(y)
        y[i] ~ Normal(x)
    end
end
y = [2.0, 3.0, 4.0]
model = setthreadsafe(g(y), true)
```

This is accomplished by creating one copy of each accumulator per thread (using `DynamicPPL.split`), and then after the model evaluation is complete, merging the result of each thread's accumulator with `DynamicPPL.combine`.

Each accumulator sees only the tilde-statements that were executed on its own thread.
However, the intent is that after merging the results from all threads, the final accumulator should be equivalent to what would have been obtained by a single-threaded evaluation (modulo ordering).
Because the accumulation process is not always commutative, you may in general end up with a different ordering of results.
However, for many accumulators such as log-probability accumulators, this is not an issue.

We can see this in action if we step through the internal DynamicPPL calls.
(Note that calling `DynamicPPL.evaluate!!` on a model where thread-safe mode has been enabled will automatically perform these steps for you.)

```@example 1
Threads.nthreads()
```

```@example 1
vi = DynamicPPL.OnlyAccsVarInfo((DynamicPPL.LogLikelihoodAccumulator(),))
tsvi = DynamicPPL.ThreadSafeVarInfo(vi)
tsvi.accs_by_thread
```

(Here it actually creates a vector of length `maxthreadid()`.
This is slightly hacky, see the warning below and links therein for more discussion.)

```@example 1
x = 1.0
model = setleafcontext(model, DynamicPPL.InitContext(InitFromParams((; x=x))))
_, tsvi = DynamicPPL._evaluate!!(model, tsvi)
tsvi.accs_by_thread
```

In the above output, the accumulators that have non-zero log-likelihoods are the ones corresponding to the threads that executed tilde-statements.

Finally, to collapse the per-thread accumulators into a single accumulator, we can call `getacc`.
This does the `combine` step for us.

```@example 1
output_acc = DynamicPPL.getacc(tsvi, Val(:LogLikelihood))
```

We can check whether this is correct:

```@example 1
output_acc.logp ≈ sum(logpdf.(Normal(x), y))
```

!!! warning
    
    The current implementation of thread safety, with one accumulator per thread, is not fully safe since it relies on indexing into a vector with `threadid()`. See [this issue](https://github.com/TuringLang/DynamicPPL.jl/issues/924) for details. In practice, though, we have not observed any problems with the current approach.
    
    There is also a possibility that DynamicPPL may shift to using 'atomic' accumulators in the future, where only one set of accumulators is maintained, but modifications to it must be performed atomically. See [this draft PR](https://github.com/TuringLang/DynamicPPL.jl/pull/1137) for details.

Ignoring the caveats above, it can be generally said that **any output that is obtained from an accumulator can be accumulated correctly in a thread-safe manner**.
In other words, full thread safety in DynamicPPL is possible as long as all the outputs you need are obtained from accumulators.

The main situation where this is not yet true is when using a full `VarInfo`, which stores a VarNamedTuple in its `varinfo.values` field.
Modifications to this field are currently not thread-safe.
However, the `values` VNT is entirely equivalent to a `VectorValueAccumulator`.
In the near future it should be possible to use a `OnlyAccsVarInfo` with a `VectorValueAccumulator` instead of a full `VarInfo`, which would allow DynamicPPL to be fully thread-safe.
