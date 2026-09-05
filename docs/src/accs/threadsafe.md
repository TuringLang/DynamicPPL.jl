# Thread-safe accumulation

DynamicPPL contains a 'thread-safe model evaluation mode', which can be accessed by calling [`DynamicPPL.setthreadsafe`](@ref) on a model.

```@example 1
using DynamicPPL, Distributions

@model function g(y)
    x ~ Normal()
    Threads.@threads for i in eachindex(y)
        y[i] ~ Normal(x)
    end
end
y = [2.0, 3.0, 4.0]
model = setthreadsafe(g(y), true)
```

This is accomplished by creating one copy of each accumulator per task (using
`DynamicPPL.split`) and merging each task's accumulator with `DynamicPPL.combine` after
model evaluation.

If you implement an accumulator, you must implement `split` and `combine` for it to work
correctly in thread-safe mode.

Each accumulator sees only the tilde-statements that were executed by its own task.
After merging the results from all tasks, the final accumulator should be equivalent to
one obtained by single-threaded evaluation, modulo ordering.
Do not aggregate, copy, serialize, reset, or reconfigure the `ThreadSafeVarInfo` while its
tasks are running because an accumulator may update mutable state in place.
Because the accumulation process is not always commutative, you may in general end up with a different ordering of results.
However, for many accumulators such as log-probability accumulators, this is not an issue.

We can see this in action if we step through the internal DynamicPPL calls.
(Note that calling `DynamicPPL.init!!` on a model where thread-safe mode has been enabled will automatically perform these steps for you.)

```@example 1
Threads.nthreads()
```

```@example 1
vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.LogLikelihoodAccumulator())
tsvi = DynamicPPL.ThreadSafeVarInfo(vi)
isempty(tsvi.accs_by_task)
```

The dictionary is initially empty. A task adds an accumulator when it first encounters a
tilde-statement.

```@example 1
x = 1.0
context = DynamicPPL.InitContext(InitFromParams((; x=x)), UnlinkAll())
_, tsvi = DynamicPPL._evaluate!!(model, context, tsvi)
length(tsvi.accs_by_task)
```

The result is the number of tasks that executed at least one tilde-statement.

Finally, `getacc` combines the per-task accumulators into one accumulator.

```@example 1
output_acc = DynamicPPL.getacc(tsvi, Val(:LogLikelihood))
```

We can check whether this is correct:

```@example 1
output_acc.logp ≈ sum(logpdf.(Normal(x), y))
```

Any output obtained from an accumulator can be accumulated correctly in thread-safe mode.
DynamicPPL can therefore provide full thread safety when all required outputs come from
accumulators.

The main situation where this is not yet true is when using a full `VarInfo`, which stores a VarNamedTuple in its `varinfo.values` field.
Modifications to this field are currently not thread-safe.
However, the `values` VNT is entirely equivalent to a `VectorValueAccumulator`.

In the near future it should hopefully be possible to use a `OnlyAccsVarInfo` with a `VectorValueAccumulator` instead of a full `VarInfo`, which would allow DynamicPPL to be fully thread-safe (though see also [this issue](https://github.com/TuringLang/DynamicPPL.jl/issues/1266) for another caveat).
