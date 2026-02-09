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

This is accomplished by creating one copy of each accumulator per thread (using `DynamicPPL.split`), and then after the model evaluation is complete, merging the result of each thread's accumulator with `DynamicPPL.combine`.

**This means that if you are implementing your own accumulator, you will need to implement the `split` and `combine` methods for it in order for it work correctly in thread-safe mode.**

Each accumulator sees only the tilde-statements that were executed on its own thread.
However, the intent is that after merging the results from all threads, the final accumulator should be equivalent to what would have been obtained by a single-threaded evaluation (modulo ordering).
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
tsvi.accs_by_thread
```

(Here it actually creates a vector of length `maxthreadid()`.
This is slightly hacky, see the warning below and links therein for more discussion.)

```@example 1
x = 1.0
model = setleafcontext(model, DynamicPPL.InitContext(InitFromParams((; x=x)), UnlinkAll()))
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

In the near future it should hopefully be possible to use a `OnlyAccsVarInfo` with a `VectorValueAccumulator` instead of a full `VarInfo`, which would allow DynamicPPL to be fully thread-safe (though see also [this issue](https://github.com/TuringLang/DynamicPPL.jl/issues/1266) for another caveat).
