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
vi = DynamicPPL.VarInfo(DynamicPPL.LogLikelihoodAccumulator())
tsvi = DynamicPPL.ThreadSafeVarInfo(vi)
isempty(tsvi.accs_by_task)
```

The dictionary is initially empty. A task adds an accumulator when it first encounters a
tilde-statement.

```@example 1
x = 1.0
context = Context(InitFromParams((; x=x)), UnlinkAll())
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

Parameter values are also accumulator outputs. `VectorValueAccumulator` and
`RawValueAccumulator` use task-local storage and combine their results after evaluation.
This does not make arbitrary mutations in the model body or shared mutable
initialisation strategies thread-safe.

Because task accumulators are combined in an unspecified order, the order of collected
variables can differ between evaluations, including variables defined outside the
threaded region. A `VarInfo` or `LogDensityFunction` built from such an evaluation
inherits that order. Each function is consistent with its own mapping, but using
`internal_values_as_vector` (`vi[:]`) from another `VarInfo` can silently give the wrong
density. Obtain vectors with that function's mapping using
[`VectorParamAccumulator`](@ref), or pass vectorised values with matching transforms to
[`to_vector_params`](@ref), as described in
[Obtaining new vectorised parameters](@ref ldf-vector-params).

## Containers under thread-safe evaluation

Under thread-safe evaluation, every container on a tilde assignment's path must
retain its identity (`===`); otherwise the assignment throws an `ArgumentError`.
The written leaf is exempt. Intermediate array slices use views, so
`x[1:2][i] ~ Normal()` writes only element `i` and `x[:, j] .~ Normal()` writes
in place. A final slice, as in `x[:, j] ~ MvNormal(...)`, is written into its parent.
Concurrent writes to the same element still require synchronisation by the caller.

Allocate mutable storage with the parameter element type via `::Type{T}=Float64`:
use `Vector{T}(undef, n)` for scalar tildes or `Matrix{T}` with
`x[:, i] ~ MvNormal(...)` for vector-valued tildes. Replace a `NamedTuple`, tuple,
or static array containing scalar parameters with a `Vector`, or use
plain scalar variables. This rule also applies to single-task and task-local
immutable containers. A `NamedTuple` of vectors can stay unchanged while its
vectors are updated in place. Immutable wrappers such as an `OffsetArray` must
also retain their identity; replacing their backing storage is rejected.

Imputing missing entries in a data argument under automatic differentiation
requires storage that can hold AD values. Pass the data as
`Vector{Union{Missing,Real}}` to retain the same log density and gradient as serial
evaluation. Serial evaluation with thread-safe mode disabled retains its usual
widening behaviour.
