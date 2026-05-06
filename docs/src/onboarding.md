# Contributor onboarding

This page summarizes recurring lessons from DynamicPPL and AbstractPPL history
for contributors who are new to Julia, Turing.jl, or DynamicPPL internals.

The source pass covered GitHub history available on 2026-05-06. For
DynamicPPL, that included 422 issues, 957 pull requests, 6,958 issue/PR
comments, 3,726 PR reviews, and 5,176 inline review comments. For AbstractPPL,
that included 46 issues, 101 pull requests, 654 issue/PR comments, 332 PR
reviews, and 441 inline review comments. Linked issues and PRs are
representative starting points, not current API documentation.

## What DynamicPPL Does

DynamicPPL is the modelling and evaluation layer under Turing.jl. It provides
`@model`, tilde (`~`) statement handling, conditioning, fixing, parameter
transforms, accumulators, and log-density interfaces for samplers and automatic
differentiation. It uses AbstractPPL for shared interfaces such as `VarName`,
contexts, and evaluator protocols.

A useful mental model:

 1. `@model` lowers user code into a model function.
 2. Each ordinary `~` statement becomes an assume or observe statement.
 3. Contexts and initialisation strategies decide where values come from.
 4. Accumulators decide which quantities are collected.
 5. `LogDensityFunction` maps named model parameters to flat vectors.

Start with these docs:

  - [Model evaluation](evaluation.md)
  - [Tilde-statements](tilde.md)
  - [Initialisation strategies](init.md)
  - [Transform strategies](transforms.md)
  - [Accumulators](accs/overview.md)
  - [VarNamedTuple](vnt/motivation.md)
  - [LogDensityFunction](ldf/overview.md)

## Core Lessons

### Prefer explicit evaluation state

For new evaluation code, prefer explicit initialisation strategies and
accumulators over adding more responsibilities to `VarInfo`. `VarInfo` remains
important, but fast paths should carry only the state they need.

A common migration shape is:

```julia
evaluate!!(model, varinfo)
```

to:

```julia
init!!(
    model,
    OnlyAccsVarInfo(accumulators...),
    InitFromParams(varinfo.values),
    varinfo.transform_strategy,
)
```

The exact strategy and accumulator set depend on the caller.

### Use names and shapes carefully

Use `@varname(x)` and `@varname(x[1])`; avoid manual construction of indexed
`VarName`s. Use subsumption for containment checks: `@varname(x)` can cover
`@varname(x[1])`, but they are not equal.

`VarName` display, sorting, prefixing, unprefixing, and serialization are
downstream-facing interface behaviour. Test nested fields, indices, ranges,
`Colon`, and non-standard indices when changing them. Avoid broad `Base`
overloads such as generic `get(obj, vn)` unless the method is clearly owned.

`VarNamedTuple` is the preferred internal container for named parameter values
where supported. Convert user-facing `NamedTuple` or `Dict{VarName}` inputs at
API boundaries. Preserve templates, shapes, and index structure so values can
round-trip between named form and flat vectors. Avoid large mostly-empty shadow
arrays and keep eltypes concrete in hot paths.

### Keep `!!` return values

DynamicPPL uses BangBang-style `!!` functions. They may mutate in place or
return a replacement object. Always use the returned value.

```julia
vi = accumulate_assume!!(vi, value, tval, logjac, vn, dist, template)
```

If your function calls a `!!` function, it usually needs to return the updated
state as well.

### Treat `@model` as Julia code

`@model` lowering must preserve ordinary Julia behaviour as well as PPL
semantics. For compiler changes, test positional and keyword arguments,
defaults, splatting, closures, interpolation, return values, no-observation
models, and data- or parameter-dependent control flow.

Macro hygiene matters. User variables, generated temporaries, and globals
should not capture each other accidentally. Returned quantities are
user-visible and are distinct from accumulated random variables.

DynamicPPL tracks variables through tilde statements. A left-hand-side value can
be treated as a model variable even when it was derived earlier in the model.

```julia
@model function f()
    x ~ Normal()
    y = x + 1
    return y ~ Normal()
end
```

If the intent is to add a likelihood term for a derived value, prefer
`@addlogprob!` or a clearer model structure. Do not copy old `.~` examples; the
dot-tilde pipeline was removed.

Passing `missing` can affect whether a value is observed or latent. Add tests
for the exact data shape you support, especially arrays with missing values,
arrays of arrays, and mutable structs.

### Test contexts with nested models

Contexts change model evaluation without rewriting the model body. `condition`,
`fix`, `decondition`, `unfix`, `to_submodel`, and prefixes all interact.

Prefer `condition`, `fix`, and `to_submodel` over hardcoded special cases. Use
the same `VarName` semantics as the tilde pipeline. Add nested-submodel tests
when changing contexts, prefixes, conditioning, or fixing.

### Know which space values live in

DynamicPPL moves between constrained model space and unconstrained sampler
space. Be explicit about which space each value lives in.

  - `val`: constrained model-space value used for distribution densities.
  - `tval`: `TransformedValue`, which may contain a linked value.
  - `logjac`: log absolute Jacobian contribution from the link transform.
  - `getlogjoint`: constrained-space log joint.
  - `getlogjoint_internal`: internal log density for sampler-facing paths.
  - `vi[:]`: internal stored vector; do not assume it is in distribution support.

`LogDensityFunction` is the usual boundary for HMC/NUTS, optimisers, and AD.
When changing log-density or transform code, test the relevant AD backends.
Avoid compiled ReverseDiff tapes for models whose control flow depends on
parameter values.

Evaluator APIs should separate structural preparation from AD-specific
preparation. `!!` evaluator and gradient APIs may reuse internal buffers, so
copy results before storing them long term.

## Julia Engineering Practices

  - Measure performance-sensitive changes. Small edits can affect inference,
    allocations, invalidation, and downstream packages.
  - Check type stability with `@inferred`, `@code_warntype`, and focused tests.
  - Benchmark generated functions, macro output, and hot-path refactors.
  - Keep field types and collection eltypes concrete in hot paths.
  - Avoid unnecessary static parameters. Julia specializes on most ordinary
    argument types, but is conservative for `Type`, `Function`, and `Vararg`.
  - Use `f(x, ::Type{T}) where {T}` when the type itself must specialize.
  - Prefer dispatch and small protocol functions over large conditional blocks.
  - Put backend-specific behaviour in package extensions or narrow integration
    layers when possible.
  - Keep interface packages lightweight. Avoid heavy dependencies unless the
    interface truly owns them.
  - Do not rely on transitive dependencies. Direct API use should have an
    explicit dependency that can be version-bound and tested.
  - Use accessors for downstream needs instead of exposing internal fields.
  - Prefer `Base.maybeview` over eager slicing when indexed access should avoid
    allocations but still support tuples and scalar indexing.
  - Avoid fragile output-type prediction. When possible, compute an initial value
    and allocate caches from the observed value.

## Copying, Accumulators, and Threading

Be explicit about aliasing. Copy stored values when later mutation by model code
would otherwise change accumulated results. Use the cheapest correct copy:
`copy` or `collect` is often enough, while `deepcopy` can be much slower.

Accumulators collect outputs from model execution, such as log probabilities,
raw values, vector values, pointwise log densities, and returned values. Add
only the accumulators you need. `copy(acc)` must not accidentally share mutable
internal state.

Avoid designs that depend on `Threads.threadid()` indexing. Promote accumulator
storage when thread-safe evaluation must hold AD tracer types. Treat threaded
assume support as subtle unless current docs and tests cover the exact case.

## Documentation, Tests, and CI

  - Keep test files self-contained.
  - Use `DynamicPPL.TestUtils` models with known ground truth when possible.
  - Add nested-submodel tests for contexts, prefixes, conditioning, or fixing.
  - Add AD tests for log-density, transform, vector-parameter, or `run_ad`
    changes.
  - Add type-stability or allocation tests for hot paths.
  - Add round-trip tests for flattening and unflattening changes.
  - Run JuliaFormatter v1 with Blue style before submitting.
  - Keep doctests deterministic. Use `StableRNGs` when examples print random
    values.
  - Use plain `julia` blocks for examples that are illustrative but should not be
    checked.
  - Put dependencies in the narrowest environment that owns them: runtime,
    extension, test, or docs.
  - Treat docs, Aqua, JET, formatting, and extension-loading failures as part of
    the change.

## API and Review Norms

  - Breaking changes need explicit justification and deprecation or versioning
    plans.
  - Internal names are not public API just because downstream packages use them,
    but Turing.jl impact still matters.
  - Prefer composable operators over special-case syntax.
  - Document and test new user-facing API with examples.
  - Generated or AI-assisted code still needs manual understanding, focused
    tests, and a clear rationale.
  - Include benchmark numbers for performance-sensitive changes.

## Further Reading

  - Evaluation state and `VarInfo`: [#1132](https://github.com/TuringLang/DynamicPPL.jl/pull/1132),
    [#1252](https://github.com/TuringLang/DynamicPPL.jl/issues/1252),
    [#1311](https://github.com/TuringLang/DynamicPPL.jl/pull/1311),
    [#1376](https://github.com/TuringLang/DynamicPPL.jl/issues/1376).
  - Named parameter storage: [#1150](https://github.com/TuringLang/DynamicPPL.jl/pull/1150),
    [#1183](https://github.com/TuringLang/DynamicPPL.jl/pull/1183),
    [#1204](https://github.com/TuringLang/DynamicPPL.jl/pull/1204),
    [#1238](https://github.com/TuringLang/DynamicPPL.jl/pull/1238),
    AbstractPPL [#117](https://github.com/TuringLang/AbstractPPL.jl/issues/117),
    [#122](https://github.com/TuringLang/AbstractPPL.jl/issues/122),
    [#136](https://github.com/TuringLang/AbstractPPL.jl/issues/136),
    [#150](https://github.com/TuringLang/AbstractPPL.jl/pull/150).
  - Tilde syntax and contexts: [#519](https://github.com/TuringLang/DynamicPPL.jl/issues/519),
    [#804](https://github.com/TuringLang/DynamicPPL.jl/pull/804),
    [#892](https://github.com/TuringLang/DynamicPPL.jl/pull/892),
    [#1221](https://github.com/TuringLang/DynamicPPL.jl/issues/1221).
  - Transforms, log densities, and AD:
    [#575](https://github.com/TuringLang/DynamicPPL.jl/pull/575),
    [#1303](https://github.com/TuringLang/DynamicPPL.jl/pull/1303),
    [#1348](https://github.com/TuringLang/DynamicPPL.jl/pull/1348),
    [#1354](https://github.com/TuringLang/DynamicPPL.jl/pull/1354),
    AbstractPPL [#155](https://github.com/TuringLang/AbstractPPL.jl/pull/155),
    [#157](https://github.com/TuringLang/AbstractPPL.jl/pull/157).
  - Julia engineering and CI: [#50](https://github.com/TuringLang/DynamicPPL.jl/pull/50),
    [#147](https://github.com/TuringLang/DynamicPPL.jl/pull/147),
    [#242](https://github.com/TuringLang/DynamicPPL.jl/pull/242),
    [#733](https://github.com/TuringLang/DynamicPPL.jl/pull/733),
    [#777](https://github.com/TuringLang/DynamicPPL.jl/issues/777),
    AbstractPPL [#25](https://github.com/TuringLang/AbstractPPL.jl/pull/25),
    [#44](https://github.com/TuringLang/AbstractPPL.jl/pull/44),
    [#120](https://github.com/TuringLang/AbstractPPL.jl/issues/120).
  - Accumulators and threading: [#429](https://github.com/TuringLang/DynamicPPL.jl/issues/429),
    [#885](https://github.com/TuringLang/DynamicPPL.jl/pull/885),
    [#925](https://github.com/TuringLang/DynamicPPL.jl/pull/925),
    [#1137](https://github.com/TuringLang/DynamicPPL.jl/pull/1137),
    [#1340](https://github.com/TuringLang/DynamicPPL.jl/pull/1340).

## Before Opening a PR

  - Identify whether the change is user-facing, internal, or downstream-facing.
  - Add the smallest tests that exercise the behaviour.
  - Add nested-submodel tests for context, prefix, conditioning, or fixing
    changes.
  - Run relevant AD backend tests for log-density or transform changes.
  - Check type stability and allocations for hot paths.
  - Check dependency placement and compat bounds when touching Project files,
    extensions, docs, or tests.
  - Include performance numbers for performance-sensitive changes.
  - Document and test new user-facing API.
