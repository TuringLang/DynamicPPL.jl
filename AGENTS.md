# AGENTS.md

Repository guidance for coding agents. See `docs/src/onboarding.md` for newcomer-oriented background.

## Project Overview

DynamicPPL.jl is the core probabilistic programming language backend for the Turing.jl ecosystem. It provides the `@model` macro for tilde (`~`) statements and infrastructure for evaluating, conditioning, fixing, transforming, and inspecting probabilistic models.

DynamicPPL builds on AbstractPPL.jl for shared PPL interfaces such as `VarName`, contexts, conditioning/fixing, and evaluator protocols.

## Tests And Formatting

  - Tests are split into Group1/Group2 via `GROUP` in `test/runtests.jl`.

  - CI also runs Aqua.jl quality checks and doctests.
  - Test files are self-contained: use package imports, not relative imports or `include()`, so they run individually with TestPicker.jl.
  - Always refresh each environment (`Pkg.update()` / `up`) before tests or doc builds — a stale manifest can cause subtle resolution and loading issues.
  - Formatting is JuliaFormatter v1 (Blue style), enforced by CI:
    
    ```bash
    julia --project -e 'using JuliaFormatter; format(".")'
    ```

## Architecture Pointers

  - Docs: model evaluation, tilde pipeline, init strategies, transform strategies, accumulators, conditioning/fixing, and thread-safe accumulation.
  - `Model` (`src/model.jl`): wraps model function, args, context; created by `@model` in `src/compiler.jl`.
  - `AbstractVarInfo` (`src/abstract_varinfo.jl`): tracks random variables and accumulated quantities during evaluation.
  - `VarName` (AbstractPPL): address for model variables, including nested fields/indices.
  - `VarNamedTuple` (`src/varnamedtuple.jl`): named-tuple-like parameter storage keyed by `VarName`.
  - `LogDensityFunction` (`src/logdensityfunction.jl`): bridge from named parameters to flat `AbstractVector{<:Real}` for samplers, optimisers, and AD via LogDensityProblems.jl.
  - `ext/`: `DynamicPPLForwardDiffExt`, `DynamicPPLMooncakeExt`, `DynamicPPLReverseDiffExt`, `DynamicPPLEnzymeCoreExt`, `DynamicPPLComponentArraysExt`, `DynamicPPLMCMCChainsExt`, and `DynamicPPLMarginalLogDensitiesExt`.
  - `DynamicPPL.TestUtils`: analytical test models (`logprior_true`, `loglikelihood_true`, etc.), `run_ad`, `ADResult`.

## Julia-specific guidance

Engineering:

  - Constrain arguments to `Float64`, `Int`, `Real`, `Array`, `Vector`, or `Matrix` only when required by the mathematics or an external API.
  - Preserve caller types with `zero`, `one`, `oftype`, `promote`, and `promote_type`. Support `Float32`, `BigFloat`, AD numbers, units, and GPU scalars where applicable.
  - Keep storage concrete with type parameters; avoid fields typed as `Number` or `AbstractVector`. Each type parameter should serve dispatch, storage, or an invariant.
  - When specialization on a `Type`, `Function`, or `Vararg` argument is needed, use an explicit type parameter such as `f(x, ::Type{T}) where {T}`.
  - Derive output containers from inputs with `similar`, or accept a destination buffer. Use `Base.maybeview` to avoid eager slices while supporting scalar and tuple indices.
  - Prefer small, dispatch-based protocols to large conditionals. Isolate backend behaviour in package extensions or narrow integration layers.
  - Check inference with `@inferred` or `@code_warntype` for generated code, custom containers, accumulators, transforms, and log-density paths. Benchmark generated functions, macro output, and hot paths.
  - Use `StableRNGs` when doctests print random values.

Public APIs:

  - Put data first. Put a callable first only to support `do`-block syntax.
  - Follow Julia keyword conventions: `dims=` for dimensions, `init=` for reductions, and `lt=`, `by=`, and `rev=` for sorting. Use a tuple for multiple dimensions where natural.
  - Pair mutating and non-mutating forms when both are useful. Keep related argument orders, keyword names, and return shapes consistent.
  - Put configuration in keywords, not positional `Bool`, small integer, or `Symbol` flags. Wrappers should forward `kwargs...`.
  - Expose downstream state through accessors, traits, or protocol functions, not direct field access.
  - Extend an appropriate `Base` method rather than adding a parallel name. Avoid broad overloads, which create ambiguities and accidental API.
  - Keep `==`, `isequal`, and `hash` consistent.
  - Give each operation one documented failure mode: an exception, `nothing`, or a sentinel.
  - Treat exported names, constructor forms, keyword arguments, aliases, abstract supertypes, and traits as API commitments. Positional and keyword constructors are separate commitments. Mark internal names already used downstream as `public`.

Probability code:

  - Distinguish sample type, mathematical support, and reference measure. Distributions with floating-point samples can have atoms, and `pdf` may denote mass or density. Censoring can mix atoms and density; truncation changes support and normalization.
  - Reject invalid distribution parameters and handle domain boundaries explicitly.
  - Pass an RNG explicitly; never rely on the global RNG.

Idioms:

  - Always reassign `!!` results; these methods may mutate or replace their input.
  - Copy `!!` results before retaining them across calls; they may alias internal state.
  - For types that own mutable evaluation state, `copy` must not share that state unless the sharing is intentional and documented.
  - Do not index task-owned storage by `Threads.threadid()` because tasks can migrate. Pass per-task buffers or use a thread-safe collection.

Testing:

  - Test generic numeric APIs with `Float32`, `BigFloat`, and a relevant AD number type; include units and GPU scalars when supported.
  - Test generic array APIs with a static array or another non-`Array` input.
  - For distribution-aware code, test boundaries, invalid parameters, finite densities where expected, and `logcdf(d, x) <= 0`.
  - Test reproducible sampling with a stable RNG and gradients with respect to both observations and distribution parameters.

## DynamicPPL Invariants

Evaluator methods follow BangBang `!!` semantics. `VarInfo` and `AccumulatorTuple` are immutable, so discarding a `!!` return value is a silent bug.

**`accumulate_assume!!`** — `val` is model-space (passed to `logpdf`); `tval` is transformed; `logjac` is the log-Jacobian of the forward link transform (zero if unlinked):

```julia
vi = accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
```

**`LogLikelihoodAccumulator`** uses `Distributions.loglikelihood`, not `logpdf` — array/product observations differ in shape and aggregation.

**Dynamic transforms** — `DynamicLink`/`Unlink` re-derive bijections from `dist` because support can depend on earlier RVs (e.g. `y ~ truncated(Normal(); lower=x)`). Use `get_raw_value(tv, dist)`; the one-argument form only works for `NoTransform` and `FixedTransform`. Never cache a fixed bijection. Use `FixedTransform`/`WithTransforms` only when support is constant, and make sure the fixed transform exactly matches the target.

**Log joint** — `getlogjoint_internal(vi) = getlogjoint(vi) - getlogjac(vi)`. Samplers in unconstrained space want `getlogjoint_internal`; constrained-space is `getlogjoint`.

**ReverseDiff** — don't use `AutoReverseDiff(; compile=true)` when model control flow depends on parameter values (compiled tapes are input-dependent).

## Review Focus

  - Prefer `OnlyAccsVarInfo` + `init!!` for new evaluation code that needs only accumulators or a subset of `VarInfo` state.
  - Avoid adding behaviour to `VarInfo` by default; it bundles values, transform state, metadata, and accumulators, but most fast paths need only part.
  - Keep evaluator APIs split: structural prep vs AD-specific prep. Backend gradient code goes in extensions.
  - Use `VarNamedTuple` as the canonical internal representation for named parameter collections in new code. Convert user-facing `NamedTuple` and `Dict{VarName}` inputs at boundaries.
  - Preserve templates, shapes, and index structure when round-tripping between named values and flat vectors.
  - Ensure `copy(acc)` does not share mutable internal state; aliased accumulator containers corrupt results when copied for `ThreadSafeVarInfo`.
  - Ensure `split(acc)` does not share mutable accumulation state; `combine` must merge that state even when it is stored outside the main value container.
  - Aggregating `ThreadSafeVarInfo` accumulators must not mutate stored state. Seed reductions with an independent copy because `combine` may consume its first argument, and test repeated reads and copies with a mutable accumulator.
  - Use `@varname(x)`, not `:x` or `VarName(:x)`. Use subsumption for containment checks, e.g. `subsumes(@varname(x), @varname(x[1]))`. Conditioning on `@varname(x)` covers subindices; conditioning on `@varname(x[1])` only matches that index.

## `@model` Compiler

`@model` lowering must preserve ordinary Julia semantics, not only probabilistic statements.

For compiler changes, test positional and keyword arguments, default values, splatting, closures, interpolation, return values, no-observation models, and data- or parameter-dependent control flow.

Keep macro hygiene explicit. User variables, generated temporaries, and globals should not capture each other accidentally. Inspect expanded code when changing compiler paths. Preserve model return values; they are user-visible and distinct from accumulated random variables.

## Threading

Implement `promote_for_threadsafe_eval(acc, T)` for accumulators with concrete float fields; the default no-op leaves them unable to hold AD tracers like ForwardDiff `Dual`s.

## Contributing Checklist

  - Non-breaking changes target `main`; breaking changes target `breaking`.
  - Julia `1.10.8` is the minimum supported version in `Project.toml`.
  - CI runs Ubuntu/Windows/macOS, Julia stable/min/1.11, and both one- and two-thread configurations.
  - Identify whether the change is user-facing, internal, or downstream-facing through Turing.jl.
  - Add the smallest tests that exercise the behavior.
  - Add nested-submodel tests for context, prefix, conditioning, or fixing changes.
  - Add AD backend tests for log-density, transform, vector-parameter, or `run_ad` changes.
  - Add round-trip tests for flattening and unflattening changes, including scalars, arrays, tuples, `NamedTuple`s, nested values, and mixed element types.
  - Check type stability and allocations for hot paths.
  - Check dependency placement and compat bounds when touching Project files, extensions, docs, or tests.
  - Include benchmark numbers for performance-sensitive changes.
  - Document and test new user-facing API.
