# AGENTS.md

Repository guidance for coding agents. See @JULIA.md for general Julia practices and `docs/src/onboarding.md` for newcomer-oriented background.

## Project Overview

DynamicPPL.jl is the core probabilistic programming language backend for the Turing.jl ecosystem. It provides the `@model` macro for tilde (`~`) statements and infrastructure for evaluating, conditioning, fixing, transforming, and inspecting probabilistic models.

DynamicPPL builds on AbstractPPL.jl for shared PPL interfaces such as `VarName`, contexts, conditioning/fixing, and evaluator protocols.

## Tests And Formatting

  - Tests are split into Group1/Group2 via `GROUP` in `test/runtests.jl`.

  - CI also runs Aqua.jl quality checks and doctests.
  - Test files are self-contained: use package imports, not relative imports or `include()`, so they run individually with TestPicker.jl.
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

## DynamicPPL Invariants

Evaluator methods follow BangBang `!!` semantics (see JULIA.md). `VarInfo` and `AccumulatorTuple` are immutable, so discarding a `!!` return value is a silent bug.

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
  - Use `@varname(x)`, not `:x` or `VarName(:x)`. Use subsumption for containment checks, e.g. `subsumes(@varname(x), @varname(x[1]))`. Conditioning on `@varname(x)` covers subindices; conditioning on `@varname(x[1])` only matches that index.

## `@model` Compiler

`@model` lowering must preserve ordinary Julia semantics, not only probabilistic statements.

For compiler changes, test positional and keyword arguments, default values, splatting, closures, interpolation, return values, no-observation models, and data- or parameter-dependent control flow.

Keep macro hygiene explicit. User variables, generated temporaries, and globals should not capture each other accidentally. Inspect expanded code when changing compiler paths. Preserve model return values; they are user-visible and distinct from accumulated random variables.

## Threading

Implement `promote_for_threadsafe_eval(acc, T)` for accumulators with concrete float fields; the default no-op leaves them unable to hold AD tracers like ForwardDiff `Dual`s. General threading guidance lives in JULIA.md.

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
