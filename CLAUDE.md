# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynamicPPL.jl is the core probabilistic programming language and backend for the [Turing.jl](https://github.com/TuringLang/Turing.jl) ecosystem. It provides the `@model` macro for defining probabilistic models with tilde (`~`) statements, and infrastructure for evaluating, conditioning, and transforming those models.

## Test Structure

Tests are split into Group1 and Group2 for CI parallelism (controlled by the `GROUP` env var in `test/runtests.jl`). CI also runs Aqua.jl quality checks and doctests.

**Important**: Each test file should be self-contained. All dependencies must come from package imports, not relative imports or `include()` statements. This enables running individual test files via [TestPicker.jl](https://github.com/theogf/TestPicker.jl).

## Formatting

Code formatting uses [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) v1 (not v2) with the **Blue style** (configured in `.JuliaFormatter.toml`). CI enforces formatting on all PRs.

```bash
julia --project -e 'using JuliaFormatter; format(".")'
```

## Architecture

For how things work, see the [docs](https://turinglang.org/DynamicPPL.jl/stable/): [model evaluation](https://turinglang.org/DynamicPPL.jl/stable/evaluation/), [tilde pipeline](https://turinglang.org/DynamicPPL.jl/stable/tilde/), [init strategies](https://turinglang.org/DynamicPPL.jl/stable/init/), [transform strategies](https://turinglang.org/DynamicPPL.jl/stable/transforms/), [accumulators](https://turinglang.org/DynamicPPL.jl/stable/accs/overview/), [conditioning/fixing](https://turinglang.org/DynamicPPL.jl/stable/conditionfix/), [threading](https://turinglang.org/DynamicPPL.jl/stable/accs/threadsafe/).

### Key Types

  - **`Model`** (`src/model.jl`): Wraps a model function with its arguments and context. Created by the `@model` macro (`src/compiler.jl`).
  - **`AbstractVarInfo`** (`src/abstract_varinfo.jl`): Interface for tracking random variables and accumulated quantities during model execution.
  - **`VarNamedTuple`** (`src/varnamedtuple.jl`): A named-tuple-like structure keyed by `VarName`s (from AbstractPPL). Used as the primary representation for parameter values.
  - **`LogDensityFunction`** (`src/logdensityfunction.jl`): Translation layer between named model parameters and flat `AbstractVector{<:Real}` for optimisers/samplers. Implements the `LogDensityProblems.jl` interface.

### Extensions (`ext/`)

Optional AD backends and integrations, loaded via Julia's package extension system:

  - `DynamicPPLForwardDiffExt` — ForwardDiff AD
  - `DynamicPPLMooncakeExt` — Mooncake AD (with precompilation workload)
  - `DynamicPPLReverseDiffExt` — ReverseDiff AD
  - `DynamicPPLEnzymeCoreExt` — Enzyme AD
  - `DynamicPPLMCMCChainsExt` — MCMCChains integration
  - `DynamicPPLMarginalLogDensitiesExt` — marginalization support

### Testing Utilities (`src/test_utils/`)

`DynamicPPL.TestUtils` provides test models with known analytical solutions (`logprior_true`, `loglikelihood_true`, etc.) and an AD testing framework (`run_ad`, `ADResult`) used across the Turing ecosystem.

## Review Guidelines

Common pitfalls and non-obvious constraints when writing or reviewing DynamicPPL code.

### Prefer `OnlyAccsVarInfo` over `VarInfo`

New code should use `OnlyAccsVarInfo` (OAVI) + `init!!`, not `VarInfo` + `evaluate!!`. VarInfo is being phased out ([#1376](https://github.com/TuringLang/DynamicPPL.jl/issues/1376)) — it carries redundant state (`vi.values` duplicates `VectorValueAccumulator`) and is slower. Don't add new features to VarInfo. The migration path: `evaluate!!(model, vi)` becomes `init!!(model, oavi, InitFromParams(vi.values), vi.transform_strategy)`.

### BangBang (`!!`) Return Values

Functions suffixed with `!!` (from BangBang.jl) attempt in-place mutation but may return a new object instead. **Always use the return value.** `VarInfo` and `AccumulatorTuple` are immutable structs, so `!!` functions unconditionally return new objects — discarding the return value is a silent bug with no warning.

```julia
# WRONG: mutation didn't happen, vi is unchanged
accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)

# RIGHT
vi = accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
```

This applies transitively: if your function calls a `!!` function, it must also return the updated state.

### Accumulator Pitfalls

See [accumulator docs](https://turinglang.org/DynamicPPL.jl/stable/accs/overview/) for the full protocol. Common mistakes:

  - **`val` vs `tval` in `accumulate_assume!!`**: `val` is always in the original unlinked space (use it for `logpdf`). `tval` is the `TransformedValue` which may hold linked values. `logjac` is the log-Jacobian of the **forward** link transform (zero if unlinked). Confusing these is a common source of wrong log-densities.
  - **`logpdf` vs `loglikelihood` for observations**: `LogLikelihoodAccumulator` uses `Distributions.loglikelihood`, not `logpdf`. For vector observations, `logpdf` returns a vector while `loglikelihood` returns a scalar sum. Using `logpdf` where `loglikelihood` is expected silently produces wrong types. See [JuliaStats/Distributions.jl#1972](https://github.com/JuliaStats/Distributions.jl/issues/1972).
  - **Aliased `copy`**: `copy(acc)` must deep-copy all mutable internal state. Aliased containers (e.g. shared `Vector` fields) corrupt results when accumulators are copied for `ThreadSafeVarInfo`.

### TransformedValue

  - **`get_raw_value(tv)` errors for `DynamicLink` and `Unlink`.** These transforms are derived from the distribution, so you must use `get_raw_value(tv, dist)`. The one-argument form only works for `NoTransform` and `FixedTransform`.
  - **`DynamicLink` re-derives the bijection from `dist` every evaluation.** This is necessary because the support of a variable can depend on other variables (e.g. `y ~ truncated(Normal(); lower=x)`), so the transform cannot be cached. When the support is known to be constant, [`FixedTransform` via `WithTransforms`](https://turinglang.org/DynamicPPL.jl/stable/fixed_transforms/) is an option.
  - **`FixedTransform` must exactly match the target.** `apply_transform_strategy` errors if a `FixedTransform` doesn't match the expected `target_transform`. Fixed transforms don't compose with re-derived transforms.

### LogDensityFunction

  - **`getlogjoint_internal` vs `getlogjoint`**: `getlogjoint_internal(vi) = getlogjoint(vi) - getlogjac(vi)`. Samplers operating in unconstrained space need `getlogjoint_internal` (the default). `getlogjoint` gives the density in constrained space without the Jacobian correction — using it for HMC/NUTS is wrong.
  - **Compiled ReverseDiff tapes are input-dependent.** If your model has control flow that depends on parameter values (e.g. `if x > 0`), compiled ReverseDiff will only give correct gradients for inputs that trigger the same branch as the compilation input. Don't use `AutoReverseDiff(; compile=true)` with parameter-dependent branching.

### `VarNamedTuple` as Primary Data Structure

`VarNamedTuple` is the canonical representation for named parameter collections throughout DynamicPPL. New code should use it everywhere — for conditioning, fixing, parameter storage, and accumulator values. `NamedTuple` and `Dict{VarName}` are accepted as user-facing input but only insofar as they are converted to `VarNamedTuple` at the boundary. Don't propagate them through internal code.

See the [VarNamedTuple docs](https://turinglang.org/DynamicPPL.jl/stable/vnt/motivation/) for motivation — it is performant, general, and provides a single source of truth for named parameter collections.

### VarName

  - **Use `@varname(x)`, not `:x` or `VarName(:x)`.** The macro constructs the correct optic for indexed access. `@varname(x[1])` creates a VarName with an index lens — constructing this manually is error-prone.
  - **Subsumption, not equality, for containment checks.** `subsumes(@varname(x), @varname(x[1]))` is `true`, but they are not `==`. Conditioning on `@varname(x)` matches all sub-indices; conditioning on `@varname(x[1])` only matches that index. Use `subsumes` when checking if a VarName is "covered by" another.

### Threading

See [threading docs](https://turinglang.org/DynamicPPL.jl/stable/accs/threadsafe/). Key edge case: `promote_for_threadsafe_eval(acc, T)` must be implemented if your accumulator stores typed containers that need to hold AD tracer types (e.g. ForwardDiff `Dual`s). The default is a no-op, which is wrong for accumulators with concrete float fields.

## Contributing

  - Non-breaking changes target `main`; breaking changes target the `breaking` branch.
  - CI runs tests on Ubuntu/Windows/macOS, Julia stable/min/1.11, with 1 and 2 threads.
  - Julia ≥ 1.10.8 required (see `[compat]` in `Project.toml`).
