# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

DynamicPPL.jl is the core probabilistic programming language and backend for
the [Turing.jl](https://github.com/TuringLang/Turing.jl) ecosystem. It provides
the `@model` macro for defining probabilistic models with tilde (`~`)
statements, and infrastructure for evaluating, conditioning, fixing,
transforming, and inspecting those models.

DynamicPPL builds on AbstractPPL.jl for shared PPL interfaces such as
`VarName`, contexts, conditioning/fixing, and evaluator protocols. For
contributor-facing context extracted from DynamicPPL and AbstractPPL project
history, see `docs/src/onboarding.md`.

## Test Structure

Tests are split into Group1 and Group2 for CI parallelism, controlled by the
`GROUP` environment variable in `test/runtests.jl`. CI also runs Aqua.jl
quality checks and doctests.

**Important**: Each test file should be self-contained. Dependencies should
come from package imports, not relative imports or `include()` statements. This
allows individual test files to be run with tools such as
[TestPicker.jl](https://github.com/theogf/TestPicker.jl).

## Formatting

Code formatting uses
[JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) v1, not v2,
with the **Blue style** configured in `.JuliaFormatter.toml`. CI enforces
formatting on all PRs.

```bash
julia --project -e 'using JuliaFormatter; format(".")'
```

## Architecture

For how things work, see the
[docs](https://turinglang.org/DynamicPPL.jl/stable/): model evaluation, the
tilde pipeline, init strategies, transform strategies, accumulators,
conditioning/fixing, threading, `VarNamedTuple`, and `LogDensityFunction`.

### Key Types

  - **`Model`** (`src/model.jl`): wraps a model function with its arguments and
    context. Created by the `@model` macro in `src/compiler.jl`.
  - **`AbstractVarInfo`** (`src/abstract_varinfo.jl`): interface for tracking
    random variables and accumulated quantities during model execution.
  - **`VarName`** (from AbstractPPL): address for model variables, including
    nested fields and indices.
  - **`VarNamedTuple`** (`src/varnamedtuple.jl`): a named-tuple-like structure
    keyed by `VarName`s. Used as the primary representation for named parameter
    values where supported.
  - **`LogDensityFunction`** (`src/logdensityfunction.jl`): translation layer
    between named model parameters and flat `AbstractVector{<:Real}` inputs for
    optimisers, samplers, and AD. Implements the `LogDensityProblems.jl`
    interface.

### Extensions (`ext/`)

Optional AD backends and integrations, loaded via Julia's package extension
system:

  - `DynamicPPLForwardDiffExt`: ForwardDiff AD
  - `DynamicPPLMooncakeExt`: Mooncake AD, with precompilation workload
  - `DynamicPPLReverseDiffExt`: ReverseDiff AD
  - `DynamicPPLEnzymeCoreExt`: EnzymeCore AD support
  - `DynamicPPLMCMCChainsExt`: MCMCChains integration
  - `DynamicPPLMarginalLogDensitiesExt`: marginalization support

### Testing Utilities (`src/test_utils/`)

`DynamicPPL.TestUtils` provides test models with known analytical solutions
(`logprior_true`, `loglikelihood_true`, etc.) and an AD testing framework
(`run_ad`, `ADResult`) used across the Turing ecosystem.

## Review Guidelines

Common pitfalls and non-obvious constraints when writing or reviewing
DynamicPPL code.

### Prefer `OnlyAccsVarInfo` over `VarInfo`

For new evaluation code, prefer `OnlyAccsVarInfo` plus `init!!` over adding
more behaviour to `VarInfo`. `VarInfo` is still important, but it combines
vector values, transform state, metadata, and accumulators, while many fast
paths need only a subset of that state.

A common migration shape is:

```julia
evaluate!!(model, vi)
```

to:

```julia
init!!(model, oavi, InitFromParams(vi.values), vi.transform_strategy)
```

Choose the actual init strategy and accumulator set from the caller's needs.

### BangBang (`!!`) Return Values

Functions suffixed with `!!` follow BangBang.jl semantics: they may mutate in
place, but they may also return a replacement object. **Always use the return
value.** Discarding the return value can silently drop updates, especially for
immutable wrappers such as `VarInfo` and `AccumulatorTuple`.

```julia
# WRONG: mutation may not happen; vi may be unchanged.
accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)

# RIGHT
vi = accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
```

This applies transitively: if your function calls a `!!` function, it usually
must also return the updated state.

### Accumulator Pitfalls

See the accumulator docs for the full protocol. Common mistakes:

  - **`val` vs `tval` in `accumulate_assume!!`**: `val` is the original
    model-space value and is what `logpdf` should see. `tval` is the
    `TransformedValue`, which may hold linked values. `logjac` is the
    log-Jacobian of the forward link transform, or zero if unlinked.
  - **`logpdf` vs `loglikelihood` for observations**:
    `LogLikelihoodAccumulator` uses `Distributions.loglikelihood`, not `logpdf`.
    For array-valued or product-like observations, the two can have different
    shapes or aggregation semantics. Use the one the accumulator protocol
    expects.
  - **Aliased `copy`**: `copy(acc)` must not share mutable internal state unless
    that sharing is intentional and documented. Aliased containers can corrupt
    results when accumulators are copied for threaded evaluation.

### TransformedValue

  - **`get_raw_value(tv)` needs a distribution for dynamic transforms.**
    `DynamicLink` and `Unlink` derive their transform from the distribution, so
    use `get_raw_value(tv, dist)`. The one-argument form is for cases where the
    transform is already fully known.
  - **`DynamicLink` re-derives the bijection from `dist` during evaluation.**
    This is necessary because support can depend on earlier random variables, for
    example `y ~ truncated(Normal(); lower=x)`. If support is known to be
    constant, consider `FixedTransform` via `WithTransforms`.
  - **`FixedTransform` must match the target transform.** Do not assume fixed
    transforms compose with re-derived dynamic transforms.

### LogDensityFunction

  - **`getlogjoint_internal` vs `getlogjoint`**: samplers operating in
    unconstrained space usually need `getlogjoint_internal`. `getlogjoint` is the
    constrained-space log joint. Using the wrong one changes Jacobian handling.
  - **Compiled ReverseDiff tapes are input-dependent.** If model control flow
    depends on parameter values, compiled ReverseDiff only gives correct
    gradients for inputs that follow the same branch as the compilation input.
    Do not use `AutoReverseDiff(; compile=true)` for parameter-dependent
    branching.
  - Keep evaluator APIs split into structural preparation and AD-specific
    preparation. Put backend-specific gradient code in extensions where possible.
  - Check aliasing in evaluator and AD APIs. `!!` methods may return buffers that
    alias internal caches; copy before exposing or storing results long term.

### `VarNamedTuple` as Primary Data Structure

`VarNamedTuple` is the canonical internal representation for named parameter
collections in new DynamicPPL code: conditioning/fixing values, parameter
storage, and accumulator values. `NamedTuple` and `Dict{VarName}` are accepted
as user-facing input, but should usually be converted to `VarNamedTuple` at API
boundaries rather than propagated internally.

Preserve templates, shapes, and index structure when working with
`VarNamedTuple`. Array-valued variables, slices, and nested fields need enough
template information to round-trip between named values and flat vectors. Avoid
large mostly-empty shadow arrays for sparse indexed variables, and keep eltypes
concrete in hot paths.

See the VarNamedTuple docs for motivation: it is performant, general, and gives
one representation for named parameter collections.

### VarName

  - **Use `@varname(x)`, not `:x` or `VarName(:x)`.** The macro constructs the
    correct optic for indexed access. `@varname(x[1])` creates a `VarName` with
    an index lens; constructing this manually is error-prone.
  - **Use subsumption for containment checks.** `subsumes(@varname(x),
    @varname(x[1]))` is `true`, but the two names are not equal. Conditioning
    on `@varname(x)` matches sub-indices; conditioning on `@varname(x[1])`
    matches only that index.
  - Treat `VarName` display, sorting, prefixing, unprefixing, and serialization
    as downstream-facing interface behaviour. Chains, saved results, and
    external packages can depend on stable names.
  - Test nested fields, indices, ranges, `Colon`, and non-standard indices when
    changing `VarName` optics.

### `@model` Compiler Changes

`@model` lowering must preserve ordinary Julia semantics, not only
probabilistic statements. For compiler changes, test positional and keyword
arguments, default values, splatting, closures, interpolation, return values,
no-observation models, and data- or parameter-dependent control flow.

Keep macro hygiene explicit. User variables, generated temporaries, and globals
should not capture each other accidentally. Inspect expanded code when changing
compiler paths. Preserve model return values; returned quantities are
user-visible and distinct from accumulated random variables.

### Threading

See the threading docs. Key edge case:
`promote_for_threadsafe_eval(acc, T)` must be implemented if an accumulator
stores typed containers that need to hold AD tracer types, such as ForwardDiff
`Dual`s. The default is a no-op, which is wrong for accumulators with concrete
float fields.

Avoid designs that index storage by `Threads.threadid()`. Julia scheduling and
thread IDs are not a stable ownership model.

## Julia Engineering Practices

  - Check type stability with `@inferred`, `@code_warntype`, and focused tests
    when changing compiler output, VNTs, accumulators, transforms, or log-density
    paths.
  - Avoid unnecessary static parameters. Julia specializes on most ordinary
    argument types, but is conservative for `Type`, `Function`, and `Vararg`.
    Use `f(x, ::Type{T}) where {T}` when the type itself must specialize.
  - Benchmark generated functions, macro output, and hot-path refactors before
    assuming simpler code is equivalent.
  - Prefer dispatch and small protocol functions over large conditional blocks.
  - Avoid broad overloads of Base functions for arbitrary input types; they can
    create method ambiguities and accidental API.
  - Put backend-specific behaviour in package extensions or narrow integration
    layers when possible.
  - Make direct dependencies explicit enough to version-bound and test. Do not
    rely on packages being loaded transitively.
  - Use accessor functions for values downstream packages need. Direct field
    access from Turing or other packages turns internal representation into
    accidental API.
  - Prefer `Base.maybeview` over eager slicing when indexed access should avoid
    allocations but still support tuples and scalar indexing.
  - Avoid fragile output-type prediction. When possible, compute an initial value
    and allocate caches from the observed value.
  - Keep doctests deterministic. Use `StableRNGs` when examples print random
    values.

## Contributing

  - Non-breaking changes target `main`; breaking changes target the `breaking`
    branch.
  - CI runs tests on Ubuntu, Windows, and macOS, across stable, minimum, and
    selected Julia versions, with both one and two threads.
  - Identify whether the change is user-facing, internal, or downstream-facing
    through Turing.jl.
  - Add the smallest tests that exercise the behaviour.
  - Add nested-submodel tests for context, prefix, conditioning, or fixing
    changes.
  - Add AD backend tests for log-density, transform, vector-parameter, or
    `run_ad` changes.
  - Add round-trip tests for flattening and unflattening changes, including
    scalars, arrays, tuples, `NamedTuple`s, nested values, and mixed element
    types.
  - Check type stability and allocations for hot paths.
  - Check dependency placement and compat bounds when touching Project files,
    extensions, docs, or tests.
  - Include benchmark numbers for performance-sensitive changes.
  - Document and test new user-facing API.
