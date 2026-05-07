# AGENTS.md

Guidance for Claude Code when working in this repository.

See also: @JULIA.md for general Julia engineering and review guidance.

## Project Overview

DynamicPPL.jl is the core probabilistic programming language backend for the Turing.jl ecosystem. It provides the `@model` macro for tilde (`~`) statements and infrastructure for evaluating, conditioning, fixing, transforming, and inspecting probabilistic models.

DynamicPPL builds on AbstractPPL.jl for shared PPL interfaces such as `VarName`, contexts, conditioning/fixing, and evaluator protocols. For project history and contributor context, see `docs/src/onboarding.md`.

## Tests And Formatting

- Tests are split into Group1 and Group2 for CI parallelism, controlled by `GROUP` in `test/runtests.jl`.
- CI also runs Aqua.jl quality checks, doctests, formatting, and multi-platform tests across selected Julia versions and thread counts.
- Each test file should be self-contained. Use package imports, not relative imports or `include()` statements, so files can run individually with tools such as TestPicker.jl.
- Formatting uses JuliaFormatter.jl v1, not v2, with Blue style from `.JuliaFormatter.toml`.

```bash
julia --project -e 'using JuliaFormatter; format(".")'
```

## Architecture Pointers

Use the docs for model evaluation, the tilde pipeline, init strategies, transform strategies, accumulators, conditioning/fixing, threading, `VarNamedTuple`, and `LogDensityFunction`.

- `Model` (`src/model.jl`): wraps a model function, arguments, and context; created by `@model` in `src/compiler.jl`.
- `AbstractVarInfo` (`src/abstract_varinfo.jl`): interface for tracking random variables and accumulated quantities during model execution.
- `VarName` (AbstractPPL): address for model variables, including nested fields and indices.
- `VarNamedTuple` (`src/varnamedtuple.jl`): named-tuple-like parameter storage keyed by `VarName`.
- `LogDensityFunction` (`src/logdensityfunction.jl`): bridge between named model parameters and flat `AbstractVector{<:Real}` inputs for samplers, optimizers, and AD.
- Optional integrations live in `ext/`: ForwardDiff, Mooncake, ReverseDiff, EnzymeCore, MCMCChains, and MarginalLogDensities.
- `DynamicPPL.TestUtils` provides analytical test models and AD helpers (`run_ad`, `ADResult`) used across the Turing ecosystem.

## DynamicPPL Review Notes

- Prefer `OnlyAccsVarInfo` plus `init!!` for new evaluation code when fast paths only need accumulators or a subset of VarInfo state.
- `VarInfo` remains important, but it combines vector values, transform state, metadata, and accumulators. Many fast paths need only part of that state, so avoid adding behavior to `VarInfo` by default.
- Functions ending in `!!` follow BangBang.jl semantics: they may mutate or return a replacement object. Always use the return value, and return updated state from callers that invoke `!!`.

```julia
# Wrong: updates may be lost.
accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)

# Right.
vi = accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
```

- In `accumulate_assume!!`, `val` is the model-space value and should be passed to `logpdf`; `tval` is the transformed value.
- `logjac` is the log-Jacobian of the forward link transform, or zero if unlinked.
- `LogLikelihoodAccumulator` uses `Distributions.loglikelihood`, not `logpdf`; array-valued or product-like observations can differ in shape or aggregation.
- `copy(acc)` must not share mutable internal state unless that sharing is intentional and documented.
- `get_raw_value(tv, dist)` is required for dynamic transforms because `DynamicLink` and `Unlink` derive their transform from the distribution; the raw value cannot be recovered correctly from `tv` alone when support is distribution-dependent.
- The one-argument `get_raw_value(tv)` is only for cases where the transform is already fully known.
- `DynamicLink` re-derives the bijection from `dist` during evaluation because support can depend on earlier random variables, such as `y ~ truncated(Normal(); lower=x)`. Do not cache or reuse a fixed bijection unless support is known to be constant.
- If support is known to be constant, consider `FixedTransform` via `WithTransforms`; fixed transforms must match the target transform.
- Samplers operating in unconstrained space usually need `getlogjoint_internal`; `getlogjoint` is the constrained-space log joint.
- Compiled ReverseDiff tapes are input-dependent; do not use `AutoReverseDiff(; compile=true)` when model control flow depends on parameter values.
- Keep evaluator APIs split into structural preparation and AD-specific preparation. Put backend-specific gradient code in extensions when possible.
- Check aliasing in evaluator and AD APIs. `!!` methods may return buffers that alias internal caches; copy before exposing results to callers, storing them long term, or reusing them after another model evaluation.

## Names And Parameter Storage

- Use `VarNamedTuple` as the canonical internal representation for named parameter collections in new code.
- Accept `NamedTuple` and `Dict{VarName}` at user-facing boundaries, but convert to `VarNamedTuple` rather than propagating them internally.
- Preserve templates, shapes, and index structure when round-tripping between named values and flat vectors.
- Avoid large mostly-empty shadow arrays for sparse indexed variables.
- Use `@varname(x)`, not `:x` or `VarName(:x)`.
- Use subsumption for containment checks: `subsumes(@varname(x), @varname(x[1]))` is true, but the names are not equal.
- Treat VarName display, sorting, prefixing, unprefixing, and serialization as downstream-facing interface behavior.
- Test nested fields, indices, ranges, `Colon`, and non-standard indices when changing VarName optics.

## `@model` Compiler

`@model` lowering must preserve ordinary Julia semantics, not only probabilistic statements.

For compiler changes, test positional and keyword arguments, default values, splatting, closures, interpolation, return values, no-observation models, and data- or parameter-dependent control flow.

Keep macro hygiene explicit. User variables, generated temporaries, and globals should not capture each other accidentally. Inspect expanded code when changing compiler paths. Preserve model return values; they are user-visible and distinct from accumulated random variables.

## Threading

- Implement `promote_for_threadsafe_eval(acc, T)` if an accumulator stores typed containers that need to hold AD tracer types such as ForwardDiff `Dual`s. The default no-op is wrong for accumulators with concrete float fields.
- Avoid designs that index storage by `Threads.threadid()`. Julia scheduling and thread IDs are not a stable ownership model.

## Contributing Checklist

- Non-breaking changes target `main`; breaking changes target `breaking`.
- Identify whether the change is user-facing, internal, or downstream-facing through Turing.jl.
- Add the smallest tests that exercise the behavior.
- Add nested-submodel tests for context, prefix, conditioning, or fixing changes.
- Add AD backend tests for log-density, transform, vector-parameter, or `run_ad` changes.
- Add round-trip tests for flattening and unflattening changes, including scalars, arrays, tuples, `NamedTuple`s, nested values, and mixed element types.
- Check type stability and allocations for hot paths.
- Check dependency placement and compat bounds when touching Project files, extensions, docs, or tests.
- Include benchmark numbers for performance-sensitive changes.
- Document and test new user-facing API.
