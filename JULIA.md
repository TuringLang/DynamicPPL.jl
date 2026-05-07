# JULIA.md

Shared day-to-day Julia practices. DynamicPPL-specific review notes live in `AGENTS.md`; newcomer context lives in `docs/src/onboarding.md`.

## Engineering

  - Write generic numeric code unless the math or an external API forces a concrete type. Avoid `Float64`/`Int`/`Real`/`Array`/`Vector`/`Matrix` constraints that aren't load-bearing.
  - Preserve caller types with `zero(x)`, `one(x)`, `oftype`, `promote`, `promote_type` — especially for `Float32`, `BigFloat`, AD numbers, units, and GPU scalars.
  - Struct fields should be concrete via type parameters, not `field::Number` or `field::AbstractVector`.
  - Julia doesn't specialize on `Type`, `Function`, or `Vararg` arguments. Use `f(x, ::Type{T}) where {T}` when the type itself must specialize.
  - Check inference (`@inferred`, `@code_warntype`) when touching generated code, custom containers, accumulators, transforms, or log-density paths.
  - Benchmark generated functions, macro output, and hot-path refactors before assuming a simpler form is equivalent.
  - Prefer dispatch and small protocol functions over large conditional blocks.
  - Avoid broad Base overloads — they create method ambiguities and accidental API.
  - Backend-specific behaviour goes in package extensions or narrow integration layers.
  - Provide accessors for values downstream packages need — direct field access from another package becomes accidental API.
  - Prefer `Base.maybeview` over eager slicing when allocation matters but tuple/scalar indexing must still work.
  - Allocate output containers from observed values rather than predicting element types up front.
  - Doctests must be deterministic — use `StableRNGs` when examples print random values.

```julia
# Avoid: too concrete, inference-hostile.
f(x::Float64) = x / 2
buf = zeros(Float64, length(xs))
struct Model
    scale::Number
end

# Prefer: generic args, input-derived allocation, concrete fields.
f(x) = x / 2
buf = similar(xs, promote_type(eltype(xs), Float64), length(xs))
struct Model{T}
    scale::T
end
```

## Idioms

  - `!!` semantics (BangBang.jl): methods ending in `!!` may mutate or return a replacement. Always reassign: `x = f!!(x, ...)`.
  - Returns from `!!` methods may alias internal state — copy before holding long-term or reusing across calls.
  - `copy(x)` must not share mutable internal state with `x` unless intentional and documented.
  - Don't index thread-owned storage by `Threads.threadid()` — task scheduling makes IDs unstable. Pass per-task buffers explicitly or use a thread-safe collection.

## Public APIs

Signatures:

  - Dimension arguments use `dims=` (tuple-valued where natural).
  - Data first; callable first when `do`-block syntax should work (`map(f, xs)`-style).
  - Pair mutating and non-mutating versions when both make sense (`sort!`/`sort`).
  - Configuration is keywords, not positional `Bool`/small `Int`/`Symbol` flags.
  - Reductions take `init=`; sorting takes `lt=`/`by=`/`rev=`.
  - Allocate output via `similar(x, ...)` or a destination buffer; don't hardcode `Vector{Float64}`.
  - Wrappers forward `kwargs...`.
  - Match argument order, keyword names, and return shape across related functions.

Types:

  - Provide protocol functions (accessors, traits) so downstream packages can extend without reaching for internals.
  - Type parameters serve dispatch, storage, or invariants — not decoration.
  - Define `hash` whenever you define `==`, consistent with `isequal`.
  - Extend an existing Base method rather than introducing a parallel name (`Base.length`, not `mylength`).
  - Pick one failure mode (throw, `nothing`, sentinel) and document it.

Public constructors, keyword arguments, exported names, aliases, abstract supertypes, and traits are long-term commitments. A public concrete type commits to both `Foo(a, b)` and `Foo(; a, b)`. Mark internal names that downstream code already depends on as `public` rather than leaving them accidental.

## Probability

When writing distribution-aware code (accumulators, transforms, log-density paths):

  - Separate sample type, mathematical support, and reference measure. Floating-point samples can still have atoms; `pdf` may be a density, a mass, or mixed for censored/truncated cases.
  - Check domain boundaries and invalid parameters explicitly.
  - Thread an explicit RNG; never reach for the global RNG implicitly.
  - Consider parameter gradients, not just gradients with respect to observations.

```julia
@test isfinite(logpdf(d, x))
@test logcdf(d, x) <= 0
@test isapprox(f(Float64(x)), Float64(f(big(x))); rtol=1e-12)
@test rand(StableRNG(1), d) == rand(StableRNG(1), d)
```

## Testing Generic Code

Exercise type variety when the contract is "works for any number type":

```julia
@test f(Float32(1)) isa Float32
@test f(big"1.0") isa BigFloat
@test ForwardDiff.derivative(f, 1.0) isa Real
@test f(SVector(1.0, 2.0)) isa SVector
```
