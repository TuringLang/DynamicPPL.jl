# JULIA.md

Guidance for coding agents reviewing or changing Julia code in this repository.

## Engineering Practices

- Write generic numeric code unless the math or an external API requires a concrete type. Avoid unnecessary `Float64`, `Int`, `Real`, `Array`, `Vector`, and `Matrix` constraints.
- Preserve caller intent with `zero(x)`, `one(x)`, `oneunit(x)`, `oftype`, `promote`, and `promote_type`, especially for `Float32`, `BigFloat`, AD numbers, units, GPU scalars, and symbolic values.
- Keep struct fields concrete; prefer parametric fields over `field::Number` or `field::AbstractVector`.
- Avoid unnecessary static parameters. Julia specializes on most ordinary argument types, but is conservative for `Type`, `Function`, and `Vararg`. Use `f(x, ::Type{T}) where {T}` when the type itself must specialize.
- Check type stability with `@inferred`, `@code_warntype`, and focused tests when changing compiler output, VNTs, accumulators, transforms, or log-density paths.
- Benchmark generated functions, macro output, and hot-path refactors before assuming simpler code is equivalent.
- Prefer dispatch and small protocol functions over large conditional blocks.
- Avoid broad overloads of Base functions for arbitrary input types; they can create method ambiguities and accidental API.
- Put backend-specific behaviour in package extensions or narrow integration layers when possible.
- Make direct dependencies explicit enough to version-bound and test. Do not rely on packages being loaded transitively.
- Use accessor functions for values downstream packages need. Direct field access from another package turns internal representation into accidental API.
- Prefer `Base.maybeview` over eager slicing when indexed access should avoid allocations but still support tuples and scalar indexing.
- Avoid fragile output-type prediction. When possible, compute an initial value and allocate caches from the observed value.
- Keep doctests deterministic. Use `StableRNGs` when examples print random values.

Pattern:

```julia
# Avoid: too concrete and inference-hostile.
f(x::Float64) = x / 2
buf = zeros(Float64, length(xs))
struct Model
    scale::Number
end

# Prefer: generic arithmetic, input-derived allocation, concrete instances.
f(x) = x / oftype(x, 2)
buf = similar(xs, promote_type(eltype(xs), Float64), length(xs))
struct Model{T}
    scale::T
end
```

Public constructors are API too: if `NormalLike` is public, both `NormalLike(mu, sigma)` and `NormalLike(; mu, sigma)` are long-term commitments.

## API And Design Review

Review API convention separately from conceptual design.

For public signatures, check:

- dimension arguments use `dims=`, including tuple-valued `dims` when natural
- data comes first, except callable-first APIs such as `map(f, xs)`
- callable arguments come first when `do`-block syntax should work
- mutating and non-mutating pairs exist when both are natural
- configuration switches are keywords, not positional `Bool`, small `Int`, or `Symbol` flags
- reductions use `init=`, and sorting follows `lt=`, `by=`, and `rev=`
- output allocation respects input type via `similar` or an explicit destination buffer
- wrappers forward reasonable `kwargs...`
- related functions use consistent names, argument order, and keyword names
- type annotations are no narrower than the dispatch contract requires

For package design, check:

- exported names clearly belong to the package purpose
- abstractions have enough implementations to justify themselves
- concrete types that downstream users need to extend have protocol functions
- type parameters are used for dispatch, storage, or invariants
- overlapping functions are not historical duplicates
- expected inverse, mutating/non-mutating, parse/show, iteration, indexing, or conversion operations are present
- exported low-level helpers do not leak implementation details
- outputs of one public function can feed into related public functions
- custom `==` has matching `hash`, and `hash` is consistent with `isequal`
- duplicated Base functionality should instead extend a Base method
- failure behavior is consistent: throwing, `nothing`, sentinel values, or invalid results
- documented or tested internal names may need `public` or export annotations

Treat public constructors, keyword arguments, exported names, aliases, abstract supertypes, and traits as long-term commitments. If a concrete type is public, its constructors are public too.

## AD And Probability Code

AD rules should preserve tangent-space meaning, not just match a finite-difference check at one ordinary `Float64` point.

For AD rule code, check:

- `rrule` and `frule` outputs have the right primal and tangent shapes
- structural arguments such as functions and types return the appropriate non-tangent marker
- true zero derivatives use the appropriate zero-tangent marker
- cotangents are projected back when they may leave the primal representation
- non-differentiable cases are represented explicitly
- structured inputs such as `Diagonal`, `Symmetric`, sparse arrays, and factorizations are tested
- higher-order AD is considered when mutation or captured buffers are introduced

For complex AD, state whether the implementation assumes holomorphic differentiation over `Complex` or real differentiation after viewing `Complex` as two real variables. Test non-holomorphic functions such as `abs2`, `real`, `imag`, and `conj` for the real interpretation.

For probability code, separate sample type, mathematical support, and reference measure. Do not infer too much from names like `Discrete` or `Continuous`; floating-point samples can still have atoms, and densities are always with respect to a reference measure. `pdf` may mean a density with units for continuous variables, a mass with respect to counting measure for discrete variables, or something more subtle for mixed or censored distributions.

For probability changes, check:

- domain boundaries and invalid parameters
- `pdf`, `logpdf`, `cdf`, `logcdf`, and `quantile` consistency
- extreme tails and near-degenerate parameters
- explicit RNG threading instead of hidden global RNG use
- parameter gradients, not just gradients with respect to observations
- type stability when parameters have different but compatible types
- strong numerical references such as `BigFloat`, known identities, or independent implementations
- deterministic examples and doctests when random values are printed

Useful numerical checks:

```julia
@test isfinite(logpdf(d, x))
@test logcdf(d, x) <= 0
@test isapprox(f(Float64(x)), Float64(f(big(x))); rtol=1e-12)
@test rand(StableRNG(1), d) == rand(StableRNG(1), d)
```

## Backend Compatibility

Backend-compatible Julia code avoids scalar indexing, hidden CPU fallbacks, runtime string construction in kernels, and exception paths that compile into invalid device code.

```julia
# Avoid scalar loops when broadcast expresses the operation.
for i in eachindex(y)
    y[i] = f(x[i])
end

# Prefer backend-aware array operations.
y .= f.(x)
```

When moving structured objects to GPU storage, prefer explicit `Adapt.jl` support per type rather than generic field walking; generic reconstruction can violate constructor invariants or miss cached fields.

## Review Checklist

Search for:

- over-specific types: `Float64`, `Float32`, `Int`, `Real`, `Array`, `Vector`, `Matrix`
- concrete allocation: `zeros(Float64, ...)`, `ones(Float64, ...)`, `similar(x, Float64, ...)`
- branch conditions such as `x == 0` or `x == 1` that may behave poorly for AD numbers or NaNs
- `ccall` or conversions that force `Cdouble`
- `collect`, scalar indexing, or CPU-only fallbacks in code that may receive GPU arrays
- broad signatures such as `f(x::Any)`, `f(x::AbstractArray)`, or very general Base overloads

Request type-variety tests when code claims to be generic:

```julia
@test f(Float32(1)) isa Float32
@test f(big"1.0") isa BigFloat
@test ForwardDiff.derivative(x -> f(x), 1.0) isa Real
@test f(SVector(1.0, 2.0)) isa SVector
```

Split large changes by risk: mechanical cleanup, tests, non-breaking fixes, performance work, API additions, and breaking design changes.

Report findings in tiers:

- breaking API changes
- non-breaking additions or compatibility shims
- internal consistency or design questions

For each finding, include the current pattern, proposed direction, why it matters, and whether it is breaking. Pause for user approval before turning review findings into implementation work, and frame uncertain design items as questions about intent.
