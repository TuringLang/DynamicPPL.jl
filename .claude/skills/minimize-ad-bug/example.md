# Worked Example: Minimizing an Enzyme AD Bug

This is an illustrative walkthrough of how to minimize an AD bug. The specific
error shown here is fabricated, but the process is representative of real bugs.

## Starting point

The user reports that Enzyme fails on this model:

```julia
using DynamicPPL, Distributions, ADTypes
import Enzyme: set_runtime_activity, Reverse

@model function regression(x, y)
    α ~ Normal(0, 10)
    σ ~ InverseGamma(2, 3)
    for i in eachindex(y)
        y[i] ~ Normal(α * x[i], σ)
    end
end

model = regression(randn(20), randn(20))
adtype = AutoEnzyme(; mode=set_runtime_activity(Reverse))
```

You should verify this by running the following code:

```julia
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll(); adtype=adtype)
params = rand(ldf)

import LogDensityProblems
LogDensityProblems.logdensity_and_gradient(ldf, params)
# ERROR: Enzyme cannot differentiate through function <blah blah>
```

## Phase 1: Simplify the model

**Try 1:** Remove the loop and use a single observation.

```julia
@model function regression2(y1)
    α ~ Normal(0, 10)
    σ ~ InverseGamma(2, 3)
    return y1 ~ Normal(α, σ)
end

model = regression2(1.5)
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll(); adtype=adtype)
params = rand(ldf)
LogDensityProblems.logdensity_and_gradient(ldf, params)
# ERROR: same error → good, bug still present
```

**Try 2:** Remove α entirely.

```julia
@model function regression3(y1)
    σ ~ InverseGamma(2, 3)
    return y1 ~ Normal(0, σ)
end

model = regression3(1.5)
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll(); adtype=adtype)
params = rand(ldf)
LogDensityProblems.logdensity_and_gradient(ldf, params)
# ERROR: same error → good
```

**Try 3:** Remove the observation too — just keep σ.

```julia
@model function regression4()
    return σ ~ InverseGamma(2, 3)
end

model = regression4()
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll(); adtype=adtype)
params = rand(ldf)
LogDensityProblems.logdensity_and_gradient(ldf, params)
# ERROR: same error → good, minimal model found
```

Pin params for determinism: `params = [0.5]`.

## Phase 2: Extract the differentiated function

Extract the internals from the LDF and call Enzyme directly:

```julia
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll())

# Extract what we need
accs = ldf._accs
varname_ranges = ldf._varname_ranges
transform_strategy = ldf.transform_strategy

function f(params, model, accs, varname_ranges, transform_strategy)
    oavi = DynamicPPL.OnlyAccsVarInfo(accs)
    _, oavi = DynamicPPL.init!!(
        model,
        oavi,
        DynamicPPL.InitFromVector(params, varname_ranges, transform_strategy),
        transform_strategy,
    )
    return getlogjoint_internal(oavi)
end

params = [0.5]

import Enzyme
Enzyme.gradient(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    f,
    params,
    Enzyme.Const(model),
    Enzyme.Const(oavi),
    Enzyme.Const(varname_ranges),
    Enzyme.Const(transform_strategy),
)
# ERROR: same error → good
```

## Phase 3a: Desugar the @model macro

Replace the call to `init!!` with the actual code that `@model` generates.
Specifically, you can replace `~` calls in the model with `tilde_assume!!` or `tilde_observe!!`:

```julia
function f(params, model, accs, varname_ranges, transform_strategy)
    context = DynamicPPL.InitContext(
        Random.default_rng(),
        DynamicPPL.InitFromVector(params, varname_ranges, transform_strategy),
        transform_strategy,
    )
    oavi = DynamicPPL.OnlyAccsVarInfo(accs)
    _, oavi = DynamicPPL.tilde_assume!!(
        context, InverseGamma(2, 3), @varname(σ), DynamicPPL.NoTemplate(), oavi
    )
    return getlogjoint_internal(oavi)
end
```

Then you can test this with:

```julia
Enzyme.gradient(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    f,
    [0.5],
    Enzyme.Const(model),
    Enzyme.Const(accs),
    Enzyme.Const(varname_ranges),
    Enzyme.Const(transform_strategy),
)
# ERROR: same error → good
```

## Phase 3b: Hardcode constant arguments

Since `transform_strategy` is always `LinkAll()`, `varname_ranges` maps `σ` to
`RangeAndLinked(1:1, true)`, and we know the model + oavi structure, hardcode
everything. You can print out the specific `varname_ranges` from the LDF to get
the exact structure.

```julia
accs = DynamicPPL.default_accumulators()
varname_ranges = DynamicPPL.@vnt begin
    σ := DynamicPPL.RangeAndLinked(1:1, true)
end
transform_strategy = LinkAll()

function f2(params)
    context = DynamicPPL.InitContext(
        Random.default_rng(),
        DynamicPPL.InitFromVector(params, varname_ranges, transform_strategy),
        transform_strategy,
    )
    oavi = DynamicPPL.OnlyAccsVarInfo(accs)
    _, oavi = DynamicPPL.tilde_assume!!(
        context, InverseGamma(2, 3), @varname(σ), DynamicPPL.NoTemplate(), oavi
    )
    return getlogjoint_internal(oavi)
end

Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f2, [0.5])
# ERROR: same error → good, now f2 has only one argument
```

## Phase 3c: Expand tilde_assume!!

Now inline what `tilde_assume!!` does under `InitContext`. For a single variable
`σ ~ InverseGamma(2, 3)` with `LinkAll()`:

```julia
using Bijectors: Bijectors
using ChangesOfVariables: with_logabsdet_jacobian

function f3(params)
    dist = InverseGamma(2, 3)

    # Step 1: Init — read linked value from params vector
    linked_vec = view(params, 1:1)
    finvlink = Bijectors.VectorBijectors.from_linked_vec(dist)

    # Step 2: Transform — unlink to get raw value + logjac
    raw_vec, inv_logjac = with_logabsdet_jacobian(finvlink, linked_vec)
    σ = raw_vec[1]
    logjac = -inv_logjac  # forward logjac

    # Step 3: Accumulate — logprior - logjac
    logp = logpdf(dist, σ)
    return logp - logjac
end

Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f3, [0.5])
# ERROR: same error → good
```

Now try removing pieces. Does it fail without the logjac?

```julia
function f3a(params)
    dist = InverseGamma(2, 3)
    linked_vec = view(params, 1:1)
    finvlink = Bijectors.VectorBijectors.from_linked_vec(dist)
    raw_vec = finvlink(linked_vec)  # no logjac
    return logpdf(dist, raw_vec[1])
end

Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f3a, [0.5])
# Works! → The bug is in with_logabsdet_jacobian
```

Does it fail with just the logjac part?

```julia
function f3b(params)
    dist = InverseGamma(2, 3)
    finvlink = Bijectors.VectorBijectors.from_linked_vec(dist)
    _, inv_logjac = with_logabsdet_jacobian(finvlink, view(params, 1:1))
    return -inv_logjac
end

Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f3b, [0.5])
# ERROR: same error → confirmed: bug is in with_logabsdet_jacobian for this bijector
```

## Phase 4: Minimize the pure function

Now we can check finite differences:

```julia
using FiniteDifferences
FiniteDifferences.grad(central_fdm(5, 1), f3b, [0.5])
# Returns correct gradient → this is a pure Enzyme bug
```

If needed, inline `with_logabsdet_jacobian` further to find the exact line.
Check what the bijector actually is:

```julia
finvlink = Bijectors.VectorBijectors.from_linked_vec(InverseGamma(2, 3))
# e.g. some composition of Exp ∘ ...
```

Then write out its implementation manually and narrow down which sub-operation
Enzyme chokes on.

## Report

 1. **Minimal reproducer:**
    
    ```julia
    using Bijectors, Distributions, ChangesOfVariables
    function f(x)
        return last(
            with_logabsdet_jacobian(
                Bijectors.VectorBijectors.from_linked_vec(InverseGamma(2, 3)), x
            ),
        )
    end
    Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f, [0.5])
    ```

 2. **AD backend:** Enzyme (Reverse mode)
 3. **Expected:** Finite differences gives `[...]`; Enzyme throws `...`
 4. **Root cause:** `with_logabsdet_jacobian` for the InverseGamma bijector
    uses `<some function>` which Enzyme cannot differentiate
 5. **Upstream:** File against Bijectors.jl (or Enzyme.jl if the function
    should be differentiable)
