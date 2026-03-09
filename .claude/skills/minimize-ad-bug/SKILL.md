---
name: minimize-ad-bug
description: Minimize an AD bug by systematically stripping a failing DynamicPPL model down to its root cause. Use when a model + AD backend gives wrong gradients or errors.
argument-hint: [description of the failing model and AD backend]
disable-model-invocation: true
---

# Minimize AD Bug

You are helping to minimize an AD bug in DynamicPPL. The user has a `@model` and an `AbstractADType` that together produce wrong results. Your goal is to systematically strip away layers of abstraction until you isolate the root cause.

For a full worked example walking through each phase, see [example.md](example.md).

## Failure modes

The bug may manifest in different ways:

  - **Numerical inaccuracy**: the AD gradient disagrees with finite differences or another backend
  - **Hard error**: the AD call throws an exception
  - **Unexpected type/value**: e.g. returns `NaN`, `Inf`, or wrong-shaped output

At each step, you need to check that **the same bug** is still present. If the error changes character (e.g. a numerical inaccuracy becomes a different exception, or the error message changes substantially), stop and report this to the user before continuing. The user will decide whether the new error is related or whether you've gone down the wrong path.

## Overall strategy

Work iteratively. At each step, try ONE simplification, re-run the failing case, and check whether the bug is still present. If removing something makes the bug disappear, put it back and try a different simplification.

The general order of simplifications (from outermost to innermost) is:

 1. **Simplify the model** (remove variables, simplify distributions, reduce dimensions)
 2. **Extract the differentiated function** (bypass LogDensityFunction, call AD directly)
 3. **Desugar `@model`** (replace with hand-written evaluation function, expand tilde_assume!! calls)
 4. **Minimize the pure function** (inline, simplify, reduce until the root cause is bare)

## Phase 1: Simplify the model

Before desugaring anything, try to make the model as small as possible while still reproducing the bug.

  - Remove variables that aren't needed to trigger the bug
  - Replace complex distributions with simpler ones (e.g. `MvNormal(mu, Sigma)` -> `Normal()`)
  - Replace complex expressions with constants where possible
  - Remove return values, `@addlogprob!`, `:=` statements if not needed
  - Reduce array dimensions (e.g. `x[1:100]` -> `x[1:2]` or scalar `x`)
  - Remove conditioning/fixing if not needed for the bug

To test whether the bug reproduces after each simplification, use `LogDensityFunction` + `LogDensityProblems.logdensity_and_gradient`:

```julia
using DynamicPPL, Distributions, ADTypes
import LogDensityProblems

# Construct the LDF. Always use LinkAll().
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll(); adtype=adtype)
params = rand(ldf)  # generate params, or use specific values

# Test: does this error, or give wrong results?
LogDensityProblems.logdensity_and_gradient(ldf, params)
```

Once you have a reproducer, pin `params` to specific values for determinism. You can compare against finite differences or a known-good backend to check correctness.

## Phase 2: Extract the differentiated function

The function that DynamicPPL actually differentiates (see `logdensity_at()`) is:

```julia
function f(params, model, oavi, varname_ranges, transform_strategy)
    _, oavi = DynamicPPL.init!!(
        model,
        oavi,
        DynamicPPL.InitFromVector(params, varname_ranges, transform_strategy),
        transform_strategy,
    )
    return getlogjoint(oavi)
end
```

To set up the arguments, create a `LogDensityFunction` and extract what you need from it:

```julia
ldf = LogDensityFunction(model, getlogjoint_internal, LinkAll())
params = rand(ldf)  # or specific values

oavi = OnlyAccsVarInfo(ldf._accs)
varname_ranges = ldf._varname_ranges
transform_strategy = ldf.transform_strategy
```

Then call AD directly on `f`. **For Enzyme, always call Enzyme directly (not via DifferentiationInterface):**

```julia
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
```

For other backends such as ReverseDiff, you can go via DifferentiationInterface:

```julia
import DifferentiationInterface as DI
import ReverseDiff

DI.value_and_gradient(
    f,
    AutoReverseDiff(),
    params,
    DI.Constant(model),
    DI.Constant(oavi),
    DI.Constant(varname_ranges),
    DI.Constant(transform_strategy),
)
```

Verify the same bug reproduces. Pin `params` to specific values for determinism.

## Phase 3: Desugar `@model` and expand tilde internals

Once you have the standalone `f` reproducing the bug, start expanding what happens inside it.

### 3a: Desugar the `@model` macro

Start by directly replacing each tilde statement with its corresponding `tilde_assume!!` or `tilde_observe!!` call. This is almost always sufficient — 99% of the time the error lies in `tilde_assume!!`, not in the other macro machinery.

```julia
# x ~ Normal()  becomes:
x, __varinfo__ = DynamicPPL.tilde_assume!!(
    __model__.context, Normal(), @varname(x), NoTemplate(), __varinfo__
)

# 1.0 ~ Normal(x)  becomes (observed data):
_, __varinfo__ = DynamicPPL.tilde_observe!!(
    __model__.context, Normal(x), 1.0, @varname(y), NoTemplate(), __varinfo__
)
```

A desugared model is a function `(model, varinfo, args...) -> (retval, varinfo)`. You can construct a `Model` manually:

```julia
function my_model_eval(__model__, __varinfo__)
    # ... desugared tilde statements ...
    return nothing, __varinfo__
end

model = DynamicPPL.Model{false}(my_model_eval, NamedTuple())
```

Only resort to the full `@macroexpand` output if the simple substitution doesn't reproduce the bug (which is rare).

### 3b: Hardcode constant arguments

A key simplification technique is to replace `Const`/`Constant` arguments with hardcoded values inside the function. This eliminates arguments and unlocks further simplifications:

  - **`transform_strategy`**: If it's `LinkAll()`, hardcode it. Many internal functions like `apply_transform_strategy` branch on `if transform_strategy isa LinkAll` — you can then inline only the `LinkAll` branch and delete all other branches.
  - **`varname_ranges`**: Replace with the literal ranges. For example, if `varname_ranges` maps `x` to `1:1` and `y` to `2:3`, just hardcode `view(params, 1:1)` and `view(params, 2:3)` directly.
  - **`oavi`**: Once accumulators are simple enough, inline their construction.
  - **`model`**: Once the model function is desugared, you can inline it directly and remove the `Model` wrapper.

Each time you hardcode something, the function gets simpler and has fewer arguments, making AD's job easier to trace.

### 3c: Expand `tilde_assume!!`

Under `InitContext`, each `tilde_assume!!` does three things:

```julia
# 1. Init: read parameter value from the vector
init_tval = DynamicPPL.init(ctx.rng, vn, dist, ctx.strategy)

# 2. Transform: compute raw value, transformed value, and log-Jacobian
x, tval, logjac = DynamicPPL.apply_transform_strategy(
    ctx.transform_strategy, init_tval, vn, dist
)

# 3. Accumulate: update accumulators (logprior, loglikelihood, etc.)
vi = DynamicPPL.setindex_with_dist!!(vi, tval, dist, vn, template)
vi = DynamicPPL.accumulate_assume!!(vi, x, tval, logjac, vn, dist, template)
```

Expand these one at a time, inlining each call, to narrow down which step causes the AD failure. Since you've already hardcoded `transform_strategy` to `LinkAll()`, you can inline only the linked branch of `apply_transform_strategy`.

## Phase 4: Minimize the pure function

At this point you should have a plain Julia function `f(θ::Vector) -> scalar`. Simplify it further:

  - Inline function calls one at a time
  - Replace library calls with their implementations
  - Remove branches that aren't taken for the test input
  - Simplify math expressions

Compare against finite differences to verify correctness:

```julia
using FiniteDifferences
fd_grad = FiniteDifferences.grad(central_fdm(5, 1), f, params)[1]
```

## Checking the bug at each step

For **numerical inaccuracy**, compare against finite differences or a known-good backend.

For **hard errors**, catch and compare the exception type and message. If the error type or message changes meaningfully after a simplification, stop and report to the user.

For either case, always pin `params` to specific values once you have a reproducer, so results are deterministic.

## Reporting

Once minimized, summarize:

 1. **Minimal reproducer** — the smallest code that demonstrates the bug
 2. **Which AD backend** is affected
 3. **Expected vs actual** gradient values (or error message)
 4. **Root cause** — which operation/function the AD backend handles incorrectly
