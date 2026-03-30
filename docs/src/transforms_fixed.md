# [Fixing transformations](@id fixed-transforms)

For some models, it may be known that the support of a variable does not change, and that the transformations should be fixed.
This allows us to avoid the overhead of recomputing the transformation at every model evaluation.

You can write your own transform strategy to do this (as described on the [main transforms page](@ref custom-transform-strategy)), but DynamicPPL also provides a built-in strategy for this, namely [`WithTransforms`](@ref).
This takes a `VarNamedTuple` of transforms plus a fallback strategy for when the target variable is not in the `VarNamedTuple`.

We'll first show the full usage of `WithTransforms` here.
It should be noted that there are more convenient ways to enable this at a high level, which will be discussed later.

```@example 1
using DynamicPPL: WithTransforms, FixedTransform, Unlink, @vnt, LinkAll

# Make your own custom transform. Note that by convention, the stored function
# always converts *from* the transformed value *to* the raw value.
function my_transform end

# Store them in a VarNamedTuple. You can mix and match different transform
# types here, as long as they are subtypes of `AbstractTransform`.
vnt = @vnt begin
    a := FixedTransform(my_transform)
    b := Unlink()
end

# Use the WithTransforms strategy to specify that these transforms should be
# used. All other variables will use the fallback strategy, here `LinkAll()`
# (note that the fallback must be a transform *strategy*, not a *transform*).
tfm_strategy = WithTransforms(vnt, LinkAll())
```

## Requirements for fixed transforms

`tfm_strategy` will eventually be the transform strategy that is passed to various DynamicPPL functions, including `init!!` and `LogDensityFunction`.
However, before we can do that, we need to make sure that the transform we are using is properly defined.
(Notice we avoided doing so above!)
Specifically, the minimum interface required is:

  - `InverseFunctions.inverse` must be implemented for the transform; this allows DynamicPPL to generate transformed values.
  - `ChangesOfVariables.with_logabsdet_jacobian` must be implemented for both the transform and its inverse; this allows DynamicPPL to apply the log-Jacobian correction.

```@example 1
# Bijectors re-exports both functions, so we can import from there.
import Bijectors

my_inverse_transform(x::Real) = [log(x)]
Bijectors.inverse(::typeof(my_transform)) = my_inverse_transform
Bijectors.inverse(::typeof(my_inverse_transform)) = my_transform

function Bijectors.with_logabsdet_jacobian(::typeof(my_transform), tfmx::AbstractVector)
    y = only(tfmx)
    return (exp(y), y)
end
function Bijectors.with_logabsdet_jacobian(::typeof(my_inverse_transform), x::Real)
    logx = log(x)
    return ([logx], -logx)
end
```

(Often, if the transform must store some kind of parameter (e.g. size), it is easier to make it a callable struct.)

Once that's done, you can use this transform strategy:

```@example 1
using DynamicPPL, Distributions, LogDensityProblems

@model function f()
    a ~ Exponential()
    return b ~ Normal()
end
model = f()

ldf = LogDensityFunction(model, getlogjoint_internal, tfm_strategy)

# Notice that the transformed value of `x` is negative here.
params = [-5.0, 2.0]
LogDensityProblems.logdensity(ldf, params)
```

In this specific instance, the transform we have chosen correlates exactly to the transform that `DynamicLink` would have chosen for `a`, so we can verify that the log-density is the same as when using `LinkAll()`:

```@example 1
ldf_link_all = LogDensityFunction(model, getlogjoint_internal, LinkAll())
LogDensityProblems.logdensity(ldf_link_all, params)
```

## Deriving a set of fixed transforms

In the above example, we manually defined a transform and a `VarNamedTuple` thereof.
Bijectors.jl already provides a large number of transforms (which are exactly the transforms that are derived at runtime), and you may want to use those directly.
To do so, you will have to run the model once with the desired transform strategy (e.g. `LinkAll()`), and collect the transforms that were used for each variable.

This is most easily done via

```@example 1
using DynamicPPL

get_fixed_transforms(model, LinkAll())
```

## Faster constructors

While the sections above clearly demonstrate how to construct a set of fixed transforms, it is still slightly verbose.
For this reason, `LogDensityFunction` in particular provides a keyword argument `fix_transforms` that allows you to specify a transform strategy for which you want to fix the transforms.

For example,

```@example 1
ldf1 = LogDensityFunction(model, getlogjoint_internal, LinkAll(); fix_transforms=true)
```

is equivalent to writing

```@example 1
tfm_strategy = WithTransforms(get_fixed_transforms(model, LinkAll()), LinkAll())
ldf2 = LogDensityFunction(model, getlogjoint_internal, tfm_strategy)
```

## Performance

It should be noted that using fixed transforms does not *always* lead to speedups, since the calculation of the transform is often very cheap and comparable to the cost of looking up the cached transform.
If you want to use this feature, we recommend benchmarking your model.
For example:

```@example fixed2
using DynamicPPL, Distributions, LogDensityProblems, Chairmarks, LinearAlgebra, ADTypes
import ForwardDiff

adtype = AutoForwardDiff()

@model function eightsch(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ MvNormal(fill(mu, J), tau^2 * I)
    for i in 1:J
        y[i] ~ Normal(theta[i], sigma[i])
    end
end
J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
model = eightsch(J, y, sigma)

ldf_dynamic = LogDensityFunction(model, getlogjoint_internal, LinkAll(); adtype=adtype)
x = rand(ldf_dynamic)

@b LogDensityProblems.logdensity($ldf_dynamic, $x)
```

```@example fixed2
@b LogDensityProblems.logdensity_and_gradient($ldf_dynamic, $x)
```

In the following code blocks, you *should* see that the fixed transform takes almost exactly the same time as the dynamic transform (although the exact number will of course have some variance).
This is because the distributions in the eight-schools model above are 'simple' enough that re-deriving them is essentially instantaneous.

```@example fixed2
ldf_fixed = LogDensityFunction(
    model, getlogjoint_internal, LinkAll(); fix_transforms=true, adtype=adtype
)
@b LogDensityProblems.logdensity($ldf_fixed, $x)
```

```@example fixed2
@b LogDensityProblems.logdensity_and_gradient($ldf_fixed, $x)
```

For other distributions, however, the fixed transform can be much faster.
