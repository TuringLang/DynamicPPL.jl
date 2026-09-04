# Conditioning and fixing

DynamicPPL allows you to first *define* a model, and then *modify* it by either conditioning on observed data, or fixing variables to specific values.
This is useful for defining models once and then using them in different ways.

As an example, one could define a linear regression model as follows:

```@example 1
using DynamicPPL, Distributions

@model function linear_regression(x)
    m ~ Normal(0, 1)
    c ~ Normal(0, 1)

    y = Vector{Float64}(undef, length(x))
    for i in eachindex(x)
        y[i] ~ Normal(m * x[i] + c, 1.0)
    end
end
```

This model has no observed data: none of its sites are conditioned, so all the `y[i]`'s are latent variables.

!!! note "Why do we need to define `y` in the model?"
    
    The definition of `y` in the model is needed so that there is somewhere to assign `y[i]` to after the tilde-statement runs. If we did not define `y`, we would get an error when trying to call `setindex!` on an undefined variable.
    
    Model arguments supply default observations for sites with the same name. Local storage such as `y` does not: its sites are latent until conditioned or fixed. Use `decondition(model, :x)` to remove an argument observation, and `condition(model; x=new_data)` to replace or restore it.

Let's create some synthetic data to work with:

```@example 1
true_m, true_c = 5.0, 3.0

x = 0:0.1:0.5
y_data = true_m .* x .+ true_c .+ randn(length(x))
```

If we run the model before conditioning on `y`, we will find that all of `m`, `c`, and `y` are drawn from the prior distribution.

```@example 1
model = linear_regression(x)

# Here, `rand(model())` samples from the prior distribution and returns a
# VarNamedTuple of latent variables.
rand(model)
```

We could, for example, do this many times, and compute the prior mean of `y`.
This is analogous to using Turing's `Prior()` sampler.

```@example 1
vnts = [rand(model) for _ in 1:1000]
mean(vnt[@varname(y)] for vnt in vnts)
```

This is useful for prior predictive checks, for example.

## Conditioning

Arguments used on the left-hand side of `~` provide default observations. For example,
`@model f(x) = x ~ Normal()` makes `f(1.0)` equivalent to `f(1.0) | (x=1.0,)`.
`decondition(f(1.0), :x)` makes `x` latent, and conditioning that model on `x=2.0`
restores an observation with the new value. Other arguments remain ordinary model inputs.

Replacing a complete argument updates its value, shape, and dispatch type parameters before
the model body runs. Partial updates preserve the remaining stored values and their array
templates. Arguments with unobserved entries retain their original storage template; the
corresponding tilde statements fill those entries during evaluation.

To condition the model on observed data, we can use the `condition` function, or its alias `|`.
The most robust way of conditioning is to provide a `VarNamedTuple` that holds the values to condition on.

```@example 1
# Construct a `VarNamedTuple` that holds the conditioning values.
observations = @vnt begin
    y := y_data
end

# Equivalently: conditioned_model = condition(model, observations).
cond_model = model | observations
```

We can inspect the values that have been conditioned on, using the `conditioned` function:

```@example 1
conditioned(cond_model)
```

If we were to run this model, we would now find that `y` is an observed variable, and thus it is not sampled:

```@example 1
parameters = rand(cond_model)
```

We can't directly draw from the posterior using DynamicPPL (`rand` still draws from the prior).
However, since this is now an observed variable, the log-likelihood associated with the newly provided `y` will be computed:

```@example 1
loglikelihood(cond_model, parameters)
```

and this quantity can be used by MCMC algorithms to draw samples from the posterior distribution.

## Fixing

Fixing is exactly the same as conditioning, except that instead of incrementing the log-likelihood, there is no log-probability contribution from the fixed variables.

In essence, fixing a variable `x ~ dist` to a value `x_val` is equivalent to replacing the statement with `x = x_val`, which removes it from the model entirely.

We can illustrate this by fixing the intercept `c` to its true value:

```@example 1
# Construct a `VarNamedTuple` that holds the fixed values.
fix_values = VarNamedTuple(; c=true_c)

fixed_model = fix(model, fix_values)
```

and sampling from the prior again:

```@example 1
parameters_fixed = rand(fixed_model)
```

If we were to repeat this many times, we would find that `y` is drawn from its prior, but because `c` is fixed, the samples will reflect that:

```@example 1
mean(vnt[@varname(y)] for vnt in [rand(fixed_model) for _ in 1:1000])
```

## Supplying parameters to condition or fix on

In the above examples we have provided the conditioning and fixing values as `VarNamedTuple`s.
Internally, DynamicPPL stores the values as `VarNamedTuple`s, and it is strongly recommended that you construct them this way.

For convenience, both `condition` and `fix` also accept a variety of different input formats:

```julia
# NamedTuple
model | (; y=y_data)

# AbstractDict{VarName}
model | Dict(@varname(y) => y_data)

# Pair
model | (@varname(y) => y_data)
```

**Note, however, that these alternative input formats are not necessarily rich enough to capture all the necessary information.
We recommend using `VarNamedTuple`s directly in all cases.**

For example, if you only wanted to condition `y[1]` but not the other `y[i]`'s, you cannot specify this via a `NamedTuple`, since `NamedTuple`s require `Symbol`s as keys.

You can easily specify this via `VarNamedTuple` and its helper macro [`@vnt`](@ref):

```@example 1
vnt = @vnt begin
    y[1] := y_data[1]
end
```

Note that in this case since the `VarNamedTuple` has no knowledge of the length or shape of `y`, DynamicPPL will assume that `y` is a `Base.Vector` of unknown length (hence the `GrowableArray` above).

This will work fine as long as `y` is indeed a `Base.Vector`.
However, if you want to avoid this, you should provide the full shape of `y` when defining the `VarNamedTuple`:

```@example 1
vnt = @vnt begin
    @template y = y_data
    y[1] := y_data[1]
end
```

Now, the variable `y` is known to have the same shape and type as `y_data`.

!!! warning
    
    If you use custom array types in DynamicPPL that have different indexing semantics from `Base.Array`, then the templating shown here becomes mandatory. For example, `OffsetArray`s may behave incorrectly if templates are not supplied.

If we run the model again, we should find that `y[1]` is no longer sampled:

```@example 1
cond_model_partial = model | vnt
rand(cond_model_partial)
```

## Missing data

Leave unobserved sites out of the conditioned values. `missing` is not a stochastic-role marker and is rejected by `condition` and `fix`.
For an array whose elements have separate tilde statements, condition only the observed indices, as in the examples above. A single multivariate draw cannot be partially conditioned.
