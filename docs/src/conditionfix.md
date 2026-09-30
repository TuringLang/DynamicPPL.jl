# Conditioning and fixing

## Binding rules

  - A conditioned value is an observation and contributes to the likelihood. A fixed
    value is a constant and contributes no log probability. Both replace sampling.
    Fixed values must cover every site they bind with a static size and shape: changing a
    fixed argument's size or shape in the body throws `ArgumentError` naming the site.
    This restriction does not apply to conditioned values.
  - With `@model`, an argument on the left of `~` supplies a default observation.
    Direct `Model` construction without bindings records no observations or argument sites.
    An argument used as a submodel's left-hand side (`a ~ to_submodel(...)`) supplies a return-value
    buffer instead; it needs no `decondition`. Even a `NamedTuple` buffer is only an
    initial return value, not a child namespace. Bind the child before `to_submodel`,
    or use `@varname(a.x)` on a parent whose submodel LHS `a` is not an argument.
  - Later bindings replace earlier ones where they overlap. A whole binding replaces
    all components; a component binding replaces only that component. A single tilde
    statement's value cannot mix conditioned, fixed, or unbound components; such sites
    throw `ArgumentError` during evaluation.
  - Parent bindings override child bindings at the same address. Use the submodel's
    names in the parent, such as `@varname(a.x)`, or unprefixed names with
    `auto_prefix=false` (unless the child was manually prefixed).
  - Binding a submodel's return value, or anything below an argument-backed return
    buffer, throws `ArgumentError` at the submodel tilde during evaluation. Bind the
    child before wrapping it with `to_submodel` instead. Binding an argument that is
    not a tilde site, or a nonexistent field of an argument, throws at the `condition`
    or `fix` call; construct the model with
    a new argument value instead.
  - `decondition` and `unfix` remove this model's bindings of their own role at the
    requested names. A name matches if it equals, contains, or is contained in a
    stored binding's address, after resolving equivalent index and property forms.
    A name with no match throws `ArgumentError`, including bindings supplied only by
    a child. Decondition child argument observations before `to_submodel`. With no
    names, all bindings of the requested role are removed. `unfix` restores the argument's
    default observation, if any, otherwise making the site latent; it never restores an
    earlier explicit condition. `decondition(m, :x)` removes explicit and argument-default
    observations at `x`, making it latent.
  - `conditioned` and `fixed` return plain values, independent of binding history:
    `VarNamedTuple`, `PartialArray`, or ordinary values. Partial removal or mixed
    roles produce plain partial values, not the original container type.
  - `missing` is rejected at construction in tilde arguments and values supplied to
    `condition` or `fix`, recursively throughout tuples, named tuples, and assigned array
    entries, including unused parts of those arguments. Custom struct fields are checked
    only at executed sites. Omit unobserved bindings, or construct with a non-missing
    argument placeholder and `decondition` it (see [Missing data](@ref)).
    `InitFromParams` rejects it when the parameter is read during initialization,
    not at construction. Leave unobserved values out instead.
  - Whole bindings use the supplied object without copying. When a component binding
    splits a whole binding, the remaining elements are captured at that time; later
    changes to the supplied container's entries are not reflected in those elements.
    The model body must not mutate bound values, directly or through an alias such
    as a `view`. Component bindings on array arguments rebuild
    the argument in O(length) per evaluation; prefer whole replacements for large arrays.
  - Bindings unused by any executed tilde statement are ignored, including unknown
    names and sites in branches that do not run.

## Example

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
    
    Local storage such as `y` does not supply observations: its sites are latent until conditioned or fixed.

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

Replacing a complete argument updates its value, shape, and dispatch type parameters before
the model body runs, provided it matches the declared argument types. For example, replacing
`x::Vector{Float64}` with `[1, 2]` throws `ArgumentError`; use `[1.0, 2.0]` instead.
Partial updates preserve the remaining stored values and their array
templates. Component values are converted to the argument array's element type;
values that cannot be represented exactly throw `ArgumentError`.
Arguments with unobserved entries retain their original storage template; the
corresponding tilde statements fill those entries during evaluation.
Defaults derived from a replaced argument are evaluated at model construction and are not recomputed.

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

Construction checks all parts of tilde arguments recursively through tuples, named tuples,
and assigned array entries, even when a part has no tilde statement. For
`@model metadata_site(p) = p.a ~ Normal()`, `metadata_site((a=1.0, b=missing))` therefore
throws at construction. Custom struct fields are not searched at construction; `missing`
is rejected when such a field is used at an executed tilde site.

Omit unobserved values from `condition` or `fix`. For argument observations, supply a
non-missing placeholder first, then call `decondition(m, :x)` (or remove a component)
to make the desired sites latent.
For an array whose elements have separate tilde statements, condition only the observed indices, as in the examples above. A single multivariate draw cannot be partially conditioned.
