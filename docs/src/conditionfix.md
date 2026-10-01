# Conditioning and fixing

## Binding rules

  - An *LHS variable* is the variable on the left-hand side of one execution of a tilde
    statement, identified by its address. An *LHS subvariable* is a part of one LHS
    variable, such as `x[1]` of `x ~ MvNormal(...)`. A *binding* pairs an address with
    a value and is stored as conditioned or fixed. The effective binding sets the LHS
    variable's *role*: observed values contribute to the likelihood; fixed values
    contribute no log probability. Both replace sampling. With no effective binding,
    the LHS variable is latent.
    An *explicit binding* is made by `condition` or `fix`.
    Fixed values must cover every LHS variable they bind with a static size and shape: changing a
    fixed argument's size or shape in the body throws `ArgumentError` naming the LHS variable.
    This restriction does not apply to conditioned values.
  - An *argument LHS variable* is an LHS variable whose address's top symbol names a
    model argument. With `@model`, the argument supplies an *argument-supplied
    observation*: a conditioned binding. Binding an argument replaces its value
    in the model body. Observed LHS variables use the value computed by the body; fixed LHS variables
    reset to their bound value when their tilde statement runs.
    Direct `Model` construction records no observations; pass `lhs_arguments` to declare
    which arguments can be bound with `condition` or `fix`.
    A *submodel return value* is the value assigned to the LHS variable by a submodel
    tilde (`a ~ to_submodel(...)`). If `a` is an argument, it supplies only the value
    before the tilde runs; its argument-supplied observation is ignored at that tilde,
    so it needs no `decondition`. A `NamedTuple` argument at a submodel tilde
    does not supply a submodel namespace. Bind the child before `to_submodel`,
    or use `@varname(a.x)` on a parent whose submodel LHS `a` is not an argument.
  - Integer indices into NamedTuples are rejected in binding addresses and LHS variables;
    use `x.a` instead of `x[1]` for `x = (a=1.0, b=2.0)` (Tuples retain integer indices).
  - Later bindings replace earlier ones where they overlap. A *whole binding* binds an
    entire value; a *partial binding* binds part of a value already bound as a whole,
    preserving the rest of its binding. Subvariables of one LHS variable cannot
    have different roles; mixing roles throws `ArgumentError` during evaluation.
  - The *effective binding* is the binding in force at an LHS variable. Precedence, highest
    first: explicit bindings of enclosing models that reach it through its submodel
    namespace, outermost first; then the explicit binding of the model containing
    the tilde; then that model's argument-supplied observation. A parent's
    argument-supplied observations never reach a child. With no effective binding,
    the LHS variable is latent and gets its value from initialization.
    The *submodel namespace* is the set of parent addresses that reach a child's
    LHS variables, such as `@varname(a.x)`, or the child's unchanged names with
    `auto_prefix=false` (unless the child was manually prefixed).
  - Explicitly binding a submodel return value throws `ArgumentError` at the submodel
    tilde. For an argument LHS variable, bindings below its address are rejected too,
    during evaluation or at the `condition` or `fix` call when the argument's type
    rules out the requested field. Bind the
    child before wrapping it with `to_submodel` instead. Binding an argument with
    no LHS variables, or a nonexistent field or index of an argument, throws at the `condition`
    or `fix` call; construct the model with a new argument value instead.
  - `decondition` removes this model's conditioned bindings; `unfix` removes its fixed
    bindings at the requested names. A name matches if it equals, contains, or is contained in a
    stored binding's address, with Symbol indices equivalent to properties.
    A name with no match throws `ArgumentError`, including bindings supplied only by
    a child. Decondition child argument-supplied observations before `to_submodel`. With no
    names, all conditioned or fixed bindings, respectively, are removed. `unfix` restores the argument-supplied
    observation, if any, otherwise making the LHS variable latent; it never restores an
    earlier explicit conditioned binding. Argument-supplied observations are rebuilt from the model's arguments,
    so `unfix(fix(decondition(m, :x); x=5.0), :x)` also restores an argument-supplied observation
    previously removed by `decondition`. `decondition(m, :x)` removes explicit and argument-supplied
    observations at `x`, making it latent.
  - `conditioned` and `fixed` return plain values, independent of binding history:
    `VarNamedTuple`, `PartialArray`, or ordinary values. Partial removal or mixed
    roles produce plain partial values, not the original container type.
  - An argument equal to `nothing` supplies no argument-supplied observation, so its
    LHS variables are latent unless explicitly bound. Only a whole argument is a
    placeholder; `nothing` inside a container and `missing` are not placeholders.
  - A value containing `missing` is rejected when a tilde statement observes or fixes
    it, with an `ArgumentError` naming the LHS variable. Parts of an argument or binding
    that no tilde statement reads may contain `missing`. It no longer marks an LHS
    variable as latent: omit its explicit binding or `decondition` the argument LHS
    variable (see [Missing data](@ref)).
    `InitFromParams` rejects it when the parameter is read during initialization,
    not at construction. Leave unobserved values out instead.
  - Whole bindings use the supplied object without copying. When a partial binding
    splits a whole binding, the remaining parts are captured at that time; later
    changes to the supplied container's entries are not reflected in those parts.
    The model body must not mutate bound values, directly or through an alias such
    as a `view`. Partial bindings on array arguments rebuild
    the argument in O(length) per evaluation; prefer whole replacements for large arrays.
  - Bindings unused by any executed LHS variable are ignored, including unknown
    names and LHS variables in branches that do not run.

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

This model has no observed data: none of its LHS variables have conditioned bindings, so all the `y[i]` LHS variables are latent.

!!! note "Why do we need to define `y` in the model?"
    
    The definition of `y` in the model is needed so that there is somewhere to assign `y[i]` to after the tilde-statement runs. If we did not define `y`, we would get an error when trying to call `setindex!` on an undefined variable.
    
    Local storage such as `y` does not supply observations: its LHS variables are latent until conditioned or fixed.

Let's create some synthetic data to work with:

```@example 1
true_m, true_c = 5.0, 3.0

x = 0:0.1:0.5
y_data = true_m .* x .+ true_c .+ randn(length(x))
```

If we run the model before conditioning on `y`, we will find that all of `m`, `c`, and `y` are drawn from the prior distribution.

```@example 1
model = linear_regression(x)

# Here, `rand(model)` samples from the prior distribution and returns a
# VarNamedTuple of latent LHS variables.
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

Binding a whole argument replaces its value, shape, and dispatch type parameters before
the model body runs, provided it matches the declared argument types. For example, replacing
`x::Vector{Float64}` with `[1, 2]` throws `ArgumentError`; use `[1.0, 2.0]` instead.
Partial updates preserve the remaining stored values and their array
templates. Values in partial bindings are converted to the argument array's element type;
values that cannot be represented exactly throw `ArgumentError`.
Arguments with unobserved entries retain their original storage template; the
corresponding tilde statements fill those entries during evaluation.
Defaults derived from an argument are evaluated at model construction; binding that argument
does not recompute them.
Keyword-splat arguments support whole replacements; their entries cannot be bound separately.

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

If we were to run this model, we would now find that the `y[i]` LHS variables are observed, and thus they are not sampled:

```@example 1
parameters = rand(cond_model)
```

We can't directly draw from the posterior using DynamicPPL (`rand` still draws from the prior).
However, since these LHS variables are now observed, the log-likelihood associated with the newly provided `y` will be computed:

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

A value containing `missing` is rejected when a tilde statement observes or fixes it.
The `ArgumentError` names the LHS variable, rather than the LHS subvariable containing
`missing`. Parts of an argument or binding that no tilde statement reads may contain
`missing`: for `@model metadata_lhs(p) = p.a ~ Normal()`,
`metadata_lhs((a=1.0, b=missing))` constructs and evaluates successfully, while
`metadata_lhs((a=missing, b=1.0))` constructs but throws when evaluated, naming `p.a`.
The same rule applies to explicit bindings from `condition` and `fix`.

`missing` no longer marks an LHS variable as latent. Omit its explicit binding, or use
`decondition(m, @varname(x))` to make an argument LHS variable latent before evaluation.
For `@model observed(x) = x ~ Normal()`, `decondition(observed(missing))(rng)` samples `x`.
For an array whose indices are separate LHS variables, decondition only the desired
indices. A single multivariate LHS variable cannot be partially conditioned:
`x ~ MvNormal(...)` rejects a value containing `missing`, naming `x`.
