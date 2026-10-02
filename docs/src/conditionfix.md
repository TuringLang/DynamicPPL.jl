# Conditioning and fixing

## Binding rules

An **LHS variable** is the addressed left side of one execution of `~`; an **LHS subvariable** is part of it, such as `x[1]` of `x ~ MvNormal(...)`.
Its **role** is latent (initialised), observed (scored in the likelihood), or fixed (no log probability).
A **binding** supplies a bound value and an observed or fixed role: `condition`/`fix` make **explicit bindings**; arguments on the LHS supply **argument-supplied observations**.
The **effective binding** wins: enclosing models' explicit bindings, outermost first; then this model's fixed binding, explicit observation, or argument-supplied observation. Without one, the LHS variable is latent.
A parent's argument-supplied observations never reach a child.

Observations form the lower layer, fixed bindings the upper. Within a layer, later bindings replace earlier ones where they overlap.

| Operation     | Effect on this model's bindings                                                                             |
|:------------- |:----------------------------------------------------------------------------------------------------------- |
| `condition`   | Adds observations below fixed bindings.                                                                     |
| `fix`         | Adds fixed bindings, shadowing observations.                                                                |
| `decondition` | Removes observations of either origin, even under a fixed binding; otherwise makes the LHS variable latent. |
| `unfix`       | Removes fixed bindings, uncovering the observation below, or leaving the LHS variable latent.               |

Removal matches equal, enclosing or contained addresses (Symbol indices match properties); no match throws `ArgumentError`.
Remove child-only bindings on the child before `to_submodel`; with no names, removal clears this model's corresponding layer.

### Argument contract and shared constraints

**Roles and values.** Subvariables of one LHS variable must have the same role; mixing roles throws `ArgumentError` at evaluation.
Every bound LHS variable takes its value from its binding. A **local LHS variable** (whose top symbol is not an argument) reads it at its tilde.
For an **argument LHS variable**, every binding, explicit or argument-supplied, acts through the argument: it replaces the argument before the body runs,
and the tilde reads the argument's current value in the body; a fixed argument LHS variable is reset to its bound value at its tilde.
Thus `f(data)` and `condition(f(...); x=data)` observe the same value even when the body transforms `x` first; observe raw data under a separate name.
Deconditioned arguments keep their old values until their tilde. For direct construction and handwritten evaluator obligations, see [`Model`](@ref).

**Shape.** A binding owns the shape of the value at its address, at any depth, within the layer being edited.
Partial bindings/removals edit within the latest owner: binding `x=ones(3)` then removing `x[3]` preserves length three.
Binding `x[1]` or `p.a` may resize that value if its type permits, never a container above it; slice/range/`:`/mask extents must match.
Removing a binding returns its address to the next owner: the shadowed binding, enclosing binding, or argument.
The body sets local storage's shape and may resize conditioned arguments. **Fixed values are static** and must cover every reached LHS variable below their address.
Fixed argument tildes reject growth, shrinkage or reshaping with `ArgumentError` naming the LHS variable.

**Addresses and submodels.** NamedTuple fields require names (`x.a`), never integer indices, in bindings and LHS variables; Tuples keep integer indices.
Bindings on prefixed models must be at or below the prefix. A **submodel namespace** reaches child LHS variables through `a.x`, or unchanged names with `auto_prefix=false` (unless manually prefixed).
A **submodel return value**, assigned by `a ~ to_submodel(...)`, cannot be explicitly bound: evaluation throws `ArgumentError`.
If `a` is an argument, its observation is ignored there; explicit bindings below it also throw, possibly earlier if its type rules out the field.
A NamedTuple argument provides no child namespace; bind the child before `to_submodel`.

**Aliasing.** Bound values are not copied: never mutate them in the body, even through a `view`. Partial bindings snapshot their whole owner's remaining parts when made; later edits are not reflected there.

Bindings unused by reached LHS variables are ignored, including branches or submodels that do not run.

**Runtime bindings** are made inside the running body, e.g. `a ~ to_submodel(condition(child(y), @varname(y[1]) => m))`.
Remade each evaluation, their values carry derivatives with respect to enclosing latent variables; bindings made beforehand hold values constant with respect to model parameters.

**No data.** Use `decondition` to make observations latent. Whole `missing`/`nothing` arguments (even if replaced in the body) and bound values containing either throw `ArgumentError` at the reading tilde, naming the LHS variable; unread parts may contain either.
See [Missing data](@ref). `InitFromParams` rejects `missing` when read, not at construction; omit unobserved values instead.

### Binding contract

**Inputs.** Use NamedTuples/keywords for whole top-level values, `VarName` pairs for any address (`:x => v` abbreviates `@varname(x) => v`), or produced `VarNamedTuple`s.
Positional inputs and tuples apply left to right; every `AbstractDict` and unsupported input throws `ArgumentError`.
One positional **binding schema**, an AbstractPPL `OfNamedTuple` type (`@of(z = of(Array, 3))`), supplies storage for partially bound local LHS variables and binds nothing.
Import `of, @of` from AbstractPPL. Schemas may appear anywhere among inputs; keywords remain data, and `|` rejects schemas.
Owners in the edited layer take precedence; conflicting storage, duplicate schemas, argument/unrelated entries or names not bound by the call throw `ArgumentError`.
Schemas materialise with `zero(T)` at binding time; resolve symbolic sizes first. Bind whole values for storage `of` cannot describe.

**Addresses.** Bind an LHS variable, part of one, or a child's LHS variable through its namespace.
Binding-time checks reject covariates, nonexistent argument fields, out-of-storage indices and unknown top symbols, except for possible submodels.
Child namespace checks wait until reached; shared unprefixed namespaces cannot reject unknown names independently of siblings.
[`check_model`](@ref) warns about bound names that no LHS variable of the model or its reached unprefixed submodels can use.

**Values.** Whole argument bindings must satisfy declared types (`Any` if undeclared), checked at binding time, and the full signature (shared parameters/`where` constraints checked during evaluation).
Binding never selects another method. Whole bindings of local LHS variables must fit existing storage types, otherwise they set them; schemas also constrain shape.
Partial bindings convert to the replaced element/field type: `1` into `Float64` becomes `1.0`; unconvertible values propagate Julia's conversion error
(`1.5` into `Int` raises `InexactError`), while successful but lossy conversions (`0.1` into `Float32`) raise `ArgumentError`, all at binding time.
Runtime AD values need compatible storage, e.g. `fill(zero(m), n)`. An `of` type fixed before evaluation fixes its element type;
under ForwardDiff/ReverseDiff build the schema from running values (`@of(z = of(Array, typeof(m), n))`), or bind a whole value.

**Storage.** Integers, `end`, ranges, `:`, logical masks and `CartesianIndex` need an argument, schema, earlier whole value, produced `VarNamedTuple`, or prefix template.
Otherwise indexed local LHS variables infer a growable array and warn; `end`/`:` cannot be resolved. Property paths need no storage. Keyword-splat entries cannot be bound separately.

**Defaults** run once at construction: binding `x` in `f(x, n=length(x))` keeps `n`; reconstruct the model to recompute them.

## [Performance](@id binding-performance)

Partial bindings rebuild array arguments in O(length) each evaluation. Under reverse-mode AD, measured costs are about 110 ns per element;
bind whole arrays when gradients are needed. This is an indicative measurement, not a fixed cost.

## Example

Consider a linear regression model:

```@example 1
using DynamicPPL, Distributions, StableRNGs
using AbstractPPL: of, @of

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
rng = StableRNG(1)
true_m, true_c = 5.0, 3.0

x = 0:0.1:0.5
y_data = true_m .* x .+ true_c .+ randn(rng, length(x))
```

If we run the model before conditioning on `y`, we will find that all of `m`, `c`, and `y` are drawn from the prior distribution.

```@example 1
model = linear_regression(x)

# Here, `rand(rng, model)` samples from the prior distribution and returns a
# VarNamedTuple of latent LHS variables.
rand(rng, model)
```

Estimate the prior mean of `y` for a prior predictive check:

```@example 1
vnts = [rand(rng, model) for _ in 1:1000]
mean(vnt[@varname(y)] for vnt in vnts)
```

## Conditioning

To condition the model on observed data, we can use the `condition` function, or its alias `|`.
Use a `NamedTuple` or keyword arguments to bind whole top-level values.

```@example 1
observations = (; y=y_data)

# Equivalently: conditioned_model = condition(model, observations).
cond_model = model | observations
```

`conditioned` and `fixed` inspect bound values; `model.args`/`model.defaults` retain construction values.
The accessors return plain values independent of history: partial removal or mixed roles may produce `VarNamedTuple`/`PartialArray` containers.

```@example 1
conditioned(cond_model)
```

The observed `y[i]` LHS variables are no longer sampled:

```@example 1
parameters = rand(rng, cond_model)
```

`rand` still draws latent values from the prior; the observations now contribute to the likelihood:

```@example 1
loglikelihood(cond_model, parameters)
```

MCMC algorithms use this likelihood to sample the posterior.

## Fixing

Fix the intercept `c` to its true value:

```@example 1
fix_values = (; c=true_c)

fixed_model = fix(model, fix_values)
```

and sampling from the prior again:

```@example 1
parameters_fixed = rand(rng, fixed_model)
```

The prior draws of `y` now use the fixed intercept:

```@example 1
mean(vnt[@varname(y)] for vnt in [rand(rng, fixed_model) for _ in 1:1000])
```

## Supplying parameters to condition or fix on

```@example 1
condition(model, (; y=y_data))
condition(model; y=y_data)
fix(model, @varname(c) => true_c)
```

To observe only `y[1]`, use a `VarName` pair and a **binding schema** supplying the
shape and element type of the local storage:

```@example 1
cond_model_partial = condition(
    model, @varname(y[1]) => y_data[1], @of(y = of(Array, length(y_data)))
)
rand(rng, cond_model_partial)
```

The schema describes local storage without observing or fixing it; `fix` accepts the same syntax.
The equivalent functional spelling is `of((y=of(Array, length(y_data)),))`.
Arguments already supply storage, so partial argument bindings need no schema. See [Binding rules](@ref) for the complete contract.

## Missing data

`missing` and `nothing` do not mark data latent. Both throw where a tilde reads them,
including values nested in containers. Unread metadata may contain either:
`@model metadata_lhs(p) = p.a ~ Normal()` accepts `(a=1.0, b=missing)`,
but `(a=missing, b=1.0)` throws on evaluation, naming `p.a`.

The default-argument idiom `@model gdemo(x=missing)` called as `gdemo()` also throws,
even if the body first replaces `x` using `if x === missing; x = Vector{T}(undef, n); end`.
Remove the argument-supplied observation with `decondition(gdemo(), @varname(x))`:

```@example missing-data
using DynamicPPL, Distributions, StableRNGs

@model function gdemo(x=missing)
    if x === missing
        x = Vector{Float64}(undef, 2)
    end
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end

model = decondition(gdemo(), @varname(x))
model(StableRNG(1))
```

`gdemo()` stores the argument-supplied observation; `decondition` removes it so the body
can allocate and sample `x`.

For an array whose indices are separate LHS variables, decondition only the desired
indices. A single multivariate LHS variable cannot be partially conditioned:
`x ~ MvNormal(...)` rejects a value containing `missing`, naming `x`.
