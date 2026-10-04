# Conditioning and fixing

Use `condition` to supply observations and `fix` to hold values constant without adding a
log-probability contribution. Both operations let you reuse a model definition with
different data.

## Linear regression example

Consider the following linear regression model:

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

This model has no observed data or conditioned bindings, so all its `y[i]` LHS variables are
latent. Defining `y` provides storage for `y[i]` after its tilde runs. Without it, `setindex!`
would throw on an undefined variable. Local storage supplies no observations, so its LHS
variables remain latent until conditioned or fixed.

Create synthetic data for this model:

```@example 1
rng = StableRNG(1)
true_m, true_c = 5.0, 3.0

x = 0:0.1:0.5
y_data = true_m .* x .+ true_c .+ randn(rng, length(x))
```

Before conditioning on `y`, the model draws `m`, `c`, and `y` from the prior distribution:

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

### Conditioning

Use `condition` or its alias `|` to condition the model on observed data:

```@example 1
observations = (; y=y_data)

# Equivalently: conditioned_model = condition(model, observations).
cond_model = model | observations
```

`conditioned` and `fixed` inspect bound values, while `model.args` and `model.defaults` retain
construction values. The accessors return plain values independent of history. Partial removal
or mixed roles may produce `VarNamedTuple` or `PartialArray` containers.

```@example 1
conditioned(cond_model)
```

The observed `y[i]` LHS variables are no longer sampled:

```@example 1
parameters = rand(rng, cond_model)
```

`rand` still draws from the prior. Observations now contribute to the likelihood:

```@example 1
loglikelihood(cond_model, parameters)
```

MCMC algorithms use this likelihood to sample the posterior.

### Fixing

Fix the intercept `c` to its true value:

```@example 1
fix_values = (; c=true_c)

fixed_model = fix(model, fix_values)
```

Then sample from the prior again:

```@example 1
parameters_fixed = rand(rng, fixed_model)
```

The prior draws of `y` now use the fixed intercept:

```@example 1
mean(vnt[@varname(y)] for vnt in [rand(rng, fixed_model) for _ in 1:1000])
```

### Supplying bound values

```@example 1
condition(model, (; y=y_data))
condition(model; y=y_data)
fix(model, @varname(c) => true_c)
```

To observe only `y[1]`, use a `VarName` pair and a binding schema to supply the shape and
element type of local storage:

```@example 1
cond_model_partial = condition(
    model, @varname(y[1]) => y_data[1], @of(y = of(Array, length(y_data)))
)
rand(rng, cond_model_partial)
```

`fix` accepts the same syntax. The equivalent functional spelling is `of((y=of(Array, length(y_data)),))`. Arguments already supply storage, so partial argument bindings need no
schema. See [Binding rules](@ref) for the complete contract.

## Binding rules

`condition` and `fix` store bindings on the called model at any address in its namespace,
including child addresses: `condition(parent, @varname(a.x) => 1.0)` binds the child's `x`.
`|` follows the same rule. Explicit bindings take precedence from the outermost model inward;
argument-supplied observations never bind a child's LHS variables.
A binding at a name shared with an unprefixed child binds both:

```@example shared-binding
using DynamicPPL, Distributions, Random
@model leaf(x) = x ~ Normal()
@model function shared(x, child)
    x ~ Normal()
    a ~ to_submodel(child, false)
    return (x, a)
end
condition(shared(2.0, leaf(3.0)), @varname(x) => 1.0)(Xoshiro(1)) # (1.0, 1.0)
```

Produced values round-trip with prefixed and unprefixed children:
`condition(model, rand(rng, model))` observes the sampled LHS variables.

An **LHS variable** is the addressed left-hand side of one execution of `~`. An **LHS
subvariable** is part of it, such as `x[1]` of `x ~ MvNormal(...)`. An LHS variable's **role**
is latent (initialised), observed (scored in the likelihood), or fixed (no log probability). A
**binding** supplies a bound value and an observed or fixed role. `condition` and `fix` make
**explicit bindings**, while arguments on the LHS supply **argument-supplied observations**.

The **effective binding** determines an LHS variable's role and value. Enclosing models'
explicit bindings take precedence, outermost first. This model's fixed binding, explicit
observation, and argument-supplied observation follow in that order. Without a binding, the LHS
variable is latent. A parent's argument-supplied observations never reach a child.

Observations form the lower layer, with fixed bindings above them. Within a layer, later
bindings replace earlier ones where they overlap.

| Operation     | Effect on this model's bindings                                                                             |
|:------------- |:----------------------------------------------------------------------------------------------------------- |
| `condition`   | Adds observations below fixed bindings.                                                                     |
| `fix`         | Adds fixed bindings, shadowing observations.                                                                |
| `decondition` | Removes observations of either origin, even under a fixed binding; otherwise makes the LHS variable latent. |
| `unfix`       | Removes fixed bindings, uncovering the observation below, or leaving the LHS variable latent.               |

Removal matches equal, enclosing, or contained addresses, with Symbol indices matching
properties. Without `Recursive()`, removal clears only bindings stored on this model, at any
address; with no names it clears that model's corresponding layer. This includes bindings the
model made at child addresses. On a removal, `Recursive()` decides depth, that is, whether
to also clear what the child holds:

```julia
@model leaf(x=2.0) = x ~ Normal()
@model outer() = a ~ to_submodel(leaf())   # the child observes x = 2.0
m = condition(outer(), @varname(a.x) => 3.0)
decondition(m, @varname(a.x))     # the child's x = 2.0 applies again
decondition(m, DynamicPPL.Recursive(), @varname(a.x))  # a.x is latent
```

The child's value returns because bindings at one address resolve outermost first: the
parent's 3.0 only shadowed the child's 2.0, so removing it uncovers the next binding inward.

With `DynamicPPL.Recursive()`, `decondition` removes observations of either origin at every
depth, leaving fixed bindings; `unfix` removes fixed bindings, uncovering observations below.
The no-name forms clear their layer throughout. Removing a valid address with no binding is
a no-op, at the call and at evaluation, including removing the same address twice. Unknown
own names throw when the model can decide. `check_model` warns about recursive removals that
no reached model uses; a local removal at a child address that names nothing is a silent no-op.

A removal belongs to the model that makes it. It reaches enclosed models, including those
built or bound in the body, but cannot remove an enclosing model's binding. Prefixing moves
the removal with the model. A later binding in the same layer replaces it at that address.
The removal holds no value or shape: the next binding or argument supplies storage.
`conditioned` and `fixed` list stored values only and take no `Recursive()` argument.

For example, a parent can make a child's observation latent even when it constructs the
child inside its body:

```@example recursive-removal
using DynamicPPL, Distributions, Random
@model observation(counts) = counts ~ Normal()
@model generator() = y ~ to_submodel(observation(missing))
latent = decondition(generator(), DynamicPPL.Recursive(), @varname(y.counts))
haskey(rand(Xoshiro(1), latent), @varname(y.counts))
```

The newly latent variable is an ordinary parameter, including for `InitFromParams` and
`LogDensityFunction`; parameter order follows evaluation order.

The supported set excludes these operations. Each throws `ArgumentError` naming the
address and a supported alternative, when the operation is made if decidable, otherwise
at the tilde:

  - Named recursive removal of a name shared by this model's own LHS and an unprefixed
    submodel: prefix the submodel, or remove without `Recursive()`.
  - Partial binding or removal below an argument that any branch uses as a submodel return
    value or namespace: bind or remove the whole argument instead. Whole bindings still
    cannot explicitly bind a reached submodel return value.
  - Binding or removal below a slice or colon prefix, such as `p[1:2].x`: edit before
    prefixing, or use an integer-indexed prefix.
  - Partial binding through a tuple nested inside an array or struct: bind the enclosing
    element whole. Tuples nested only in tuples or NamedTuples remain supported.

Whole bindings and removals remain subject to the usual rules.

### Argument contract and shared constraints

Subvariables of one LHS variable must have the same role. Mixing roles throws `ArgumentError`
during evaluation. A bound **local LHS variable**, whose top symbol does not name an argument,
reads its effective binding's value at its tilde.

For an **argument LHS variable**, whose top symbol names an argument, every binding replaces the
argument before the body runs. Its tilde reads the argument's current value. A fixed argument
LHS variable is also reset to its bound value at its tilde. Thus `f(data)` and
`condition(f(...); x=data)` observe the same value even if the body transforms `x` first. To
observe raw data, use a separate name. Deconditioned arguments keep their old values until their
tilde. See [`Model`](@ref) for direct construction and handwritten evaluator obligations.

The two rules differ when the body writes the variable before its tilde:

```@example 1
@model observe_argument(x) = (x = 3.0; x ~ Normal(); x)
@model observe_local() = (x = 3.0; x ~ Normal(); x)
condition(observe_argument(0.0); x=1.0)(), condition(observe_local(); x=1.0)()
```

The conditioned argument observes `3.0`, the value the body leaves in it, as
`observe_argument(1.0)` does; the conditioned local LHS variable observes the bound `1.0`.

A binding owns the shape of the value at its address, at any depth, within the layer being
edited. Partial bindings and removals use the latest owner: binding `x=ones(3)` then removing
`x[3]` preserves length three. Binding `x[1]` or `p.a` may resize that value if its type
permits, but cannot resize a container above it. A binding of a slice, range, `:`, or mask must
match the extent it addresses. Removing a binding returns its address to the next owner: the
shadowed binding, enclosing binding, or argument.

The body sets the shape of local storage and may resize conditioned arguments. Fixed values must
retain their size and shape and cover every reached LHS variable below their address. Fixed
argument tildes reject growth, shrinkage, or reshaping with an `ArgumentError` naming the LHS
variable. Shape validation stops at the LHS variable's address; it does not inspect nested
values inside a whole structured LHS variable. In-place mutation of a whole fixed argument
is not detected, because it also mutates the stored binding; the body must not mutate it.

NamedTuple fields must be addressed by name (`x.a`), never by integer index, in both bindings
and LHS variables. Tuples retain integer indices. Bindings on prefixed models must be at or
below the prefix. A **submodel namespace** reaches child LHS variables through addresses such as
`a.x`, or through unchanged names with `auto_prefix=false` unless manually prefixed.

Explicitly binding a **submodel return value**, assigned by `a ~ to_submodel(...)`, throws
`ArgumentError` during evaluation. If `a` is an argument, its observation is ignored at this
tilde. When `a` is a model argument, explicit bindings at or below `a` also throw, possibly
at binding time if its type excludes the requested field. When `a` is local, `a.x` can bind
the child's `x`. A NamedTuple argument provides no submodel namespace, so bind the child
before `to_submodel`.
Bindings unused by reached LHS variables are ignored, including branches or submodels not run.

Bound values are not copied, so the body must not mutate them, even through a `view`. Partial
bindings take a shallow snapshot when made: the owner's container is copied one level deep,
but nested mutable values remain shared and must not be mutated either. For example, in a
model that observes each scalar of `x = [[1., 2.], [3., 4.]]`, binding `@varname(x[1][1]) => 9.`
does not isolate `x[2]`. Subsequently setting `x[2][1] = 99.` makes evaluation return
`[[9.0, 2.0], [99.0, 4.0]]` if the model returns `x`.

**Runtime bindings** are made inside the running body, as in `a ~ to_submodel(condition(child(y), @varname(y[1]) => m))`. Remade at each evaluation, their values
carry derivatives with respect to enclosing latent variables. Bindings made beforehand hold
values constant with respect to model parameters.

See [Missing data](@ref) for making observations latent. `InitFromParams` rejects `missing`
when read, not at construction. Omit unobserved values instead.

### Binding contract

Partial argument bindings act through arrays, tuples, NamedTuples and plain struct fields
(properties must match fields). Other containers, such as dictionaries, throw
`ArgumentError` at binding time; bind or `decondition` the whole value instead.

Use NamedTuples or keyword arguments for whole top-level values, and `VarName` pairs for any
address. The pair `:x => v` abbreviates `@varname(x) => v`. `VarNamedTuple`s produced by
DynamicPPL are also accepted. However, `conditioned(m)` includes argument-supplied values
ignored at submodel-return tildes; remove those entries before passing the listing to
`condition`, since explicit bindings of return values are rejected. Positional inputs and tuples apply left to right. Every
`AbstractDict` and other unsupported input throws `ArgumentError`.

A **binding schema** is an AbstractPPL `OfNamedTuple` type, such as `@of(z = of(Array, 3))`. It
supplies storage for partially bound local LHS variables without binding them. A binding
schema shapes the binding, not the model's storage: the body still sets each local's size, and
bound indices outside it are unused. Import `of, @of`
from AbstractPPL. One binding schema may appear anywhere among positional inputs. Keywords
remain binding data, and `|` rejects binding schemas. Owners in the edited layer take
precedence. Conflicting storage or duplicate binding schemas throw `ArgumentError`. Entries for
arguments, unrelated names, or names not bound by the call also throw `ArgumentError`. Binding
schemas are converted with `zero(T)` at binding time, so resolve symbolic sizes first. Bind
whole values for storage that `of` cannot describe.

Bindings must address LHS variables, parts of them, or child LHS variables through a
submodel namespace. Binding-time checks reject covariates, nonexistent argument fields,
indices outside storage, and unknown top symbols. Only a literal `to_submodel(child, false)`
tilde allows unknown top symbols, since its unprefixed child may own them. Other right-hand
sides, including `truncated(...)` and `filldist(...)`, do not relax this check. Child
namespace checks wait until the child is reached. Shared unprefixed namespaces cannot reject
unknown names independently of siblings. [`check_model`](@ref) warns about bound names that
no LHS variable in the model or its reached unprefixed submodels can use.

Whole argument bindings must satisfy declared types (`Any` if undeclared) at binding time. The
full signature must also hold, with shared type parameters and `where` constraints checked
during evaluation. Binding never selects another method. Whole bindings of local LHS variables
must fit existing storage types, or set them if absent. Binding schemas also constrain shape.

Partial bindings convert values to the replaced element or field type at binding time. For
example, `1` becomes `1.0` in `Float64`. Unconvertible values propagate Julia's conversion
error, such as `InexactError` for `1.5` into `Int`. Successful but lossy conversions, such as
`0.1` into `Float32`, raise `ArgumentError`. Runtime AD values need compatible storage, such as
`fill(zero(m), n)`. An `of` type fixed before evaluation fixes its element type. Under
ForwardDiff or ReverseDiff, build the binding schema from running values (`@of(z = of(Array, typeof(m), n))`), or bind a whole value.

Integer indices, `end`, ranges, `:`, logical masks, and `CartesianIndex` need an argument,
binding schema, earlier whole value, produced `VarNamedTuple`, or prefix template. Otherwise,
indexed local LHS variables infer a growable array and warn, and `end` or `:` cannot be
resolved. Property paths need no storage. Keyword-splat entries cannot be bound separately.

Defaults are evaluated once at construction. Binding `x` in `f(x, n=length(x))` keeps `n`.
Construct the model again to recompute defaults.

## [Performance](@id binding-performance)

Partial bindings rebuild array arguments in O(length) per evaluation. Measured reverse-mode AD
costs (about 110 ns per element) are indicative, not fixed. Bind whole arrays for gradients.

## Missing data

`missing` and `nothing` do not mark data latent. Whole `missing` or `nothing` arguments throw
`ArgumentError` at the reading tilde, even if replaced in the body. Bound values containing
either also throw there. Both errors name the LHS variable. Unread parts may contain either. For
example, `@model metadata_lhs(p) = p.a ~ Normal()` accepts `(a=1.0, b=missing)`, but
`(a=missing, b=1.0)` throws on evaluation, naming `p.a`.
Partial bindings into whole `missing`/`nothing` arguments, even after deconditioning, throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.

The default-argument idiom `@model gdemo(x=missing)` called as `gdemo()` also throws, even if
the body first replaces `x` using `if x === missing; x = Vector{T}(undef, n); end`. Remove the
stored argument-supplied observation with `decondition(gdemo(), @varname(x))` so the body can
allocate and sample `x`:

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

For an array whose indices are separate LHS variables, decondition only the desired indices. A
single multivariate LHS variable cannot be partially conditioned: `x ~ MvNormal(...)` rejects a
value containing `missing`, naming `x`.
