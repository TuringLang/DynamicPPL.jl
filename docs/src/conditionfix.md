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

To observe only `y[1]`, use a `VarName` pair and a binding template to supply the shape and
element type of local storage:

```@example 1
cond_model_partial = condition(
    model, @varname(y[1]) => y_data[1], @of(y = of(Array, length(y_data)))
)
rand(rng, cond_model_partial)
```

`fix` accepts the same syntax. The equivalent functional spelling is `of((y=of(Array, length(y_data)),))`. Arguments already supply storage, so partial argument bindings need no
template. See [Binding rules](@ref) for the complete contract.

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

Removal matches equal, enclosing, or contained addresses. Without `Recursive()`, removal
clears only bindings stored on this model, at any address; with no names it clears that
model's corresponding layer. This includes bindings the model made at child addresses. On
a removal, `Recursive()` decides depth, that is, whether to also clear what the child holds:

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
a no-op, at the call and at evaluation, including removing the same address twice. Removals
reject the addresses bindings reject when the model can decide: unknown top symbols,
nonexistent fields, and indices outside storage. `check_model` warns about recursive removals
that no reached model uses; a local removal at a child address that names nothing is a silent no-op.

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
@model generator() = y ~ to_submodel(observation(2.0))
latent = decondition(generator(), DynamicPPL.Recursive(), @varname(y.counts))
haskey(rand(Xoshiro(1), latent), @varname(y.counts))
```

The newly latent variable is an ordinary parameter, including for `InitFromParams` and
`LogDensityFunction`; parameter order follows evaluation order.

The supported set excludes these operations. Each throws `ArgumentError` with a
supported alternative, when the operation is made if decidable, otherwise at the tilde:

  - Named recursive removal of a name shared by this model's own LHS and an unprefixed
    submodel: prefix the submodel, or remove without `Recursive()`.
  - A slice, colon or mask in an explicit or automatic prefix, such as `p[1:2]`:
    use properties and scalar integer indices (not `Bool`). `CartesianIndex` is
    expanded into integer coordinates; `begin` and `end` use the prefix template.
    A sliced return LHS remains allowed with `to_submodel(child, false)`.
  - Partial binding or removal through a tuple or struct owner, at any depth: bind or
    remove the enclosing tuple or struct whole.

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

NamedTuple fields must be addressed by name (`x.a`), never by index (including `x[1]`,
`x[:a]`, or `x[:]`), in bindings, removals and LHS variables. Ordinary Julia indexing in the
body is unaffected. Tuple observations and complete local LHS variables retain integer
indices. Bindings on prefixed models must be at or below the prefix. A **submodel namespace**
reaches child LHS variables through addresses such as `a.x`, or through unchanged names with
`auto_prefix=false` unless manually prefixed.

A submodel tilde must use a local LHS variable. If `a` is a model argument,
`a ~ to_submodel(child())`, `a[1] ~ to_submodel(child())`, and
`a.x ~ to_submodel(child())` throw `ArgumentError` when the tilde runs. Use a new local
name and condition the child's LHS variables to supply observations. Ordinary Julia assignment
can still copy the return value into an argument; a later distribution tilde observes
that argument's current value.

Explicitly binding a **submodel return value**, assigned by local `a ~ to_submodel(...)`,
also throws `ArgumentError` during evaluation. Bind `a.x` to observe the child's `x`.
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

Partial argument bindings act through supported arrays and NamedTuples.
Tuple and struct owners cannot be partially rebuilt, even inside arrays or NamedTuples.
Whole replacement, argument observations through their fields or indices, and complete local
LHS bindings (including produced `VarNamedTuple`s) remain supported. Other containers, such as dictionaries, throw
`ArgumentError` at binding time; bind or `decondition` the whole value instead.

Partial array bindings and removals (including `decondition`) rebuild only `Array` and
Array-backed `OffsetArray`, `ComponentArray` and `DimArray` storage. Each owner along the
edited path must be supported; untouched leaves are unrestricted. Views, reshapes,
`Transpose`/`Adjoint`, immutable arrays, ranges, ReverseDiff tracked arrays, `MVector`,
`SizedArray` and `BitArray` throw `ArgumentError`, so a child partially deconditioned inside a
model may work with ForwardDiff but throw with ReverseDiff when its argument is a tracked
array. Bind or decondition the whole value, or use `collect(v)` if losing axes or metadata is
acceptable. Whole operations support all array types, except as follows.

Latent draws are written into the argument, and DynamicPPL never converts an argument
implicitly. An argument held in immutable array storage, such as a range, an `SVector`, an
`SMatrix` or a `Fill`, or holding one in a NamedTuple, tuple or array, therefore cannot be made
latent. Storage is unwrapped through `parent`; `MVector`, `SizedArray`, views of mutable
arrays and SparseArrays types are accepted, while another immutable array type without a
`parent` method is rejected even if it holds mutable buffers. A `decondition` or `unfix` that
would leave such an argument latent throws `ArgumentError` when called, even when the tilde
draws the argument whole (`x ~ MvNormal(...)`); a parent's recursive removal throws when the
model is evaluated. Pass `collect(x)` instead, or `missing` when the tilde draws `x` whole.
A read-only wrapper over a mutable parent, such as `Symmetric` or `Diagonal`, and immutable
storage inside a struct field or a `Dict` are not detected and fail at evaluation. Whole
`condition` and `fix` of such an argument remain supported.

Use NamedTuples or keyword arguments for whole top-level values, and `VarName` pairs for any
address. The pair `:x => v` abbreviates `@varname(x) => v`. `VarNamedTuple`s produced by
DynamicPPL are also accepted. Positional inputs and tuples apply left to right.
`condition` and `fix` reject every `AbstractDict` and other unsupported input with
`ArgumentError`; `model | dict` throws `MethodError`. Use a NamedTuple or ordered pairs.

A **binding template** is an AbstractPPL `OfNamedTuple` type, such as `@of(z = of(Array, 3))`. It
supplies storage for partially bound local LHS variables without binding them. A binding
template shapes the binding, not the model's storage: the body still sets each local's size, and
bound indices outside it are unused. Import `of, @of`
from AbstractPPL. One binding template may appear anywhere among positional inputs. Keywords
remain binding data, and `|` rejects binding templates. Owners in the edited layer take
precedence. Conflicting storage or duplicate binding templates throw `ArgumentError`. Entries for
arguments and names not bound by the call are ignored. Templates use absolute names, as in
`rand(model)` output: for `prefix(m, @varname(p))`, use `@of(p = @of(z = of(Array, 3)))`.
Deferred entries stay with bindings in the edited layer and follow `prefix`. Removing the
last binding beneath an entry drops it; partial removals retain it for surviving bindings.
A later whole binding supplies the new storage. Across submodels, outer bindings and template
entries take precedence; recursive removals also clear child entries with their bindings.

Binding templates are converted with `zero(T)` at binding time, so resolve symbolic sizes first. Bind
whole values for storage that `of` cannot describe.

A sample can supply a binding template with `of(rand(rng, model))`. This describes storage,
not observations: array values and partial masks are discarded, while shape and element type
are retained. It is a snapshot; rebuild it when the sample layout changes. Plain numeric
arrays, exactly representable scalars and nested namespaces are supported. Custom arrays,
growable entries and scalars whose type `of` would widen are rejected.

An indexed prefix, such as `prefix(model, @varname(p[1, 2]))`, cannot take a template: the
template grammar cannot express indexed namespaces such as `p[1, 2].y`, so `of(rand(rng, model))`
throws for such samples. Supply storage before applying the indexed prefix, or through an
argument.

```jldoctest binding_template
julia> using DynamicPPL, Distributions; using AbstractPPL: of; using StableRNGs: StableRNG

julia> @model function template_demo()
           y = zeros(3)
           for i in eachindex(y)
               y[i] ~ Normal()
           end
           return y
       end;

julia> m = template_demo(); template = of(rand(StableRNG(1), m));

julia> condition(m, @varname(y[1]) => 0.5, template)(StableRNG(2))[1]
0.5
```

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
must fit existing storage types, or set them if absent. Binding templates also constrain shape.

Templates for a child's variables are applied when that child runs, so its arguments keep
their own storage. At the parent, use concrete indices or establish a whole binding first;
`end` and `:` cannot be resolved from a template before the child is known. To use them,
bind the child directly before passing it to `to_submodel`.

Partial bindings convert values to the replaced element or field type at binding time
(or when the child runs for a binding made through its parent). For
example, `1` becomes `1.0` in `Float64`. Unconvertible values propagate Julia's conversion
error, such as `InexactError` for `1.5` into `Int`. Successful but lossy conversions, such as
`0.1` into `Float32`, raise `ArgumentError`. Runtime AD values need compatible storage, such as
`fill(zero(m), n)`. An `of` type fixed before evaluation fixes its element type. Under
ForwardDiff or ReverseDiff, build the binding template from running values (`@of(z = of(Array, typeof(m), n))`), or bind a whole value.

Integer indices, `end`, ranges, `:`, logical masks, and `CartesianIndex` need an argument,
binding template, earlier whole value, produced `VarNamedTuple`, or prefix template. Otherwise,
indexed local LHS variables infer a growable array and warn, and `end` or `:` cannot be
resolved. Property paths need no storage. An indexed prefix has the template limitation described
above; establish its storage before applying the prefix, or supply it through an argument.
Keyword-splat entries cannot be bound separately.

With growable storage, every tilde for a symbol must use the same number of indices as its
bindings; otherwise evaluation throws `ArgumentError`. Linear and Cartesian addresses are
distinct: binding `x[2]` does not bind `x[2, 1]`. This also applies when the model itself mixes
index counts on one local symbol, such as `x = zeros(2, 2); x[1] ~ Normal(); x[2, 2] ~ Normal()`.
To bind such a symbol by parts, supply storage with a binding schema (`@of(x = of(Array, 2, 2))`)
or a model argument. Allocating the local array inside the model does not supply binding
storage. With storage, bindings follow Julia's indexing semantics.

Defaults are evaluated once at construction. Binding `x` in `f(x, n=length(x))` keeps `n`.
Construct the model again to recompute defaults.

## [Performance](@id binding-performance)

Partial bindings rebuild array arguments in O(length) per evaluation. Measured reverse-mode AD
costs (about 110 ns per element) are indicative, not fixed. Bind whole arrays for gradients.

## Missing data

An argument on the LHS supplies no observation in two cases, decided when the model is
constructed. A whole `missing` or `nothing` argument, whether positional, keyword or default,
makes its LHS variables latent, exactly as `decondition(model, @varname(x))` does. A `missing`
element `x[i]` or `x[i, j]` of a top-level argument array leaves that element latent, exactly
as deconditioning it does. The array's element type must admit `Missing`, and it must be one
that partial edits support (see [Binding contract](@ref)); other arrays holding `missing`, such
as views, throw at construction, and `collect(v)` converts them. Arguments that do not occur
on the LHS are never inspected.

Later changes to the argument do not change these roles. An array holding `missing` is
snapshotted at construction, so mutating it afterwards has no effect, whereas a fully
observed array is used in place.

Every other `missing` or `nothing` throws `ArgumentError` where a tilde reads it, naming the
LHS variable: `nothing` elements, `missing` deeper down (`x[i][j]`), and placeholders in
tuples, NamedTuple fields or struct fields. Unread parts may hold either: `@model
metadata_lhs(p) = p.a ~ Normal()` accepts `(a=1.0, b=missing)`, but `(a=missing, b=1.0)`
throws on evaluation, naming `p.a`. A multivariate LHS variable such as `x ~ MvNormal(...)` is
latent when all its elements are `missing`, observed when none are, and throws when it has
both.

Bindings never hold placeholders: a `condition` or `fix` value containing `missing` or
`nothing` anywhere throws when the binding is made. Use `decondition` or `unfix` instead. In
consequence, `conditioned(f((a=1.0, b=missing)))` lists the placeholder, and passing that
listing back to `condition` throws. Incomplete partial bindings into whole `missing`/`nothing`
arguments also throw `ArgumentError` when bound; supply a concrete argument such as
`f(zeros(n))` or a whole binding. A produced `VarNamedTuple` array entry whose mask is
complete binds as a whole value and supplies its own storage.

The body may replace a placeholder argument; each tilde then writes its draw into the new
storage:

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

model = gdemo()
model(StableRNG(1))
```

Complete sampled arrays can then be bound back to the placeholder model:

```@example missing-data
values = rand(StableRNG(1), model)
condition(gdemo(), values)(StableRNG(2)) == values[@varname(x)]
```

`missing` elements leave only those elements latent:

```@example missing-data
keys(rand(StableRNG(1), gdemo(Union{Missing,Float64}[1.0, missing])))
```
