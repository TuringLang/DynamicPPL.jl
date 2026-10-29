# Conditioning and fixing

`condition` stores observations on a model; their distributions contribute to the
likelihood. `fix` stores values whose distributions contribute no density. Both
return a model with the requested bindings. `rand` draws its remaining latent
variables from their prior distributions, not from the posterior.

```@example bindings
using DynamicPPL, Distributions, Random
using DynamicPPL: @vnt
using DynamicPPL.AbstractPPL: @of, of

@model function linear_regression(x)
    m ~ Normal()
    c ~ Normal()
    y = zeros(length(x))
    for i in eachindex(x)
        y[i] ~ Normal(m * x[i] + c, 1)
    end
    return y
end

x = [0.0, 0.5, 1.0]
y_data = [3.0, 5.5, 8.0]
model = linear_regression(x)
observed = condition(model; y=y_data)
parameters = rand(Xoshiro(1), observed)
loglikelihood(observed, parameters)
```

The local allocation of `y` provides a destination for each tilde assignment. It
does not make `y` observed. A model argument that appears on a tilde's left-hand
side supplies observations when the model is called directly; submodels use
explicit bindings instead.

```@example bindings
fixed_model = fix(observed; c=3.0)
@assert fixed(fixed_model)[@varname(c)] == 3.0
@assert conditioned(fixed_model)[@varname(y)] == y_data
rand(Xoshiro(1), fixed_model)
```

## Binding inputs and storage

`condition` and `fix` accept NamedTuples, keyword arguments, Symbol or `VarName`
pairs, tuples of these inputs, and `VarNamedTuple`s. `model | values` abbreviates
`condition(model, values)`. Dictionaries are not binding inputs.

```@example bindings
@assert conditioned(model | (; y=y_data)) == conditioned(observed)
@assert conditioned(model | (@varname(y) => y_data)) == conditioned(observed)
@assert conditioned(condition(model, :y => y_data)) == conditioned(observed)
@assert conditioned(condition(model, ((y=y_data,), :c => 3.0)))[@varname(c)] == 3.0
```

An indexed partial address needs storage when the binding is made. Storage can
come from the called model's arguments, an existing whole binding, a storage
carrying `VarNamedTuple`, or an explicit template. A future allocation in a model
body does not supply storage to the binding call.

Pass at most one `@of`/`of` template, as the last positional argument. A template
elsewhere throws `ArgumentError`. Template names are absolute, including any
model or child prefix.

```@example bindings
partial = condition(model, @varname(y[1]) => y_data[1], @of(y = of(Array, 3)))
@assert partial(Xoshiro(1))[1] == y_data[1]

observations = @vnt begin
    @template y = y_data
    y[1] := y_data[1]
end
@assert (model | observations)(Xoshiro(1))[1] == y_data[1]
```

A template describes storage, not observations. Unbound positions remain latent,
and unused template entries do not appear in `conditioned` or `fixed`. Templates
replace a `missing` model argument, but concrete arguments keep their own storage.
Array extents, selection shapes and aliases are checked at binding time; this
includes `begin`, `end`, masks and linear versus Cartesian indexing.

Typed templates also constrain conversion. Values must be exactly representable
in the specified type; an inexact conversion throws. Use the runtime value's type
when constructing templates inside differentiated code.

```@example bindings
half = condition(model, @varname(y[1]) => 0.5, @of(y = of(Array, Float32, 3)))
@assert conditioned(half)[@varname(y[1])] === 0.5f0
```

A run records only the latent sites it reached, omitting observed and fixed sites
and untaken branches. A subset of that `VarNamedTuple` retains storage for its
selected values. This supports blockwise conditioning without reconstructing
bindings from bare pairs.

```@example bindings
run_values = rand(Xoshiro(2), model)
block = subset(run_values, [@varname(y[1])])
blocked = condition(model, block)
@assert blocked(Xoshiro(1))[1] == run_values[@varname(y[1])]
storage_only = condition(model, @varname(y[1]) => 0.5, of(block))
@assert storage_only(Xoshiro(1))[1] == 0.5
```

Rebuild such storage snapshots when parameters change the model's layout.
`of(vnt)` supports plain numeric arrays, supported real scalars and nested named
fields. It cannot preserve scalar `Bool` values or indexed namespaces; supply
explicit storage for those cases. Pass raw model-space values to binding
APIs, not linked `TransformedValue`s or a `VarInfo`.

## Binding rules

Each executed tilde has one role: latent, observed or fixed. Separate scalar
tildes can have different roles. A single multivariate tilde cannot be partly
observed or fixed while its remaining positions are latent.

A later binding in the same role replaces the selected values. Overlap between
roles throws; remove the old role before assigning the new one. Overlapping
addresses within a single binding call also throw.

```@example bindings
replaced = condition(observed; y=[1.0, 2.0, 3.0])
@assert replaced(Xoshiro(1)) == [1.0, 2.0, 3.0]
changed_role = fix(decondition(observed, @varname(y)); y=y_data)
@assert changed_role(Xoshiro(1)) == y_data
```

Bindings must address declared tilde variables or child namespaces. Covariates
cannot be bound. Supported partial owners are arrays and NamedTuples with named
fields; arbitrary struct, tuple and dictionary reconstruction is not supported.
Bind or remove the whole value when partial edits are unsupported.

Argument bindings are prepared before the model body runs. Body computations
therefore see replaced arguments. An observed argument tilde uses the value
computed by the body; a fixed argument tilde restores its bound value. Local
variables obtain their bindings at the tilde itself. Replacement arguments must
satisfy declared argument types.

Evaluation copies argument storage that can receive latent writes, preserving
the caller's data. Shared backing storage between latent arguments is rejected;
covariates and fully bound branches have different copy requirements. Immutable
backing storage, unsupported dictionary keys in copied graphs, and custom numeric
leaves containing mutable storage can also be rejected. AD extensions handle
their supported tracer types. Use ordinary mutable arrays for latent writes;
copying is not a promise to reconstruct every custom container.

A handwritten `Model` evaluator must perform the argument preparation and role
lookup that `@model` generates. Calling `tilde_observe!!` directly does not look up
model bindings. Prefer `@model`; `args_on_lhs` metadata alone does not implement
binding semantics in a handwritten evaluator.

## Submodels

Every submodel contributes its tilde address. In `a ~ to_submodel(child)`, a child
variable `x` has address `a.x`. An explicit child prefix composes with that address.
The tilde receives the child's return value; that return is not itself a bindable
random variable.

```@example bindings
@model inner(y, x) = (y ~ Normal(x, 1); y)
@model outer(y, x) = a ~ to_submodel(condition(inner(missing, x), (y=y,)))
@assert outer(1.0, 0.0)(Xoshiro(1)) == 1.0
```

Submodels cannot receive observations as model arguments. Pass covariates as
arguments and use `condition` for observations. `fix` would omit their likelihood.
For partly missing arrays, bind only the observed entries and supply a template
as the last positional argument. It replaces the `missing` argument with storage
of the specified shape, so the model can iterate over it; unbound entries remain
latent. The same storage rule applies to `fix`, which omits the fixed entries'
densities.

```@example bindings
@model function inner_vec(y)
    for i in eachindex(y)
        y[i] ~ Normal()
    end
    return y
end
@model function outer_vec(y)
    dims = size(y)
    child = condition(
        inner_vec(missing),
        (@varname(y[i]) => y[i] for i in eachindex(y) if !ismissing(y[i]))...,
        @of(y = of(Array, dims...)),
    )
    return a ~ to_submodel(child)
end
partial_child = outer_vec([1.0, missing])
@assert partial_child(Xoshiro(1)) == [1.0, rand(Xoshiro(1), Normal())]
@assert loglikelihood(partial_child, (; a=(; y=[1.0, 0.4]))) == logpdf(Normal(), 1.0)
```

Parent bindings cannot overwrite a child's own bindings, even in the same role;
remove those bindings first. Child ownership and storage compatibility are
checked when the child is reached, without executing it at bind time.

Indexed addresses need storage for the namespace as well as any indexed child
variable. Plain property paths and whole values need no index storage.

```@example bindings
@model scalar_child() = (x ~ Normal(); x)
@model function indexed_parent()
    z = zeros(3)
    for i in eachindex(z)
        z[i] ~ to_submodel(scalar_child())
    end
    return z
end
m = indexed_parent()
bound_child = condition(m, @varname(z[1].x) => 1.0, @of(z = of(Array, 3)))
@assert bound_child(Xoshiro(1))[1] == 1.0
```

A whole NamedTuple below an unreached child can denote either a whole variable or
a child namespace. Removing a field, or changing its field set through whole or
field writes, requires storage that distinguishes these structures. Supply that
storage with the original binding, or remove the whole bound root before
replacing it. An array's shape alone does not establish the surrounding field
structure. Removal functions do not take a positional template.

A `VarNamedTuple` can distinguish the namespace from its whole NamedTuple value:

```@example bindings
@model fields_child() = (x ~ Normal(); y ~ Normal(); (x=x, y=y))
@model fields_middle() = b ~ to_submodel(fields_child())
@model fields_parent() = a ~ to_submodel(fields_middle())
structured = VarNamedTuple((a=VarNamedTuple((b=(x=1.0, y=2.0),)),))
fields_bound = condition(fields_parent(), structured)
fields_removed = decondition(fields_bound, @varname(a.b.x))
@assert !haskey(conditioned(fields_removed), @varname(a.b.x))
@assert fields_removed(Xoshiro(1)).y == 2.0
```

Submodel tilde addresses accept properties and scalar non-Boolean integer indices,
including scalar `CartesianIndex` coordinates. They do not accept slices or masks,
and cannot be rooted in an argument. Assign a child's return to an ordinary local
name, then use ordinary Julia assignment to copy that result into a slice or an
argument container.

## Removing bindings

`decondition(model, names...)` and `unfix(model, names...)` remove the selected role
from the model and its submodels. With no names, they remove all bindings in that
role. These instructions also apply to children constructed during evaluation.
Removing a whole name covers its descendants; removing one indexed position does
not remove its siblings. Removing a valid absent address is a no-op on values.

```@example bindings
latent_child = decondition(outer(1.0, 0.0), @varname(a.y))
@assert haskey(rand(Xoshiro(1), latent_child), @varname(a.y))
@assert isempty(conditioned(decondition(observed)))
@assert isempty(fixed(unfix(changed_role)))
```

Removal does not restore an overwritten observation. Partial removal retains the
storage needed to interpret remaining edits. Whole removal releases the role's
bound data and conversion constraint; argument storage remains available where
applicable. Observation and fix roles share a root's shape but can retain distinct
conversion types.

`conditioned` and `fixed` list bindings held by the queried model, including its
own argument observations. They do not execute the model to discover bindings
created inside child bodies. A listing contains values and their storage, not the
persistent removal instructions; rebinding a listing need not reproduce a model
that also has such instructions.

Models compare and hash by binding state. `==` delegates to `isequal`; arguments
and defaults compare by content. Binding roles, storage constraints and removal
instructions can distinguish models with the same current listing.

## Missing data

A whole `missing` or `nothing` argument supplies no observation. A one-level
`missing` entry in a supported argument array is latent. This constructor behaviour
does not permit placeholders in explicit `condition` or `fix` inputs: those throw.

```@example bindings
@model function measurements(y)
    for i in eachindex(y)
        y[i] ~ Normal()
    end
    return y
end
imputation = measurements(Union{Missing,Float64}[missing, 2.0])
@assert conditioned(imputation)[@varname(y[2])] == 2.0
@assert haskey(rand(Xoshiro(1), imputation), @varname(y[1]))
```

Each imputed entry must have its own tilde. A partly observed array cannot supply
one `MvNormal` tilde. Nested placeholders in explicit bindings are rejected, and
executed observed or fixed values are checked again. A partly bound whole-placeholder
argument is not sufficient storage for evaluating the argument.

## Diagnostics

`check_model` reports unused bindings, template entries and named removals for the
execution it checks. Dormant bindings do not make ordinary evaluation fail merely
because their branch was not reached. A warning does not prove that the address is
invalid on every branch; an executed invalid address still throws.

```@example bindings
@model function optional_child(run)
    q ~ Normal()
    if run
        a ~ to_submodel(scalar_child())
    end
    return q
end
dormant = condition(optional_child(false), @varname(a.x) => 1.0)
DynamicPPL.DebugUtils.check_model(Xoshiro(1), dormant)
```

## Migration to model-owned bindings

Bindings now live on `Model`, rather than in conditioning or fixing contexts.
Use `condition`, `fix`, `decondition`, `unfix`, `conditioned` and `fixed` instead of
constructing or inspecting the removed `CondFixContext` representation. Direct
evaluators must follow the preparation and lookup rules described above.

  - Import `@vnt` explicitly with `using DynamicPPL: @vnt`; it is no longer exported.
  - Replace `model | Dict(@varname(x) => value)` with
    `condition(model, @varname(x) => value)`. For an indexed partial address, supply
    storage too. Dictionaries remain accepted by parameter initialisation APIs such
    as `InitFromParams`; this change concerns bindings only.
  - Replace a storage-free partial binding such as
    `condition(m, @varname(z[1].x) => 1.0)` with
    `condition(m, @varname(z[1].x) => 1.0, @of(z = of(Array, 3)))` when the namespace
    has length three. A partial `@vnt` without concrete storage is insufficient.
  - Move an `@of`/`of` binding template to the last positional argument. For example,
    use `condition(m, @varname(z[1].x) => 1.0, template)`.
  - Replace `unused ~ to_submodel(prefix(child, @varname(a)), false)` with
    `a ~ to_submodel(child)`. Unprefixed submodels no longer exist.
    `to_submodel(child, true)` remains accepted; `false` throws with migration advice.
    Update parameter, binding, chain and Gibbs block addresses to include the child
    name. `a ~ to_submodel(prefix(child, @varname(b)))` uses `a.b.x`, not `b.x`.
  - Replace `to_submodel(inner(y, x))` with
    `to_submodel(condition(inner(missing, x), (y=y,)))` when `y` is fully observed.
    For partly missing arrays, bind only observed indices and provide storage:
    `condition(inner_vec(missing), @varname(y[1]) => 1.0, @of(y = of(Array, 2)))`.
    The runnable `outer_vec([1.0, missing])` example in [Submodels](@ref) retains
    `y[2]` as latent. Top-level argument observations remain supported.
  - Existing bare and named `decondition`/`unfix` calls now reach submodels too.
    Remove the old explicit recursion marker from calls; there is no local-only
    public removal mode. To edit a child in isolation, edit that child before
    composing the parent.
  - For field removal or a field-set-changing rewrite below an unreached whole
    NamedTuple binding, supply structural storage with the binding, or remove the
    whole root and bind its replacement.
  - Replace explicit bindings containing `missing` with bindings only for observed
    positions and their storage. For argument-array imputation, use the constructor
    form in [Missing data](@ref).
  - `keys`, `values`, `haskey`, `length` and `isempty` on a `VarInfo` now require a
    value-recording accumulator. Replace queries on `VarInfo()` with queries on
    `VarInfo(VectorValueAccumulator())` after evaluation, or use `rand(rng, model)`
    when raw named parameter values are required.

For blockwise inference, retain `condition(model, subset(run_values, names))` when
that subset carries the necessary storage. Do not turn the subset into a dictionary
or a list of bare pairs. A subset cannot split one multivariate tilde into roles.

### Arguments and containers

Use absolute names on a prefixed model:
`condition(prefix(m, @varname(p)); y=v)` becomes
`condition(prefix(m, @varname(p)), @varname(p.y) => v)`.
For NamedTuple fields in bindings, removals and tildes, replace positional or
Symbol indexing such as `y[1]` or `y[:a]` with `y.a`.

Replace overlapping observation and fix bindings with removal followed by binding:
`fix(condition(m; x=data); x=value)` becomes
`fix(decondition(condition(m; x=data), @varname(x)); x=value)`.
Before replacing a child's own binding from its parent, remove that role at the
child's qualified address. Keep the original model if its observations must be
restored later.

For unsupported nested placeholders, provide concrete data and remove the latent
positions. For unsupported partial edits of tuples, structs or dictionaries,
replace the whole value or use a NamedTuple/array representation. If latent
writeability checks reject a range or array wrapper, use `collect` when losing
its axes or metadata is acceptable.

```@example bindings
@model nested_measurements(y) = (y[1][1] ~ Normal(); y[1][2] ~ Normal(); y)
nested_imputation = decondition(nested_measurements([[1.0, 0.0]]), @varname(y[1][2]))
@assert haskey(rand(Xoshiro(1), nested_imputation), @varname(y[1][2]))
whole_tuple = fix(decondition(measurements((1.0, 2.0))); y=(9.0, 2.0))
@assert whole_tuple(Xoshiro(1)) == (9.0, 2.0)
mutable_input = decondition(measurements(collect(1.0:2.0)))
@assert length(rand(Xoshiro(1), mutable_input)) == 2
```

For partial writes into `Float32` storage, pass `Float32(0.1)` instead of `0.1`.
For runtime AD values, use storage such as `fill(zero(θ), n)` or a template such as
`@of(y = of(Array, typeof(θ), n))`. To change a root's shape, remove both roles at
that root before supplying its new whole value or template. Fixed values must
still fit the storage used by executed tildes.

To observe raw data after transforming an argument in the body, use a separate
local tilde variable. Reconstruct the model and its partial bindings after
mutating argument storage. `conditioned(m)` includes argument observations; do
not merge `model.args` into it to recover them.

Keyword splats in `model.defaults` are nested under their generated name. For
`@model f(; kw...)`, use `model.defaults.var"#splat#kw".z` instead of
`model.defaults.z`. Bind a keyword-splat tuple as a whole, for example
`condition(m; kw=(; y=2))`. Inside the body, `kw` is Julia's `Base.Pairs`;
read its entries with `kw[:y]`. A closure's tilde variable cannot shadow a model
argument: if `x` is an argument, change `(x -> x ~ Normal())` to
`(z -> z ~ Normal())`.

### Direct construction and context access

Replace construction or dispatch using `Model{Threaded,missings}` with `@model`
and `decondition` for latent arguments. Direct constructors need `args_on_lhs`
to identify argument observations. A handwritten evaluator must also implement
argument preparation and binding lookup, or reuse an evaluator from `@model`.

```@example bindings
generated = measurements([1.0, 2.0])
manual = Model{false}(generated.f, generated.args, generated.defaults; args_on_lhs=(:y,))
@assert manual(Xoshiro(1)) == [1.0, 2.0]
```

Replace `conditioned(context)` and `fixed(context)` with model queries.
Replace `decondition_context` and `unfix_context` with model removals.
Replace `hasconditioned`, `getconditioned`, `hasfixed`, `getfixed` and their nested
variants with `haskey(conditioned(m), vn)`, `conditioned(m)[vn]` and the corresponding
`fixed(m)` operations. The removed argument/context role helpers, including
`inmissings`, `isassumption` and `isfixed`, also give way to model binding queries.
For debug introspection, use the callable and argument types returned by
`gen_evaluator_call_with_types`; do not assume that callable is `model.f`.

To transfer bindings formerly carried by `contextualize(new, old.context)`, replay
the model's listings when their addresses and storage fit the new model:

```@example bindings
old = fixed_model
new = unfix(decondition(model))
new = condition(new, conditioned(old))
new = fix(new, fixed(old))
@assert conditioned(new) == conditioned(old)
@assert fixed(new) == fixed(old)
```

This does not transfer removal instructions or bindings created in child bodies;
reapply those separately. Argument observations become explicit bindings. Preserve
prefixes, and replay any whole-value resize before the partial removals that depend
on it; a surviving partial listing need not reconstruct the earlier storage.
