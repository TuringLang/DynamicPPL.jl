# Conditioning and fixing

`condition` stores observations on a model; their distributions contribute to the
likelihood. `fix` stores values whose distributions contribute no density. Both
return a model with the requested bindings. `rand` draws its remaining latent
variables from their prior distributions, not from the posterior.

```@example bindings
using DynamicPPL, Distributions, Random
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

`condition` and `fix` accept `NamedTuple`s, keyword arguments, `Symbol` or `VarName`
pairs, tuples of these inputs, and `VarNamedTuple`s. `model | values` abbreviates
`condition(model, values)`. Dictionaries are not binding inputs.

```@example bindings
@assert conditioned(model | (; y=y_data)) == conditioned(observed)
@assert conditioned(model | (@varname(y) => y_data)) == conditioned(observed)
@assert conditioned(condition(model, :y => y_data)) == conditioned(observed)
@assert conditioned(condition(model, ((y=y_data,), :c => 3.0)))[@varname(c)] == 3.0
```

An indexed partial address needs storage when the binding is made. Storage can
come from concrete model arguments, including those reached through nested or
prefixed child models passed as arguments, an existing whole binding, a
`VarNamedTuple` that retains storage, or an explicit template. A future allocation
in a model body does not supply storage to the binding call; a child constructed
inside the body still needs a template.

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
cannot supply unbound positions in a `missing` or `nothing` argument: such partial
bindings throw `ArgumentError`, including with typed or snapshot templates. Bind
the whole value, bind every position with a template, or pass a concrete argument
(for example one with `missing` entries) as storage. Concrete arguments keep their
own storage.
Array extents, selection shapes and aliases are checked at binding time; this
includes `begin`, `end`, masks and linear versus Cartesian indexing.

Typed templates also constrain conversion. Values must be exactly representable
in the specified type; an inexact conversion throws. Use the runtime value's type
when constructing templates inside differentiated code.

```@example bindings
half = condition(model, @varname(y[1]) => 0.5, @of(y = of(Array, Float32, 3)))
@assert conditioned(half)[@varname(y[1])] === 0.5f0
```

`rand(rng, model)` records only the latent sites it reaches, omitting observed and
fixed sites and untaken branches. A subset of that `VarNamedTuple` retains storage
for its selected values. This supports blockwise conditioning without
reconstructing bindings from bare pairs.

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
observed or fixed while its remaining positions are latent. An array whose
elements hold both observed and fixed values is evaluated with dynamic dispatch:
for 100 element tildes, about 80 µs and 1,600 allocations, against 0.4 µs when
a single role binds the array.

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
cannot be bound. Partial edits support plain `Array`s, array-backed
`ComponentArray`s, tuples, and fields of `NamedTuple`s or opted-in
structs. ComponentArray bindings and tildes use indices: write `x[:a] ~ ...`,
not `x.a ~ ...`, which throws an `ArgumentError`. Partial edits of `OffsetArray`
and `DimArray` throw; whole bindings remain supported. Bind the whole value for
other containers, including unclaimed structs and dictionaries.
Inferred storage templates still require plain numeric arrays because `of`
cannot encode component axes.

Tuple elements keep their own types, and a tilde write replaces only the
element it writes. A tuple variable bound by element takes its tuple from the
model value; an `of(Array, n)` template gives it only its length. Partial
bindings into argument storage or a typed template must convert exactly to the
type of the element they bind. Tuple slice bindings such as `x[1:2]` and `x[:]`
accept tuple or array values of the same length, but a tilde over a tuple slice
throws unless it reads an observed argument or one bound whole by `condition`
or `fix`. Mooncake cannot yet differentiate a loop over a tuple argument with
`missing` entries; use an array argument or unroll the loop.

Argument bindings are prepared before the model body runs. Body computations
therefore see replaced arguments. An observed argument tilde uses the value
computed by the body; a fixed argument tilde restores its bound value. Local
variables obtain their bindings at the tilde itself. Replacement arguments must
satisfy declared argument types.

Evaluation copies latent argument storage for plain `Array`s, array-backed
`ComponentArray`s, tuples, nested `NamedTuple`s and opted-in structs.
Copies preserve container types and axes. Numeric matrix transposes, including
those admitting `missing`, also support partial edits and latent storage, but
Julia's reconstruction produces a plain `Matrix`; vector-parent and nonnumeric
transposes are unsupported.
Other values are shared as leaves, except unclaimed `AbstractArray`s and
`AbstractDict`s, which are rejected where latent storage is required. Fully bound branches and covariates do not need latent copies. Latent
tilde assignments cannot write into or descend through unclaimed objects;
replacing a whole leaf is allowed.

Storage shape, signature, alias and cycle checks descend only into containers
claimed by `_element_optic`; all other values are leaves. Latent source
selection follows the same rule. Cyclic latent containers and storage shared
between a latent path and another inspected path are rejected. Alias checks
inspect array leaves too, so a covariate view sharing a latent argument's storage
is rejected; derive the covariate from the drawn value inside the body or pass
independent storage. Sharing hidden inside other unclaimed objects is not
detected. Convert unsupported storage explicitly when losing axes or metadata
is acceptable.

Structs opt into field-addressed storage with
`DynamicPPL._element_optic(::MyStruct) = AbstractPPL.Property`. This is an
internal extension point. Fields are read through `propertynames` and
`getproperty` and rebuilt with `ConstructionBase.setproperties`, including for
mutable structs. The caller's struct and its claimed child containers are not
mutated.

When a struct becomes latent, binding validation probes reconstruction with
`ConstructionBase.setproperties(x, ConstructionBase.getproperties(x))`.
Reconstruction errors propagate at that boundary. Whole-bound branches and
observed siblings are not probed. Custom properties or keyword-only
constructors may require ConstructionBase methods; reconstruction follows
`constructorof` and must preserve the type's invariants. Field types must
accommodate the values used during evaluation, including AD numbers when
needed.

A handwritten `Model` evaluator must perform the argument preparation and role
lookup that `@model` generates. Calling `tilde_observe!!` directly does not look up
model bindings. Prefer `@model`; `args_on_lhs` metadata alone does not implement
binding semantics in a handwritten evaluator.

## Submodels

A prefixed submodel contributes its tilde address. In `a ~ to_submodel(child)`, a child
variable `x` has address `a.x`. An explicit child prefix composes with that address.
The tilde receives the child's return value; that return is not itself a bindable
random variable.

```@example bindings
@model inner(y, x) = (y ~ Normal(x, 1); y)
@model outer(y, x) = a ~ to_submodel(condition(inner(missing, x), (y=y,)))
@assert outer(1.0, 0.0)(Xoshiro(1)) == 1.0
```

Submodels cannot retain observations supplied as model arguments. Pass
covariates as arguments and use `condition` for observations. `fix` would omit
their likelihood. A whole parent `condition` or `fix` replaces a child
argument's observation; a whole parent `decondition` makes it latent. For partly
missing arrays, pass concrete storage and remove its argument observations
before binding only the observed entries. Unbound entries remain latent. The
same storage rule applies to `fix`, which omits the fixed entries' densities.

```@example bindings
@model function inner_vec(y)
    for i in eachindex(y)
        y[i] ~ Normal()
    end
    return y
end
@model function outer_vec(y)
    child = condition(
        decondition(inner_vec(y)),
        (@varname(y[i]) => y[i] for i in eachindex(y) if !ismissing(y[i]))...,
    )
    return a ~ to_submodel(child)
end
partial_child = outer_vec([1.0, missing])
@assert partial_child(Xoshiro(1)) == [1.0, rand(Xoshiro(1), Normal())]
@assert loglikelihood(partial_child, (; a=(; y=[1.0, 0.4]))) == logpdf(Normal(), 1.0)
```

Parent bindings cannot overwrite a child's explicit bindings, even in the same role;
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

An unprefixed submodel, `a ~ to_submodel(child, false)`, contributes no address: its
variable `x` has the parent's address `x`. The rules above apply with an empty prefix, so
parent bindings and removals at `x` reach the child, the child cannot observe through
its arguments, and parent bindings cannot overwrite the child's own. Because the parent
cannot list such a child's roots before evaluation, binding and removal functions accept
any root on it; a binding at the tilde's left-hand side `a` throws when the child is
reached, unless `a` is also one of the child's roots. A parent's argument observations
stay with the parent's own tildes: if such a tilde does not run, a child using the same
root draws it as a latent variable and ignores the argument. Binding a child
variable named like a parent covariate, including a keyword splat, throws "covariates
cannot be bound"; rename one of them or prefix the child.
The model body must contain the tilde as `a ~ to_submodel(child, false)`; an unprefixed
submodel built elsewhere throws.

```@example bindings
@model unprefixed_parent() = a ~ to_submodel(scalar_child(), false)
unprefixed_bound = condition(unprefixed_parent(); x=1.0)
@assert unprefixed_bound(Xoshiro(1)) == 1.0
@assert haskey(rand(Xoshiro(1), decondition(unprefixed_bound, :x)), @varname(x))
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

Model comparison and hashing include the evaluator, arguments, defaults, context
and binding state. `==` delegates to `isequal`; arguments and defaults compare by
content. Binding roles, retained storage constraints and removal instructions can
distinguish models with the same current listing.

## Missing data

A whole `missing` or `nothing` argument supplies no observation. A `missing`
entry directly in a plain argument `Array` or tuple is latent. This constructor behaviour
does not permit placeholders in explicit `condition` or `fix` inputs: those throw.
Unlike versions before 0.43, an accessed value that is `=== missing` is not
always unobserved. Missing values inside `NamedTuple` fields, nested arrays or
tuples, or structs are rejected. For a top-level array argument `x`,
a model body can use `all(ismissing, x)` to test whether every entry is
unobserved before executing its tildes.

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

Each imputed entry must have its own tilde. A multivariate tilde, such as one
`MvNormal`, over an argument array or tuple, or a slice of one, that contains
`missing` throws, also after `decondition`. Nested placeholders in explicit bindings are rejected, and
observed and fixed values are checked again when their tildes execute. To bind
part of an argument while leaving positions unbound, pass an array of the
required dimensions with `missing` entries. For an `Array{Missing}` such as
`fill(missing, 3)`, the model sees a copy whose element type admits the bound
values; other storage must hold them exactly. A wholly `missing` or `nothing`
argument must be bound completely, even when a template is supplied.

```@example bindings
storage = Array{Union{Missing,Float64}}(missing, 3)
@assert all(ismissing, storage)
partial = condition(measurements(storage), @varname(y[2]) => 2.0)
@assert conditioned(partial)[@varname(y[2])] == 2.0
```

## Common questions

**When are bindings resolved?** A model's bindings never change: `condition`,
`fix` and their inverses return new models. Each `~` resolves its address when
it runs, so data-dependent indices work; storage mismatches throw there.

**Why are removals through a parent applied late?** A child may be constructed
inside the parent's body, so its removals are retained until it is reached;
see [Removing bindings](@ref).

**Does an argument observation see body transformations?** Yes: its `~` uses
the argument's current value after reassignment, with no Jacobian, whereas an
explicit `condition` replaces the value before the body runs; see
[Arguments and containers](@ref).

**Can a record be bound field by field?** Yes, with its shape supplied by a
record template, argument storage, or a prior whole binding. Binding every field
in the same role then gives the same values and density as binding the whole
record; a partial record cannot supply a single tilde. Initialising a record in
the model body does not supply binding storage. For an array of records,
`of(Array, n)` describes only the array positions, and record-element templates
are not supported. Supply argument storage or bind the whole array before
rebinding its record fields.

**Does rebinding one field of an argument record change how a submodel treats it?**
Untouched fields retain their argument observations, so the model still cannot
be used as a submodel. Replace every argument observation explicitly, or bind
the whole record, as described in [Submodels](@ref).

## Migration to model-owned bindings

Bindings now live on `Model`, rather than in conditioning or fixing contexts.
Use `condition`, `fix`, `decondition`, `unfix`, `conditioned` and `fixed` instead of
constructing or inspecting the unused `CondFixContext` representation. Direct
evaluators must follow the preparation and lookup rules described above.

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
  - Replace `to_submodel(inner(y, x))` with
    `to_submodel(condition(inner(missing, x), (y=y,)))` when `y` is fully observed.
    For partly missing arrays, bind only observed indices and provide storage:
    `condition(decondition(inner_vec([1.0, missing])), @varname(y[1]) => 1.0)`.
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
positions. For unsupported partial edits, replace the whole value or use a plain
Array/NamedTuple representation. For latent storage, use `collect` on a range or
array wrapper only when losing its axes or metadata is acceptable.

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
`model.defaults.z`. A keyword splat on a tilde's left-hand side is bound as a whole
`NamedTuple`: keywords observe it, an empty splat leaves it latent, and `condition`
replaces it; `fix` of an observed splat needs `decondition` first. A splat that the
body only reads, as in `kw[:y]`, is a covariate and cannot be bound.

```@example bindings
@model splat_model(; kw...) = kw ~ product_distribution((y=Normal(),))
@assert conditioned(splat_model(; y=2.0)) == VarNamedTuple(; kw=(y=2.0,))
@assert haskey(rand(Xoshiro(1), splat_model()), @varname(kw))
@assert logjoint(condition(splat_model(); kw=(; y=2.0)), (;)) ≈ logpdf(Normal(), 2.0)
```

A closure's tilde variable cannot shadow a model argument: if `x` is an argument,
change `(x -> x ~ Normal())` to `(z -> z ~ Normal())`.

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
`isassumption` and `isfixed`, also give way to model binding queries.
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
