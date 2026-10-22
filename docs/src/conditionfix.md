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

`fix` accepts the same syntax; the functional template spelling is
`of((y=of(Array, length(y_data)),))`. Non-placeholder arguments supply their own storage.
See [Binding rules](@ref) for the complete contract.

## Migrating from 0.42

These examples use small models whose LHS variables can be bound separately:

```jldoctest migration
julia> using DynamicPPL, Distributions, StableRNGs

julia> @model function elements(x)
           x === missing && (x = zeros(2))
           for i in eachindex(x)
               x[i] ~ Normal()
           end
           return x
       end;

julia> @model fields(x) = (x.a ~ Normal(); x.b ~ Normal(); x);

julia> @model nested(x) = (x[1][1] ~ Normal(); x[1][2] ~ Normal(); x);

julia> struct Record
           a::Int
           b::Int
       end

```

  - Whole `missing` arguments and element-wise `missing` in supported argument arrays keep
    working. Elsewhere, replace placeholders with concrete storage and decondition the parts:
    `nested([[1.0, missing]])` → `decondition(nested([[1.0, 0.0]]), @varname(x[1][2]))`.
    Explicit bindings never accept placeholders:
    `condition(elements(zeros(2)); x=[1.0, missing])` → bind concrete data, then decondition
    `x[2]`. Use `unfix` for placeholders formerly passed to `fix`.
    
    ```jldoctest migration
    julia> length(rand(StableRNG(1), elements(missing)))
    2
    
    julia> haskey(rand(StableRNG(1), elements([1.0, missing])), @varname(x[2]))
    true
    
    julia> m = decondition(nested([[1.0, 0.0]]), @varname(x[1][2]));
    
    julia> haskey(rand(StableRNG(1), m), @varname(x[1][2]))
    true
    
    julia> m = decondition(condition(elements(zeros(2)); x=[1.0, 0.0]), @varname(x[2]));
    
    julia> m(StableRNG(1))[1] == 1.0 && haskey(rand(StableRNG(1), m), @varname(x[2]))
    true
    
    julia> m = unfix(fix(decondition(elements(zeros(2))); x=[1.0, 0.0]), @varname(x[2]));
    
    julia> haskey(rand(StableRNG(1), m), @varname(x[2]))
    true
    ```

  - Partial edits through views and other unsupported array storage now throw. For
    `decondition(elements(view(zeros(2), :)), @varname(x[1]))`, pass `collect(v)` instead
    if losing axes or metadata is acceptable. Whole deconditioning of immutable storage
    also requires writable storage: `decondition(elements(1.0:2.0))` → collect the range.
    
    ```jldoctest migration
    julia> v = view(zeros(2), :);
    
    julia> m = decondition(elements(collect(v)), @varname(x[1]));
    
    julia> haskey(rand(StableRNG(1), m), @varname(x[1]))
    true
    
    julia> length(rand(StableRNG(1), decondition(elements(collect(1.0:2.0)))))
    2
    ```
  - Partial edits rebuilding tuples, structs or `Base.Pairs` now throw; bind or remove the
    enclosing value whole. For example, `fix(elements((1.0, 2.0)), @varname(x[1]) => 9.0)`
    → `fix(decondition(elements((1.0, 2.0))); x=(9.0, 2.0))`. For
    `fix(pairdata(pairs((a=1.0,))), @varname(x[:a]) => 3.0)`, use a whole `Base.Pairs` value.
    
    ```jldoctest migration
    julia> fix(decondition(elements((1.0, 2.0))); x=(9.0, 2.0))(StableRNG(1))
    (9.0, 2.0)
    
    julia> fix(decondition(condition(fields(Record(1, 2)); x=(a=3, b=4))); x=(a=5, b=4))(StableRNG(1))
    (a = 5, b = 4)
    
    julia> @model pairdata(x) = (x[:a] ~ Normal(); x);
    
    julia> fixed(fix(decondition(pairdata(pairs((a=1.0,)))); x=pairs((a=3.0,))))[@varname(x)][:a]
    3.0
    ```
  - NamedTuple fields now use properties in bindings, removals and tildes:
    `x[1]` or `x[:a]` → `x.a`. Ordinary Julia indexing in the body is unchanged.
    
    ```jldoctest migration
    julia> condition(fields((a=1.0, b=2.0)), @varname(x.a) => 3.0)(StableRNG(1))
    (a = 3.0, b = 2.0)
    
    julia> haskey(
               rand(StableRNG(1), decondition(fields((a=1.0, b=2.0)), @varname(x.a))), @varname(x.a)
           )
    true
    ```
  - Prefixes now require properties or non-`Bool` scalar integer indices:
    `prefix(m, @varname(p[1:2]))` → `prefix(m, @varname(p[2]))`. For an automatically
    prefixed sliced return LHS, use `to_submodel(child, false)` to keep the slice without a prefix.
    
    ```jldoctest migration
    julia> @model leaf(x=missing) = x ~ Normal();
    
    julia> prefix(leaf(2.0), @varname(p[2]))(StableRNG(1))
    2.0
    ```
  - A named recursive removal cannot overlap a parent's LHS and an unprefixed child's LHS.
    `decondition(shared(2.0, leaf(3.0)), DynamicPPL.Recursive(), @varname(x))` → remove
    without `Recursive()` to clear only the parent's observation, or prefix the child.
    
    ```jldoctest migration
    julia> @model shared(x, child) = (x ~ Normal(); a ~ to_submodel(child, false); (x, a));
    
    julia> decondition(shared(2.0, leaf(3.0)), @varname(x))(StableRNG(1))[2]
    3.0
    ```
  - A submodel tilde cannot be rooted at a model argument: `x ~ to_submodel(child)` with
    argument `x` → use a local return name. Bind the child's LHS variables, or assign the
    returned value to the argument afterwards.
    
    ```jldoctest migration
    julia> @model returned(x, child) = (result ~ to_submodel(child); x = result; x);
    
    julia> returned(0.0, condition(leaf(); x=2.0))(StableRNG(1))
    2.0
    ```
  - A position can have only one binding across roles and models. Replace
    `fix(condition(m; x=data); x=value)` with
    `fix(decondition(condition(m; x=data), :x); x=value)`. Removing the fix makes `x`
    latent; retain the original model to restore its observation. Before binding a child
    address from a parent, remove the child's binding with `Recursive()` in the matching
    role. A child's own removal cannot erase an ancestor binding.
  - An observed array with immutable storage cannot become latent or fixed through removal.
    Reconstruct it with mutable storage, for example `elements(collect(1.0:2.0))`.
  - A partial binding of a child-local variable needs storage before the child's body.
    Supply a binding template or bind storage on the child first; allocation inside the
    body is insufficient, including in a branch that never executes.

## Binding rules

`condition` binds an observed value and `fix` binds a fixed one. A position holds at most
one binding across both roles and all models. Binding it again on the same model with the
same role replaces its value, including an argument-supplied observation. Binding it with
the other role, or from a parent over a child's binding, throws; remove the existing binding
first. `|` is the template-free alias for `condition`.

```jldoctest binding_rules
julia> using DynamicPPL, Distributions, LinearAlgebra, StableRNGs;
       using AbstractPPL: of, @of;

julia> @model vector_model() = x ~ MvNormal(zeros(2), I);

julia> m = condition(vector_model(); x=[1.0, 2.0]);

julia> conditioned(condition(m; x=[3.0, 4.0]))[@varname(x)]
2-element Vector{Float64}:
 3.0
 4.0

julia> fix(m, @varname(x[1]) => 0.0)
ERROR: ArgumentError: Cannot bind `x[1]`: a position is already bound by the other role; remove that binding first, using `Recursive()` for a child binding.
[...]
```

Binding part of a variable, such as an index or field, needs storage with a known shape:
a non-placeholder argument, an earlier whole binding, a `VarNamedTuple` such as
`rand(rng, model)`, or a binding template.
A whole value can establish its own shape. A template supplies storage but binds nothing;
unused entries are ignored. See [Binding contract](@ref) for template syntax and limits, and
[Binding roles and removal](@ref) for how removal preserves storage.

Bindings can reach child addresses: `condition(parent, @varname(a.x) => 1.0)` binds the
child's `x`. Conflicts with a child's bindings, including its argument-supplied observations,
are checked when the child runs; remove the child's binding first:

```jldoctest binding_rules
julia> @model inner(x) = x ~ MvNormal(zeros(2), I);

julia> @model outer() = a ~ to_submodel(inner([1.0, 2.0]));

julia> bound = condition(outer(), @varname(a.x) => [0.0, 0.0]);

julia> rand(StableRNG(1), bound)
ERROR: ArgumentError: Cannot bind `a.x`: a position is already bound by another model; remove that binding first, using `Recursive()` for a child binding.
[...]

julia> @model outer2() = a ~ to_submodel(decondition(inner([1.0, 2.0])));

julia> fix(outer2(), @varname(a.x) => [0.0, 0.0])(StableRNG(1))
2-element Vector{Float64}:
 0.0
 0.0
```

A reached child needs storage for its local parts before its body runs, even when the branch
containing their tilde is not taken; allocation inside the body is insufficient.
Storage checks wait until child entry, since the parent may construct the child inside its body.

```jldoctest binding_rules
julia> @model function local2()
           x = Vector{Float64}(undef, 2)
           x[1] ~ Normal()
           x[2] ~ Normal()
           return x
       end;

julia> @model outer3() = a ~ to_submodel(local2());

julia> partial = condition(outer3(), @varname(a.x[1]) => 0.5);

julia> partial(StableRNG(1))
ERROR: ArgumentError: Cannot bind `a.x` without storage; supply an `@of` template, a whole value, a `VarNamedTuple` with storage, or storage on the child before prefixing.
[...]

julia> condition(outer3(), @varname(a.x[1]) => 0.5,
                 @of(a = @of(x = of(Array, 2))))(StableRNG(1))[1]
0.5
```

Argument-supplied observations never bind a child's LHS variables. An explicit binding at
a name shared with an unprefixed child binds both:

```@example shared-binding
using DynamicPPL, Distributions, Random
@model leaf(x) = x ~ Normal()
@model function shared(x, child)
    x ~ Normal()
    a ~ to_submodel(child, false)
    return (x, a)
end
condition(shared(2.0, decondition(leaf(3.0))), @varname(x) => 1.0)(Xoshiro(1))
```

Produced values round-trip with prefixed and unprefixed children:
`condition(model, rand(rng, model))` observes the sampled LHS variables.

An **LHS variable** is the addressed left-hand side of one execution of `~`. An **LHS
subvariable** is part of it, such as `x[1]` of `x ~ MvNormal(...)`. An LHS variable's **role**
is latent (initialised), observed (scored in the likelihood), or fixed (no log probability). A
**binding** supplies a bound value and an observed or fixed role. `condition` and `fix` make
**explicit bindings**, while arguments on the LHS supply **argument-supplied observations**.

The **effective binding** determines an LHS variable's role and value after applicable
removals. A position without a binding is latent. Disjoint bindings may share a top-level
variable, but all positions of one executed tilde must have the same role.

Addresses within one call must be disjoint after resolution through storage, and a selection
cannot repeat a position.
Field and flat-index selectors can overlap once a child supplies its actual array layout;
such overlaps are checked before the child's body runs.

### [Rule table](@id binding-rule-table)

The table defines the supported binding rules and when each is checked. The
[entry-point matrix](@ref binding-phase-matrix) shows how those phases apply to each operation.
Preparation assembles model arguments and resolves child-local partial bindings before the
body. A local branch that does not execute still needs valid binding storage. Checks that
need an unreached child wait until that child is entered; role, coverage and body-size checks
require an executed tilde.

| Rule                        | Contract                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Checked when                                                                                                                                                                             |
|:--------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Address interpretation      | Bindings and removals act at or below the called model's prefix. Scalar `CartesianIndex` expands to integer coordinates. Storage resolves linear and Cartesian aliases. A pending child address retains one index style until the child supplies storage; mixing styles before then throws.                                                                                                                                                                                                                                                                                                                                                                                                                                      | At binding/removal, or before the addressed child's body.                                                                                                                                |
| Model ownership             | Only declared LHS top symbols may be bound or removed when ownership is knowable. Covariates and separate keyword-splat entries cannot be bound. Only a literal `to_submodel(child, false)` tilde defers unknown top symbols to an unprefixed child; shared unprefixed namespaces cannot reject unknown names independently of siblings. Valid unmatched removals are no-ops, including repeated removals.                                                                                                                                                                                                                                                                                                                       | At binding/removal where decidable; child ownership when reached. Direct construction checks the supplied argument metadata (see [`Model`](@ref)).                                       |
| Field and container grammar | NamedTuple fields use properties, never indices, including at tildes. Partial edits traverse only NamedTuple owners, `Array`, and Array-backed `OffsetArray`, `ComponentArray` and `DimArray` owners; untouched leaves are unrestricted. Tuple/struct rebuilding and other array families are rejected. Whole replacement, argument observations through fields/indices, and complete local bindings remain supported. Ordinary Julia body indexing is unaffected.                                                                                                                                                                                                                                                               | At partial binding/removal; field spelling also at executed tildes. Whole operations still obey latent writeability and value rules.                                                     |
| Recursive ambiguity         | A named recursive removal overlapping both a parent's LHS and an unprefixed child's LHS is rejected. Binding the shared name and clearing a layer recursively without names remain allowed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | When the affected child is reached.                                                                                                                                                      |
| Submodel LHS and prefixes   | A submodel tilde cannot be rooted at an enclosing model argument, including its indices or fields. Prefixes allow properties and non-Bool scalar integers after scalar Cartesian expansion; slices, colons and masks are rejected. `begin` and `end` need prefix storage.                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Argument-root rejection only when the submodel tilde executes; prefix grammar when applying an explicit prefix or entering an automatically prefixed child.                              |
| Latent writeability         | Latent draws use a private per-evaluation copy, preserving the caller's argument. That copy may widen its element type with or without AD; a whole tilde replaces its value. Immutable backing arrays are rejected through `parent`, nested tuples/NamedTuples/arrays and latent branches of partially bound storage, even for a whole tilde. SparseArrays and supported AD storage are exempt. This is not a general object-graph writeability guarantee; see [Storage written by latent tildes](@ref) for limits.                                                                                                                                                                                                              | After removal exposes latent storage, including constructor placeholder removal; also before the body when preparing a latent argument copy. Recursive removals check reached children.  |
| Numeric leaves              | A custom number in a latent branch is rejected if its fields reach mutable storage, even if the number itself is immutable. Isbits values, `BigFloat`, `BigInt`, AD values supported by package extensions, and Base numbers whose fields recursively satisfy the exemption are allowed. Fully bound branches and covariates are exempt.                                                                                                                                                                                                                                                                                                                                                                                         | When preparing a latent argument copy, not unconditionally at binding/removal.                                                                                                           |
| Argument sharing            | Repeated latent memory within/across arguments or through an argument-observed branch is rejected, even if the body only reads or replaces it. Views, reshapes and `unsafe_wrap` aliases count. Original storage of explicitly replaced branches and separate explicit binding values do not count. Sharing only among observations/covariates is allowed. Errors identify argument roots, not full alias paths; aliases created by the body are not checked.                                                                                                                                                                                                                                                                    | At binding/removal, including constructor placeholder removal; after inherited removals and insertions form the final state at child entry. No new sharing scan at each tilde.           |
| Retained-copy overlap       | An argument containing an AD value and a separate array sharing its buffer throws when copied for sampling. Give the separate array independent storage.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | When preparing a latent argument copy.                                                                                                                                                   |
| Dictionary keys             | In an argument with latent parts, dictionary keys anywhere in its copied graph must be isbits values, `Symbol`s or `String`s, including keys in bound siblings. Non-isbits immutable keys can also be rejected. A wholly bound argument or separate covariate is exempt.                                                                                                                                                                                                                                                                                                                                                                                                                                                         | When preparing a latent argument copy, not unconditionally at binding/removal.                                                                                                           |
| Index arity                 | Arguments, binding templates and retained geometry follow Julia indexing. A pending child-variable index resolves through the child's storage, so linear `a.x[2]` can reach matrix element `x[2, 1]`. Without namespace storage, an indexed submodel prefix must use the pending namespace's index count.                                                                                                                                                                                                                                                                                                                                                                                                                        | At construction for mixed pending styles; at the reached child for pending indices.                                                                                                      |
| Selection geometry          | Source extent and selected shape are distinct: a Boolean vector requires its full source extent but selects `count(mask)` entries; an empty integer selection has extent zero. Child indices are bounded by the selected shape. Indexed plain locals require storage when bound. Pending child parts resolve against child storage; unresolved `begin`, `end` and `:` throw at binding and suggest integer indices. Property paths need no storage.                                                                                                                                                                                                                                                                              | At binding/removal, or before a reached child's body for pending parts.                                                                                                                  |
| Constructor placeholders    | Only a whole top-level `missing`/`nothing` LHS argument (positional, keyword or default), or `missing` entries one index into a supported top-level argument array, remove observations at construction. The array element type must admit `Missing`; unassigned slots and covariates are skipped. Zero-dimensional arrays are included. Roles are not recomputed after mutation. Arrays with missing entries are snapshotted; fully observed arrays are used in place.                                                                                                                                                                                                                                                          | User model construction, via ordinary removal. Internal reconstruction and the explicit-values constructor do not rerun this classification.                                             |
| Explicit placeholders       | Explicit bindings reject `missing`/`nothing` reached through assigned array entries, tuple/NamedTuple entries or defined object fields. Numbers, numeric arrays, strings, symbols and types are opaque to this traversal. Use `decondition`/`unfix` instead. Incomplete partial bindings into a whole placeholder argument are rejected; a complete produced `VarNamedTuple` array entry binds whole and supplies storage.                                                                                                                                                                                                                                                                                                       | When the explicit binding is made; this does not replace executed-value checks.                                                                                                          |
| Executed placeholders       | An observed/fixed tilde rejects other placeholders in the value it reads, including nested argument data and body-created values, using the same traversal as explicit bindings. Unread parts may contain placeholders. `missing` takes diagnostic precedence over `nothing`. Latent draws are not reclassified as observations.                                                                                                                                                                                                                                                                                                                                                                                                 | Only when the observed/fixed tilde executes.                                                                                                                                             |
| Storage ownership           | Each root has one shape, shared by both layers. Removal retains local geometry; whole removal of both layers restores an argument's own storage. Arguments, used templates, whole bound values and concrete `VarNamedTuple` storage establish it. A whole value or template can replace it only after both layers are empty at the root. Partial bindings cannot resize known nested storage. A `NamedTuple` splits into leaf roots only at a submodel prefix; elsewhere it has one root. A `VarNamedTuple` binds its leaves as written. Roots are never implicitly split or merged.                                                                                                                                                                                                                   | At binding/removal; child storage when reached.                                                                                                                                          |
| Binding templates           | At most one positional binding template supplies storage without binding values; keywords remain data, and the pipe alias rejects templates. Templates use absolute names; argument and unused entries are ignored. Existing shapes win while either layer has stored values, including wholly masked storage; conflicts reject. A template’s shape persists until another template replaces it. Namespace storage survives removal. Child-owned template candidates follow pending bindings and prefixes. Template entries for every child's names are checked when that child runs. Templates materialize with `zero(T)` at binding time. See [Binding contract](@ref) for sample-derived templates and indexed-prefix limits. | At binding; deferred entries and conflicts when the child is reached. `check_model` warns about entries no reached LHS can use.                                                          |
| Bounds, types and shape      | Parts must fit enclosing storage without resizing it and convert exactly to its element/field type. Whole argument bindings must satisfy declared types; whole local bindings fit a binding template’s type when one exists; otherwise they constrain geometry only. Whole templated values also match shape. Partial bindings need storage; mixed roles within one LHS variable reject. Fixed values cover their contents; omitted NamedTuple fields remain latent. Fixed arrays and whole observations of locals follow the [size-check rules](@ref binding-size-checks); checks stop at the LHS address and do not detect in-place mutation of bound values.                                                                   | Bounds/types/conversion at binding (inherited bindings when the child runs); shared signature constraints at evaluation. Multivariate roles, coverage and body-size checks at executed tildes. |

### [Entry-point and phase matrix](@id binding-phase-matrix)

This matrix applies the [rule table](@ref binding-rule-table). A later preparation check is
not an unconditional rejection at binding or removal. Binding, lookup and removal need not
accept identical index syntax.

| Entry point                             | Address ownership                                                    | Writeability                                                                             | Sharing and copy                                                                                   | Indices                                                                          | Placeholders                                                                             | Templates and shape                                                         |
|:--------------------------------------- |:-------------------------------------------------------------------- |:---------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------- |
| `condition` / pipe alias                | Address, model ownership and partial-container rules.                | Partial-container rule now; latent writeability/numeric leaves during later preparation. | Argument sharing now; retained-copy overlap and keys during checked copying.                       | Bounds, arity and selection geometry through storage operations.                 | Explicit placeholders.                                                                   | Observation-layer owners and value checks; pipe takes no template.          |
| `fix`                                   | Same binding address checks.                                         | Same partial-container rule; fully fixed storage needs no latent writes.                 | Same call/copy distinction as `condition`.                                                         | Same binding storage checks.                                                     | Explicit placeholders; use `unfix` to remove.                                            | Fixed-layer owners and value checks.                                        |
| `decondition`                           | Removal address checks; recursive ambiguity at reached child.        | Resulting latent storage now; numeric leaves during checked copying.                     | Argument sharing now; overlap and keys during checked copying.                                     | Removal uses bounds/geometry; absent pending addresses are no-ops.               | No recursive scan; constructor removal uses this path.                                   | Keep surviving observation owners and templates.                            |
| `unfix`                                 | Same removal address checks.                                         | Only resulting latent storage; numeric leaves later.                                     | Same call/copy distinction as `decondition`.                                                       | Same removal storage checks.                                                     | No new scan.                                                                             | Removed positions become latent; preserve surviving templates.              |
| User model construction                 | Roles and argument LHS metadata checked.                             | Placeholder removal checks latent writeability; otherwise during later preparation.      | Placeholder removal checks sharing; overlap/keys later. Internal reconstruction reuses validation. | Missing-element addresses, including zero-dimensional arrays.                    | Constructor placeholders only; not rerun by explicit-values construction.                | Argument defaults own initial storage; no binding-template input.           |
| Argument preparation and ordinary tilde | Field spelling and runtime role lookup at tilde.                     | Latent writeability and numeric leaves before body; private-copy write/rebind at tilde.  | Overlap and keys before body; no new sharing walk per tilde.                                       | Child-local parts resolve before the body; ordinary Julia body indexing remains. | Executed placeholders at observed/fixed tilde.                                           | Runtime roles, coverage, assembly and body-size checks.                    |
| Submodel entry                          | Submodel LHS, prefixes, recursive ambiguity and inherited ownership. | Child preparation checks latent writeability/numeric leaves.                             | Sharing after inherited removals and insertions; overlap/keys in child preparation.                | Same child storage checks after prefix resolution.                               | Constructor placeholders if a child is constructed; executed placeholders at its tildes. | Prepare surviving templates; inherited and child bindings must be disjoint. |

### Binding roles and removal

| Operation     | Effect on this model's bindings                                          |
|:------------- |:------------------------------------------------------------------------ |
| `condition`   | Adds observations or replaces this model's observations.                 |
| `fix`         | Adds fixed values or replaces this model's fixed values.                 |
| `decondition` | Removes explicit or argument-supplied observations, leaving them latent. |
| `unfix`       | Removes fixed bindings, leaving their positions latent.                  |

Removal never restores an older binding; keep the original model if a temporary change must
be undone.

Removal matches equal, enclosing, or contained addresses. Without `Recursive()`, it clears
only bindings stored on the called model, including bindings at child addresses. With no
names, it clears that model's corresponding role. `Recursive()` also clears child bindings:

```julia
@model leaf(x=2.0) = x ~ Normal()
@model outer() = a ~ to_submodel(leaf())
original = outer()
decondition(original, @varname(a.x))()  # The child's observation remains 2.0.
released = decondition(original, DynamicPPL.Recursive(), @varname(a.x))
m = condition(released, @varname(a.x) => 3.0)
decondition(m, @varname(a.x))           # a.x is latent; no earlier observation returns.
```

Binding `a.x` on `original` would collide with the child's observation when reached.
Recursive removal permits the ancestor insertion by clearing the child first. It also
reaches argument-supplied observations and runtime bindings. Each removal affects only its
role. `check_model` warns about recursive removals unused by all reached models; an ordinary
removal at an unused child address is a silent no-op.

A removal belongs to the model that makes it. It reaches enclosed models, including those
built or bound in the body, but cannot remove an enclosing model's binding. Prefixing moves
the removal with the model. A later binding replaces a local removal only at the addresses it
covers. A recursive removal remains in force: it clears child bindings before later parent
bindings apply. `conditioned` and `fixed` list stored values only and take no `Recursive()`
argument.

Removing parts keeps the variable's storage, including its shape and element type, even
after every part is gone:

```jldoctest binding_removal
julia> using DynamicPPL, Distributions, LinearAlgebra;

julia> @model vector_model() = x ~ MvNormal(zeros(2), I);

julia> m = vector_model();
       m = condition(m; x=[1, 2]);
       m = decondition(m, @varname(x[1]));
       m = decondition(m, @varname(x[2]));

julia> condition(m, @varname(x[1]) => 0.1)
ERROR: InexactError: Int64(0.1)
[...]
```

Removing the whole root discards that role's values and storage element type but keeps the
local binding shape. Once both roles' storage is cleared, a whole value can set a new shape;
an argument instead reverts to its own storage. A template-established shape and type
constraint persist until a new template replaces them. A namespace template cannot change
an array root's shape while a `Recursive()` removal remains below it; rebuild the model.

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

For a named recursive removal rejected under **Recursive ambiguity** in the
[rule table](@ref binding-rule-table), prefix the submodel or remove without `Recursive()`.
For a sliced submodel return LHS, use `to_submodel(child, false)` to avoid a slice prefix.
For a partial edit through a tuple or struct owner, bind or remove that owner whole.

### Argument contract and shared constraints

Subvariables of one LHS variable must have the same role. Mixing roles throws `ArgumentError`
during evaluation. A bound **local LHS variable**, whose top symbol does not name an argument,
reads its effective binding's value at its tilde.

For an **argument LHS variable**, whose top symbol names an argument, every binding replaces the
argument before the body runs. Whole observations follow the argument's current value,
including newly appended positions; partial observations keep their selected positions. A fixed argument
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

Fixed arrays and whole observations of locals must match the storage at an executed tilde,
for element-wise and whole-array tildes. Fixed arguments also require this check; observed
arguments follow body resizing. When the body has not assigned a whole local array, its
fixed value must instead match the distribution's sample size (`size(dist)`); a mismatch
raises `ArgumentError`. This distribution-size check does not apply to a local with body
storage. Fixing a field does not require complete fixed coverage of its enclosing
`NamedTuple`. Shape checks stop at the LHS address; they do not inspect interiors beyond a
whole-variable tilde.

The **Storage ownership** and **Bounds, types and shape** rules in the
[rule table](@ref binding-rule-table) govern edits at every depth. For example, binding
a local `x=ones(3)` then removing all values of `x` still preserves length three, so `x[end]`
continues to mean `x[3]`. To change that length, clear both layers at `x`, then bind a
whole replacement. An argument's observation counts too:

```julia
condition(decondition(model_with_argument(zeros(2))); x=ones(3))
```

After whole removal of both layers at an argument root, its shape reverts to the argument’s own shape. Removing elements individually retains the masked replacement storage.

A namespace template cannot change its root’s shape while a recursive removal remains
below that root. The error names the root and asks you to rebuild the model; a template
with the same shape remains allowed.
A `NamedTuple` bound at a submodel prefix groups leaves: after binding `a=(x=1.0,)`,
a later binding of `a.y` adds another leaf. A whole structured value of a variable instead
has one root covering the parts it contains. A fixed value covers only what it holds:
`fix(m; q=(a=1.0,))` fixes `q.a` and leaves `q.b` latent.

#### [Size checks at executed tildes](@id binding-size-checks)

| Binding at an executed tilde     | Size given by the body checked against the binding?                |
| -------------------------------- | ------------------------------------------------------------------ |
| Fixed, whole or partial          | Yes. A mismatch throws.                                            |
| Whole observation of a local     | Yes. A mismatch throws rather than dropping bound values.          |
| Whole observation of an argument | No. The observation follows the value the body gives the argument. |
| Partial observation              | No. Each tilde reads only its own positions.                       |
| Latent                           | Nothing to compare.                                                |

Only executed tildes check, so an untaken branch checks nothing. The check compares along
the address path to the bound variable, not inside a whole structured value. A bound position
no executed tilde reads is not an evaluation error in either role; `check_model` reports it.
For example, this applies to a partial binding of `y[3]` through `@of(y = of(Array, 3))`
when the body makes `y` length 2. Mutating a bound value in place, such as `push!(x, 0.0)`,
is undetected and must be avoided.

A shape describes geometry, including known nested sizes. Whole argument replacements of
another type remain allowed when their geometry and the declared signature fit. Partial
conversion uses the edited layer's element or field type. A previously unassigned slot or a
missing top-level argument array entry acquires its inner size from its first explicit value;
removing the value retains that size unless whole removal restores an argument's own storage.
The body sets local storage's shape, subject to the checks above.

For NamedTuple fields, use `x.a` rather than `x[1]`, `x[:a]` or `x[:]`. Tuple observations
and complete local LHS variables retain integer indices. A **submodel namespace** reaches
child LHS variables through addresses such as `a.x`, or through unchanged names with
`auto_prefix=false` unless manually prefixed.

For example, if `a` is an argument, `a ~ to_submodel(child())`,
`a[1] ~ to_submodel(child())` and `a.x ~ to_submodel(child())` are rejected by the
**Submodel LHS and prefixes** rule. Use a new local name and condition the child's LHS
variables. Ordinary Julia assignment can still copy the return value into an argument;
a later distribution tilde observes that argument's current value.

Explicitly binding a **submodel return value**, assigned by local `a ~ to_submodel(...)`,
also throws `ArgumentError` during evaluation. Bind `a.x` to observe the child's `x`.
An unreached submodel performs no child-dependent checks.

Bound values are not copied, so the body must not mutate them, even through a `view`. A partial
binding copies containers along its edited path when storage first becomes available: at the
binding call for ancestor-held storage, or afresh at each child entry for live child storage.
Deferred checks never recreate an existing snapshot. Containers off the path remain shared. In a
model that observes each scalar of `x = [[1., 2.], [3., 4.]]`, binding `@varname(x[1][1]) => 9.`
does not isolate `x[2]`. Subsequently setting `x[2][1] = 99.` makes evaluation return
`[[9.0, 2.0], [99.0, 4.0]]` if the model returns `x`.

**Runtime bindings** are made inside the running body, as in `a ~ to_submodel(condition(child(y), @varname(y[1]) => m))`. Remade at each evaluation, their values
carry derivatives with respect to enclosing latent variables. Bindings made beforehand hold
values constant with respect to model parameters.

See [Missing data](@ref) for making observations latent. `InitFromParams` rejects `missing`
when read, not at construction. Omit unobserved values instead.

### Storage written by latent tildes

The [rule table](@ref binding-rule-table) defines latent writeability, numeric leaves,
argument sharing, retained-copy overlap and dictionary keys; the
[phase matrix](@ref binding-phase-matrix) distinguishes call-time checks from argument copying.

For shared storage such as `f((a=v, b=v))` or `f([v, v])`, pass independent storage, as in
`f((a=v, b=copy(v)))`, and read the latent value through its own address. A whole tilde such
as `x.a ~ MvNormal(...)` replaces its array, and widening a draw's storage also leaves other
references pointing to the old array. The separate-binding exemption permits
`condition(decondition(f(v)), @varname(y) => v)`. Under ReverseDiff, `copy(m)` of a tracked
array retains storage; use an allocating operation such as `m .+ 0` to preserve derivatives
with independent storage.

For a dictionary in an argument with latent parts, use isbits, `Symbol` or `String` keys,
or pass it as a separate covariate. For custom numbers whose fields reach mutable storage,
keep that storage outside the number. An argument containing an AD value and a separate array
sharing its buffer throws when copied for sampling. Give the separate array independent
storage; for example, `b[1] = ReverseDiff.value(z)` → `b[1] = copy(ReverseDiff.value(z))`.

The writeability check rejects ranges, `SVector`, `SMatrix` and `Fill` storage, but accepts
`MVector`, `SizedArray`, views of mutable arrays and SparseArrays. An immutable wrapper
without a `parent` method can be rejected even if it holds mutable buffers. Conversely,
a read-only wrapper over a mutable parent, such as `Symmetric` or `Diagonal`, and immutable
storage inside a struct field or `Dict` are not detected and can fail at evaluation.
Pass `collect(x)`, or `missing` for a whole tilde such as `x ~ MvNormal(...)`.
Whole `condition` and `fix` of these arguments remain supported.

Aliases made in the model body are not detected:

```julia
@model function local_alias()
    v = zeros(1)
    w = v
    v[1] ~ Normal()  # under AD, replaces `v`; `w` keeps the old array
    return 0.0 ~ Normal(w[1])
end
```

Take such an alias after the tilde, or allocate storage that holds the draws, such as
`zeros(Real, 1)`. Replacing an array to fit a draw also changes the argument's type in the
copy: for `@model g(x) = (x[1] ~ Normal(); x)`, `decondition(g(zeros(Int, 1)))()` returns a
`Vector{Float64}`.

### Binding contract

The partial-container rule in the [rule table](@ref binding-rule-table) applies to every
owner along an edited path. For example, views, reshapes, `Transpose`/`Adjoint`, immutable
arrays, ranges, ReverseDiff tracked arrays, `MVector`, `SizedArray` and `BitArray` do not
support partial edits. A child partially deconditioned inside a model may therefore work
with ForwardDiff but throw with ReverseDiff. Bind or decondition the whole value, or use
`collect(v)` if losing axes or metadata is acceptable. Whole operations are subject to the
separate latent-writeability rule, not the partial-array whitelist.

Use NamedTuples or keyword arguments for whole top-level values, and `VarName` pairs for any
address. The pair `:x => v` abbreviates `@varname(x) => v`. `VarNamedTuple`s produced by
DynamicPPL are also accepted.
`condition` and `fix` reject every `AbstractDict` and other unsupported input with
`ArgumentError`; `model | dict` throws `MethodError`. Use a NamedTuple or ordered pairs.

A **binding template** is an AbstractPPL `OfNamedTuple` type, such as `@of(z = of(Array, 3))`.
Import `of, @of` from AbstractPPL. Its contract is in the
[rule table](@ref binding-rule-table): it supplies binding storage, while the body still
sets local storage's size. Whole observations of locals must match that size; partial bindings
at unread positions are reported by `check_model`. See
[Size checks at executed tildes](@ref binding-size-checks). A used template for storage that
already exists only checks it, unless both roles' storage is cleared at the root so the
template can replace its shape. Under `prefix(m, @varname(p))`, use absolute names:
`@of(p = @of(z = of(Array, 3)))`.
Resolve symbolic sizes before binding; bind whole values for storage that `of` cannot describe.

A sample can supply a binding template with `of(rand(rng, model))`. This describes storage,
not observations: array values and partial masks are discarded, while shape and element type
are retained. It is a snapshot; rebuild it when the sample layout changes. Plain numeric
arrays, exactly representable scalars and nested namespaces are supported. Custom arrays,
growable entries, pending bindings and scalars whose type `of` would widen are rejected.
Listings can hold pending bindings and round-trip through `condition` or `fix`.
Before a body-built child runs, an unresolved `NamedTuple` keeps its supplied address,
such as `a.b`. Several writes at that address list their merged fields, with later writes
replacing shared fields. Removed fields are absent. A child passed directly as a model
argument provides its namespace information, so its bindings list at leaf addresses such
as `a.b.x`.

An indexed prefix, such as `prefix(model, @varname(p[1, 2]))`, cannot take a template: the
template grammar cannot express indexed namespaces such as `p[1, 2].y`, so `of(rand(rng, model))`
throws for such samples. Supply storage before applying the indexed prefix, or through an
argument.

```jldoctest binding_template
julia> using DynamicPPL, Distributions;
       using AbstractPPL: of;
       using StableRNGs: StableRNG;

julia> @model function template_demo()
           y = zeros(3)
           for i in eachindex(y)
               y[i] ~ Normal()
           end
           return y
       end;

julia> m = template_demo();
       template = of(rand(StableRNG(1), m));

julia> condition(m, @varname(y[1]) => 0.5, template)(StableRNG(2))[1]
0.5
```

See **Model ownership** and **Bounds, types and shape** in the
[rule table](@ref binding-rule-table) for binding-time and evaluation-time checks.
[`check_model`](@ref) reports partially bound positions no executed tilde reads. It also warns
about bound names or template entries that no LHS variable in the model or its reached
submodels can use; a child in an untaken branch might still use them.
Binding never selects another model method.

A child binding template is a candidate until the child runs. Arguments keep their own
storage and ignore the candidate. At the parent, `begin`, `end` and `:` resolve only against
already applicable storage in the edited role; a candidate child template cannot resolve them.
Use integer indices for a pending binding, or bind the child directly before `to_submodel`.

For exact partial conversion, `1` becomes `1.0` in `Float64`. Unconvertible values propagate
Julia's conversion error, such as `InexactError` for `1.5` into `Int`; successful but lossy
conversion, such as `0.1` into `Float32`, raises `ArgumentError`. An AD real value that
cannot fit ordinary numeric storage also raises `ArgumentError`, with advice to change the
storage element type. For example, a `Float64` template cannot hold ForwardDiff `Dual` values. For runtime bindings under AD, build storage from the running
value `θ`, using `fill(zero(θ), n)` or `@of(z = of(Array, typeof(θ), n))`, or bind a whole value.

Binding `p[1, 1].x` when the model runs `p[i] ~ to_submodel(...)` needs namespace storage,
such as `@of(p = of(Array, 2, 2))`, to resolve the different index spellings. A pending child
binding instead resolves through the child's own storage: `a.x[2]` reaches `x[2, 1]` when
the child supplies a 2×2 matrix before its body. An all-false Boolean vector still needs its
full source extent, although its selected shape has length zero; `Int[]` needs extent zero.
A pending mask keeps its original extent check while any of its selector survives, including
an all-false mask.
Whole removal or covering same-role replacement cancels the discarded selector's check.

Defaults are evaluated once at construction. Binding `x` in `f(x, n=length(x))` keeps `n`.
Construct the model again to recompute defaults.

## [Performance](@id binding-performance)

Partial bindings rebuild array arguments in O(length) per evaluation. Measured reverse-mode AD
costs (about 110 ns per element) are indicative, not fixed. Bind whole arrays for gradients.

## Missing data

The constructor, explicit-binding and executed-value placeholder rules are in the
[rule table](@ref binding-rule-table), with their phases in the
[entry-point matrix](@ref binding-phase-matrix). See [Binding contract](@ref) for supported
partial storage. For a view containing `missing`, pass `collect(v)` to supply such storage.

For example, `@model metadata_lhs(p) = p.a ~ Normal()` accepts `(a=1.0, b=missing)`;
`(a=missing, b=1.0)` throws on evaluation, naming `p.a`. A multivariate LHS variable such as
`x ~ MvNormal(...)` is latent when all its elements are `missing`, observed when none are,
and rejected for mixed roles when only some are missing.

`conditioned(f((a=1.0, b=missing)))` can list an unread placeholder; passing that listing
back to `condition` rejects it under the explicit-placeholder rule. For an incomplete partial
binding into a whole `missing`/`nothing` argument, supply a concrete argument such as
`f(zeros(n))` or a whole binding instead.

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
