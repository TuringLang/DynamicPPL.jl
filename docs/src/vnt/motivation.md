# Motivation

When executing a DynamicPPL model, it is very often necessary to store information about the random variables in the model.
Consider, e.g.

```@example 1
using DynamicPPL, Distributions

@model function f1()
    x ~ Normal(0, 1)
    y ~ Normal(x, 1)
    z ~ Normal(y, 1)
    return nothing
end
nothing # hide
```

When we evaluate the model, we want to know the values of `x`, `y`, and `z` (even if the return value of the function is `nothing`; the return value is typically used for other purposes).
Sometimes we might want to store other information about them, like their distributions.
This information can be extracted and used by inference algorithms, for example.

In general this is done by storing some kind of mapping from [`AbstractPPL.VarName`s](@extref AbstractPPL varname) to values.
This is the job of `VarNamedTuple`s, or `VNT` for short.
Before we go into that, let's [take a short view back to the past](https://www.youtube.com/watch?v=FlFt_W4664M) to see various approaches that don't quite work.
This will provide a motivation for the design of `VNT`.

# Historical approaches

Julia contains two main data structures for key-value maps: `NamedTuple`s and `Dict`s.

## NamedTuple

`NamedTuple`s are extremely lightweight data structures and are completely type stable.
For example, in the above, we could store the values in a `NamedTuple` like so:

```@example 1
(x=0.5, y=1.2, z=-0.3)
nothing # hide
```

(or whatever the sampled values are.)
The issue with this, however, is that the keys of `NamedTuple`s are symbols.
This is fine for variables that are single identifiers (i.e., VarNames with an optic of `AbstractPPL.Iden`; which we'll call 'identity `VarName`s').

But in a model like the following, we have the VarNames `x[1]`, `x[2]`, `x[3]`, and `y.a`.
These cannot be properly converted into `Symbol`s without loss of information, and this makes the output `NamedTuple` unsuitable for subsequent use.

```@example 1
@model function f2()
    x = Vector{Float64}(undef, 3)
    for i in eachindex(x)
        x[i] ~ Normal()
    end

    y = (; a=1.0)
    return y.a ~ Normal()
end
nothing # hide
```

In the past, there was something called `SimpleVarInfo` which stored variables in `NamedTuple`s (possibly converting non-identity `VarName`s to symbols with loss of information).
This was a very fast data structure, but was unsuitable for general use.

## `Dict`

In order to be _completely_ general, we could use a `Dict{VarName, Any}`.
This allows us to store arbitrary `VarName`s as keys, and arbitrary values.
Again, there was a `SimpleVarInfo` implementation that used this approach (specifically an `OrderedDict`).

```@example 1
Dict(
    @varname(x[1]) => 0.5,
    @varname(x[2]) => -1.2,
    @varname(x[3]) => 1.7,
    @varname(y.a) => 0.3,
)
```

The main issue with Dicts is performance.
In general there is substantial overhead in hashing and looking up keys in a `Dict`, compared to `NamedTuple` field access which is just direct memory access.
Furthermore, accessing values is type-unstable (unless all the values can be made to share the same concrete type, which is the case above, but is unrealistic).

## `Metadata` and `VarNamedVector`

In an attempt to get around the performance issue, DynamicPPL used to have data structures called `Metadata` and `VarNamedVector`.
These structures essentially attempted to remove as much overhead as possible from `Dict`s by flattening their elements (i.e. variable values) and storing vectorised forms of them in contiguous vectors.
Furthermore, they were often wrapped in `NamedTuple`s to get some type stability benefits.
This approach was somewhat successful and was used for many years: it is also described in the [Turing.jl *ACM Transactions on Probabilistic Machine Learning* paper](https://dl.acm.org/doi/10.1145/3711897).

While this fixed many performance problems with `Dict`, at its very core, it is still a mapping of `VarName`s to values, and therefore suffers from a lack of what we call *'constructiveness'*.

## Constructiveness

This is a very subtle issue, but caused a number of headaches with DynamicPPL.
As we will soon see, this is the main motivation behind VNT, and even so VNT does not fully solve it.

Consider the following model:

```@example 1
@model function f3()
    x = Vector{Float64}(undef, 2)
    x[1] ~ Normal()
    return x[2] ~ Normal()
end
```

If at the end of the model we were to ask *what the value of `x` was*, we would have no way of knowing this.
Let's say we are using a `Dict` to store the values.
We would have something like

```@example 1
d = Dict(@varname(x[1]) => 0.5, @varname(x[2]) => -1.2)
```

but attempting to get `d[@varname(x)]` would return `nothing`, since there is no such key in the `Dict`.
In fact, we cannot even access `x[1:2]`, despite both indices obviously being defined.

```@example 1
haskey(d, @varname(x)), haskey(d, @varname(x[1:2]))
```

### What's the problem with this?

One issue that a lack of constructiveness causes is when reading information back from a Chains object.
MCMCChains.jl, for example, breaks up all variables into its constituent scalar components.
Thus, even if you have a multivariate distribution

```@example 1
using LinearAlgebra

@model function f4()
    x ~ MvNormal(zeros(2), I)
    return x
end
nothing # hide
```

sampling from this model would give a chain with keys `x[1]` and `x[2]`.
When calling functions such as `returned` or `predict` on the chain, we have to somehow 'reconstruct' the value of `x` from its components, so that when executing the model we can use the actual value of `x`.

!!! info
    
    Some workarounds exist for this (see [this PR](https://github.com/TuringLang/AbstractPPL.jl/pull/125), and `AbstractPPLDistributionsExt`).
    
    ```@example 1
    using AbstractPPL
    
    no_dist = AbstractPPL.hasvalue(d, @varname(x))
    with_dist = AbstractPPL.hasvalue(d, @varname(x), MvNormal(zeros(2), I))
    
    (no_dist, with_dist)
    ```

Another problem was when conditioning on values.
In such cases, you had to condition on `x` in full, rather than its components:

```julia
model = f4() | Dict(@varname(x) => [1.0, -1.0])  # This would work.
model = f4() | Dict(@varname(x[1]) => 1.0, @varname(x[2]) => -1.0)  # This would not.
```

(This code block isn't run, because with the current version of DynamicPPL and VNT, both versions will work.)

Finally, we are unable to properly use different indexing schemes.
For example, `x[1]` and `x[1,1]` mean the same thing if `x` is a `Matrix`, but the dictionary will think that they are different `VarName`s.
Fundamentally, the issue is that the indexing semantics inside a `Dict` are not the same as the indexing semantics inside the model itself.

## Desiderata

In summary, we want a data structure that:

 1. Can store arbitrary `VarName`s as keys, with arbitrary values.

 2. Is performant and as type stable where possible. In particular, property access should always be type-stable, and indexing should be type-stable as long as the indexed values are homogeneous in type.
 3. Is constructive.

`VarNamedTuple` generally solves these issues, with a few very niche edge cases that will be discussed at the very end.
