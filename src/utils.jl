# singleton for indicating if no default arguments are present
struct NoDefault end
const NO_DEFAULT = NoDefault()

"""
    @addlogprob!(ex)

Add the result of the evaluation of `ex` to the joint log probability.

# Examples

This macro allows you to [include arbitrary terms in the likelihood](https://github.com/TuringLang/Turing.jl/issues/1332)

```jldoctest; setup = :(using Distributions)
julia> myloglikelihood(x, μ) = loglikelihood(Normal(μ, 1), x);

julia> @model function demo(x)
           μ ~ Normal()
           @addlogprob! myloglikelihood(x, μ)
       end;

julia> x = [1.3, -2.1];

julia> loglikelihood(demo(x), (μ=0.2,)) ≈ myloglikelihood(x, 0.2)
true
```

and to [reject samples](https://github.com/TuringLang/Turing.jl/issues/1328):

```jldoctest; setup = :(using Distributions, LinearAlgebra)
julia> @model function demo(x)
           m ~ MvNormal(zero(x), I)
           if dot(m, x) < 0
               @addlogprob! -Inf
               # Exit the model evaluation early
               return
           end
           x ~ MvNormal(m, I)
           return
       end;

julia> logjoint(demo([-2.1]), (m=[0.2],)) == -Inf
true
```

!!! note
    The `@addlogprob!` macro increases the accumulated log probability regardless of the evaluation context,
    i.e., regardless of whether you evaluate the log prior, the log likelihood or the log joint density.
    If you would like to avoid this behaviour you should check the evaluation context.
    It can be accessed with the internal variable `__context__`.
    For instance, in the following example the log density is not accumulated when only the log prior is computed:  
    ```jldoctest; setup = :(using Distributions)
    julia> myloglikelihood(x, μ) = loglikelihood(Normal(μ, 1), x);

    julia> @model function demo(x)
               μ ~ Normal()
               if DynamicPPL.leafcontext(__context__) !== PriorContext()
                   @addlogprob! myloglikelihood(x, μ)
               end
           end;

    julia> x = [1.3, -2.1];

    julia> logprior(demo(x), (μ=0.2,)) ≈ logpdf(Normal(), 0.2)
    true

    julia> loglikelihood(demo(x), (μ=0.2,)) ≈ myloglikelihood(x, 0.2)
    true
    ```
"""
macro addlogprob!(ex)
    return quote
        $(esc(:(__varinfo__))) = acclogp!!(
            $(esc(:(__context__))), $(esc(:(__varinfo__))), $(esc(ex))
        )
    end
end

"""
    addargnames!(args)

Adds names to unnamed arguments in `args`.

The names are generated with `gensym(:arg)` to avoid conflicts with other variable names.

# Examples

```jldoctest; filter = r"var\\"##arg#[0-9]+\\""
julia> args = :(f(x::Int, y, ::Type{T}=Float64)).args[2:end]
3-element Vector{Any}:
 :(x::Int)
 :y
 :($(Expr(:kw, :(::Type{T}), :Float64)))

julia> DynamicPPL.addargnames!(args)

julia> args
3-element Vector{Any}:
 :(x::Int)
 :y
 :($(Expr(:kw, :(var"##arg#301"::Type{T}), :Float64)))
```
"""
function addargnames!(args)
    if isempty(args)
        return nothing
    end

    @inbounds for i in eachindex(args)
        arg = args[i]
        if MacroTools.@capture(arg, ::T_)
            args[i] = Expr(:(::), gensym(:arg), T)
        elseif MacroTools.@capture(arg, ::T_ = val_)
            args[i] = Expr(:kw, Expr(:(::), gensym(:arg), T), val)
        end
    end

    return nothing
end

"""
    getargs_dottilde(x)

Return the arguments `L` and `R`, if `x` is an expression of the form `L .~ R` or
`(~).(L, R)`, or `nothing` otherwise.
"""
getargs_dottilde(x) = nothing
function getargs_dottilde(expr::Expr)
    return MacroTools.@match expr begin
        (.~)(L_, R_) => (L, R)
        (~).(L_, R_) => (L, R)
        # Julia 1.6: see https://github.com/TuringLang/Turing.jl/issues/1525
        (L_ .~ R_) => (L, R)
        x_ => nothing
    end
end

"""
    getargs_tilde(x)

Return the arguments `L` and `R`, if `x` is an expression of the form `L ~ R`, or `nothing`
otherwise.
"""
getargs_tilde(x) = nothing
function getargs_tilde(expr::Expr)
    return MacroTools.@match expr begin
        (~)(L_, R_) => (L, R)
        x_ => nothing
    end
end

"""
    getargs_assignment(x)

Return the arguments `L` and `R`, if `x` is an expression of the form `L = R`, or `nothing`
otherwise.
"""
getargs_assignment(x) = nothing
function getargs_assignment(expr::Expr)
    return MacroTools.@match expr begin
        (L_ = R_) => (L, R)
        x_ => nothing
    end
end

"""
    getargs_coloneq(x)

Return the arguments `L` and `R`, if `x` is an expression of the form `L := R`, or `nothing`
otherwise.
"""
getargs_coloneq(x) = nothing
function getargs_coloneq(expr::Expr)
    return MacroTools.@match expr begin
        (L_ := R_) => (L, R)
        x_ => nothing
    end
end

function to_namedtuple_expr(syms)
    length(syms) == 0 && return :(NamedTuple())

    names_expr = Expr(:tuple, QuoteNode.(syms)...)
    return :(NamedTuple{$names_expr}(($(syms...),)))
end

# FIXME: the prob macro still uses this.
function to_namedtuple_expr(syms, vals)
    length(syms) == 0 && return :(NamedTuple())

    names_expr = Expr(:tuple, QuoteNode.(syms)...)
    vals_expr = Expr(:tuple, vals...)
    return :(NamedTuple{$names_expr}($vals_expr))
end

"""
    link_transform(dist)

Return the constrained-to-unconstrained bijector for distribution `dist`.

By default, this is just `Bijectors.bijector(dist)`.

!!! warning
    Note that currently this is not used by `Bijectors.logpdf_with_trans`,
    hence that needs to be overloaded separately if the intention is
    to change behavior of an existing distribution.
"""
link_transform(dist) = bijector(dist)

"""
    invlink_transform(dist)

Return the unconstrained-to-constrained bijector for distribution `dist`.

By default, this is just `inverse(link_transform(dist))`.

!!! warning
    Note that currently this is not used by `Bijectors.logpdf_with_trans`,
    hence that needs to be overloaded separately if the intention is
    to change behavior of an existing distribution.
"""
invlink_transform(dist) = inverse(link_transform(dist))

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

vectorize(d, r) = vectorize(r)
vectorize(r::Real) = [r]
vectorize(r::AbstractArray{<:Real}) = copy(vec(r))
vectorize(r::Cholesky) = copy(vec(r.UL))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).

"""
    reconstruct([f, ]dist, val)

Reconstruct `val` so that it's compatible with `dist`.

If `f` is also provided, the reconstruct value will be
such that `f(reconstruct_val)` is compatible with `dist`.
"""
reconstruct(f, dist, val) = reconstruct(dist, val)

# No-op versions.
reconstruct(::UnivariateDistribution, val::Real) = val
reconstruct(::MultivariateDistribution, val::AbstractVector{<:Real}) = copy(val)
reconstruct(::MatrixDistribution, val::AbstractMatrix{<:Real}) = copy(val)
function reconstruct(
    ::Distribution{ArrayLikeVariate{N}}, val::AbstractArray{<:Real,N}
) where {N}
    return copy(val)
end
reconstruct(::Inverse{Bijectors.VecCorrBijector}, ::LKJ, val::AbstractVector) = copy(val)

function reconstruct(dist::LKJCholesky, val::AbstractVector{<:Real})
    return reconstruct(dist, Matrix(reshape(val, size(dist))))
end
function reconstruct(dist::LKJCholesky, val::AbstractMatrix{<:Real})
    return Cholesky(val, dist.uplo, 0)
end
reconstruct(::LKJCholesky, val::Cholesky) = val

function reconstruct(
    ::Inverse{Bijectors.VecCholeskyBijector}, ::LKJCholesky, val::AbstractVector
)
    return copy(val)
end

function reconstruct(
    ::Inverse{Bijectors.PDVecBijector}, ::MatrixDistribution, val::AbstractVector
)
    return copy(val)
end

# TODO: Implement no-op `reconstruct` for general array variates.

reconstruct(d::Distribution, val::AbstractVector) = reconstruct(size(d), val)
reconstruct(::Tuple{}, val::AbstractVector) = val[1]
reconstruct(s::NTuple{1}, val::AbstractVector) = copy(val)
reconstruct(s::NTuple{2}, val::AbstractVector) = reshape(copy(val), s)
function reconstruct!(r, d::Distribution, val::AbstractVector)
    return reconstruct!(r, d, val)
end
function reconstruct!(r, d::MultivariateDistribution, val::AbstractVector)
    r .= val
    return r
end
function reconstruct(d::Distribution, val::AbstractVector, n::Int)
    return reconstruct(size(d), val, n)
end
function reconstruct(::Tuple{}, val::AbstractVector, n::Int)
    return copy(val)
end
function reconstruct(s::NTuple{1}, val::AbstractVector, n::Int)
    return copy(reshape(val, s[1], n))
end
function reconstruct(s::NTuple{2}, val::AbstractVector, n::Int)
    tmp = reshape(val, s..., n)
    orig = [tmp[:, :, i] for i in 1:n]
    return orig
end
function reconstruct!(r, d::Distribution, val::AbstractVector, n::Int)
    return reconstruct!(r, d, val, n)
end
function reconstruct!(r, d::MultivariateDistribution, val::AbstractVector, n::Int)
    r .= val
    return r
end

# Uniform random numbers with range 4 for robust initializations
# Reference: https://mc-stan.org/docs/2_19/reference-manual/initialization.html
randrealuni(rng::Random.AbstractRNG) = 4 * rand(rng) - 2
randrealuni(rng::Random.AbstractRNG, args...) = 4 .* rand(rng, args...) .- 2

istransformable(dist) = link_transform(dist) !== identity

#################################
# Single-sample initialisations #
#################################

inittrans(rng, dist::UnivariateDistribution) = Bijectors.invlink(dist, randrealuni(rng))
function inittrans(rng, dist::MultivariateDistribution)
    # Get the length of the unconstrained vector
    b = link_transform(dist)
    d = Bijectors.output_length(b, length(dist))
    return Bijectors.invlink(dist, randrealuni(rng, d))
end
function inittrans(rng, dist::MatrixDistribution)
    # Get the size of the unconstrained vector
    b = link_transform(dist)
    sz = Bijectors.output_size(b, size(dist))
    return Bijectors.invlink(dist, randrealuni(rng, sz...))
end
function inittrans(rng, dist::Distribution{CholeskyVariate})
    # Get the size of the unconstrained vector
    b = link_transform(dist)
    sz = Bijectors.output_size(b, size(dist))
    return Bijectors.invlink(dist, randrealuni(rng, sz...))
end
################################
# Multi-sample initialisations #
################################

function inittrans(rng, dist::UnivariateDistribution, n::Int)
    return Bijectors.invlink(dist, randrealuni(rng, n))
end
function inittrans(rng, dist::MultivariateDistribution, n::Int)
    return Bijectors.invlink(dist, randrealuni(rng, size(dist)[1], n))
end
function inittrans(rng, dist::MatrixDistribution, n::Int)
    return Bijectors.invlink(dist, [randrealuni(rng, size(dist)...) for _ in 1:n])
end

#######################
# Convenience methods #
#######################
collectmaybe(x) = x
collectmaybe(x::Base.AbstractSet) = collect(x)

#######################
# BangBang.jl related #
#######################
function set!!(obj, optic::AbstractPPL.ALLOWED_OPTICS, value)
    opticmut = BangBang.prefermutation(optic)
    return Accessors.set(obj, opticmut, value)
end
function set!!(obj, vn::VarName{sym}, value) where {sym}
    optic = BangBang.prefermutation(
        AbstractPPL.getoptic(vn) ∘ Accessors.PropertyLens{sym}()
    )
    return Accessors.set(obj, optic, value)
end

#############################
# AbstractPPL.jl extensions #
#############################
# This is preferable to `haskey` because the order of arguments is different, and
# we're more likely to specialize on the key in these settings rather than the container.
# TODO: I'm not sure about this name.
"""
    canview(optic, container)

Return `true` if `optic` can be used to view `container`, and `false` otherwise.

# Examples
```jldoctest; setup=:(using Accessors; using DynamicPPL: canview)
julia> canview(@o(_.a), (a = 1.0, ))
true

julia> canview(@o(_.a), (b = 1.0, )) # property `a` does not exist
false

julia> canview(@o(_.a[1]), (a = [1.0, 2.0], ))
true

julia> canview(@o(_.a[3]), (a = [1.0, 2.0], )) # out of bounds
false
```
"""
canview(optic, container) = false
canview(::typeof(identity), _) = true
function canview(optic::Accessors.PropertyLens{field}, x) where {field}
    return hasproperty(x, field)
end

# `IndexLens`: only relevant if `x` supports indexing.
canview(optic::Accessors.IndexLens, x) = false
function canview(optic::Accessors.IndexLens, x::AbstractArray)
    return checkbounds(Bool, x, optic.indices...)
end

# `ComposedOptic`: check that we can view `.inner` and `.outer`, but using
# value extracted using `.inner`.
function canview(optic::Accessors.ComposedOptic, x)
    return canview(optic.inner, x) && canview(optic.outer, optic.inner(x))
end

"""
    parent(vn::VarName)

Return the parent `VarName`.

# Examples
```julia-repl; setup=:(using DynamicPPL: parent)
julia> parent(@varname(x.a[1]))
x.a

julia> (parent ∘ parent)(@varname(x.a[1]))
x

julia> (parent ∘ parent ∘ parent)(@varname(x.a[1]))
x
```
"""
function parent(vn::VarName)
    p = parent(getoptic(vn))
    return p === nothing ? VarName(vn, identity) : VarName(vn, p)
end

"""
    parent(optic)

Return the parent optic. If `optic` doesn't have a parent,
`nothing` is returned.

See also: [`parent_and_child`].

# Examples
```jldoctest; setup=:(using Accessors; using DynamicPPL: parent)
julia> parent(@o(_.a[1]))
(@o _.a)

julia> # Parent of optic without parents results in `nothing`.
       (parent ∘ parent)(@o(_.a[1])) === nothing
true
```
"""
parent(optic::AbstractPPL.ALLOWED_OPTICS) = first(parent_and_child(optic))

"""
    parent_and_child(optic)

Return a 2-tuple of optics `(parent, child)` where `parent` is the
parent optic of `optic` and `child` is the child optic of `optic`.

If `optic` does not have a parent, we return `(nothing, optic)`.

See also: [`parent`].

# Examples
```jldoctest; setup=:(using Accessors; using DynamicPPL: parent_and_child)
julia> parent_and_child(@o(_.a[1]))
((@o _.a), (@o _[1]))

julia> parent_and_child(@o(_.a))
(nothing, (@o _.a))
```
"""
parent_and_child(optic::AbstractPPL.ALLOWED_OPTICS) = (nothing, optic)
function parent_and_child(optic::Accessors.ComposedOptic)
    p, child = parent_and_child(optic.outer)
    parent = p === nothing ? optic.inner : p ∘ optic.inner
    return parent, child
end

"""
    splitoptic(condition, optic)

Return a 3-tuple `(parent, child, issuccess)` where, if `issuccess` is `true`,
`parent` is a optic such that `condition(parent)` is `true` and `child ∘ parent == optic`.

If `issuccess` is `false`, then no such split could be found.

# Examples
```jldoctest; setup=:(using Accessors; using DynamicPPL: splitoptic)
julia> p, c, issucesss = splitoptic(@o(_.a[1])) do parent
           # Succeeds!
           parent == @o(_.a)
       end
((@o _.a), (@o _[1]), true)

julia> c ∘ p
(@o _.a[1])

julia> splitoptic(@o(_.a[1])) do parent
           # Fails!
           parent == @o(_.b)
       end
(nothing, (@o _.a[1]), false)
```
"""
function splitoptic(condition, optic)
    current_parent, current_child = parent_and_child(optic)
    # We stop if either a) `condition` is satisfied, or b) we reached the root.
    while !condition(current_parent) && current_parent !== nothing
        current_parent, c = parent_and_child(current_parent)
        current_child = current_child ∘ c
    end

    return current_parent, current_child, condition(current_parent)
end

"""
    remove_parent_optic(vn_parent::VarName, vn_child::VarName)

Remove the parent optic `vn_parent` from `vn_child`.

# Examples
```jldoctest; setup = :(using Accessors; using DynamicPPL: remove_parent_optic)
julia> remove_parent_optic(@varname(x), @varname(x.a))
(@o _.a)

julia> remove_parent_optic(@varname(x), @varname(x.a[1]))
(@o _.a[1])

julia> remove_parent_optic(@varname(x.a), @varname(x.a[1]))
(@o _[1])

julia> remove_parent_optic(@varname(x.a), @varname(x.a[1].b))
(@o _[1].b)

julia> remove_parent_optic(@varname(x.a), @varname(x.a))
ERROR: Could not find x.a in x.a

julia> remove_parent_optic(@varname(x.a[2]), @varname(x.a[1]))
ERROR: Could not find x.a[2] in x.a[1]
```
"""
function remove_parent_optic(vn_parent::VarName{sym}, vn_child::VarName{sym}) where {sym}
    _, child, issuccess = splitoptic(getoptic(vn_child)) do optic
        o = optic === nothing ? identity : optic
        VarName(vn_child, o) == vn_parent
    end

    issuccess || error("Could not find $vn_parent in $vn_child")
    return child
end

# HACK(torfjelde): This makes it so it works on iterators, etc. by default.
# TODO(torfjelde): Do better.
"""
    unflatten(original, x::AbstractVector)

Return instance of `original` constructed from `x`.
"""
function unflatten(original, x::AbstractVector)
    lengths = map(length, original)
    end_indices = cumsum(lengths)
    return map(zip(original, lengths, end_indices)) do (v, l, end_idx)
        start_idx = end_idx - l + 1
        return unflatten(v, @view(x[start_idx:end_idx]))
    end
end

unflatten(::Real, x::Real) = x
unflatten(::Real, x::AbstractVector) = only(x)
unflatten(::AbstractVector{<:Real}, x::Real) = vcat(x)
unflatten(::AbstractVector{<:Real}, x::AbstractVector) = x
unflatten(original::AbstractArray{<:Real}, x::AbstractVector) = reshape(x, size(original))

function unflatten(original::Tuple, x::AbstractVector)
    lengths = map(length, original)
    end_indices = cumsum(lengths)
    return ntuple(length(original)) do i
        v = original[i]
        l = lengths[i]
        end_idx = end_indices[i]
        start_idx = end_idx - l + 1
        return unflatten(v, @view(x[start_idx:end_idx]))
    end
end
function unflatten(original::NamedTuple{names}, x::AbstractVector) where {names}
    return NamedTuple{names}(unflatten(values(original), x))
end
function unflatten(original::AbstractDict, x::AbstractVector)
    D = ConstructionBase.constructorof(typeof(original))
    return D(zip(keys(original), unflatten(collect(values(original)), x)))
end

# TODO: Move `getvalue` and `hasvalue` to AbstractPPL.jl.
"""
    getvalue(vals, vn::VarName)

Return the value(s) in `vals` represented by `vn`.

Note that this method is different from `getindex`. See examples below.

# Examples

For `NamedTuple`:

```jldoctest
julia> vals = (x = [1.0],);

julia> DynamicPPL.getvalue(vals, @varname(x)) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> DynamicPPL.getvalue(vals, @varname(x[1])) # different from `getindex`
1.0

julia> DynamicPPL.getvalue(vals, @varname(x[2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]
```

For `AbstractDict`:

```jldoctest
julia> vals = Dict(@varname(x) => [1.0]);

julia> DynamicPPL.getvalue(vals, @varname(x)) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> DynamicPPL.getvalue(vals, @varname(x[1])) # different from `getindex`
1.0

julia> DynamicPPL.getvalue(vals, @varname(x[2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]
```

In the `AbstractDict` case we can also have keys such as `v[1]`:

```jldoctest
julia> vals = Dict(@varname(x[1]) => [1.0,]);

julia> DynamicPPL.getvalue(vals, @varname(x[1])) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> DynamicPPL.getvalue(vals, @varname(x[1][1])) # different from `getindex`
1.0

julia> DynamicPPL.getvalue(vals, @varname(x[1][2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]

julia> DynamicPPL.getvalue(vals, @varname(x[2][1]))
ERROR: KeyError: key x[2][1] not found
[...]
```
"""
getvalue(vals::NamedTuple, vn::VarName) = get(vals, vn)
getvalue(vals::AbstractDict, vn::VarName) = nested_getindex(vals, vn)

"""
    hasvalue(vals, vn::VarName)

Determine whether `vals` has a mapping for a given `vn`, as compatible with [`getvalue`](@ref).

# Examples
With `x` as a `NamedTuple`:

```jldoctest
julia> DynamicPPL.hasvalue((x = 1.0, ), @varname(x))
true

julia> DynamicPPL.hasvalue((x = 1.0, ), @varname(x[1]))
false

julia> DynamicPPL.hasvalue((x = [1.0],), @varname(x))
true

julia> DynamicPPL.hasvalue((x = [1.0],), @varname(x[1]))
true

julia> DynamicPPL.hasvalue((x = [1.0],), @varname(x[2]))
false
```

With `x` as a `AbstractDict`:

```jldoctest
julia> DynamicPPL.hasvalue(Dict(@varname(x) => 1.0, ), @varname(x))
true

julia> DynamicPPL.hasvalue(Dict(@varname(x) => 1.0, ), @varname(x[1]))
false

julia> DynamicPPL.hasvalue(Dict(@varname(x) => [1.0]), @varname(x))
true

julia> DynamicPPL.hasvalue(Dict(@varname(x) => [1.0]), @varname(x[1]))
true

julia> DynamicPPL.hasvalue(Dict(@varname(x) => [1.0]), @varname(x[2]))
false
```

In the `AbstractDict` case we can also have keys such as `v[1]`:

```jldoctest
julia> vals = Dict(@varname(x[1]) => [1.0,]);

julia> DynamicPPL.hasvalue(vals, @varname(x[1])) # same as `haskey`
true

julia> DynamicPPL.hasvalue(vals, @varname(x[1][1])) # different from `haskey`
true

julia> DynamicPPL.hasvalue(vals, @varname(x[1][2]))
false

julia> DynamicPPL.hasvalue(vals, @varname(x[2][1]))
false
```
"""
function hasvalue(vals::NamedTuple, vn::VarName{sym}) where {sym}
    # LHS: Ensure that `nt` indeed has the property we want.
    # RHS: Ensure that the optic can view into `nt`.
    return haskey(vals, sym) && canview(getoptic(vn), getproperty(vals, sym))
end

# For `dictlike` we need to check wether `vn` is "immediately" present, or
# if some ancestor of `vn` is present in `dictlike`.
function hasvalue(vals::AbstractDict, vn::VarName)
    # First we check if `vn` is present as is.
    haskey(vals, vn) && return true

    # If `vn` is not present, we check any parent-varnames by attempting
    # to split the optic into the key / `parent` and the extraction optic / `child`.
    # If `issuccess` is `true`, we found such a split, and hence `vn` is present.
    parent, child, issuccess = splitoptic(getoptic(vn)) do optic
        o = optic === nothing ? identity : optic
        haskey(vals, VarName(vn, o))
    end
    # When combined with `VarInfo`, `nothing` is equivalent to `identity`.
    keyoptic = parent === nothing ? identity : parent

    # Return early if no such split could be found.
    issuccess || return false

    # At this point we just need to check that we `canview` the value.
    value = vals[VarName(vn, keyoptic)]

    return canview(child, value)
end

"""
    nested_getindex(values::AbstractDict, vn::VarName)

Return value corresponding to `vn` in `values` by also looking
in the the actual values of the dict.
"""
function nested_getindex(values::AbstractDict, vn::VarName)
    maybeval = get(values, vn, nothing)
    if maybeval !== nothing
        return maybeval
    end

    # Split the optic into the key / `parent` and the extraction optic / `child`.
    parent, child, issuccess = splitoptic(getoptic(vn)) do optic
        o = optic === nothing ? identity : optic
        haskey(values, VarName(vn, o))
    end
    # When combined with `VarInfo`, `nothing` is equivalent to `identity`.
    keyoptic = parent === nothing ? identity : parent

    # If we found a valid split, then we can extract the value.
    if !issuccess
        # At this point we just throw an error since the key could not be found.
        throw(KeyError(vn))
    end

    # TODO: Should we also check that we `canview` the extracted `value`
    # rather than just let it fail upon `get` call?
    value = values[VarName(vn, keyoptic)]
    return child(value)
end

"""
    update_values!!(vi::AbstractVarInfo, vals::NamedTuple, vns)

Return instance similar to `vi` but with `vns` set to values from `vals`.
"""
function update_values!!(vi::AbstractVarInfo, vals::NamedTuple, vns)
    for vn in vns
        vi = DynamicPPL.setindex!!(vi, get(vals, vn), vn)
    end
    return vi
end

"""
    float_type_with_fallback(x)

Return type corresponding to `float(typeof(x))` if possible; otherwise return `Real`.
"""
float_type_with_fallback(::Type) = Real
float_type_with_fallback(::Type{T}) where {T<:Real} = float(T)

"""
    infer_nested_eltype(x::Type)

Recursively unwrap the type, returning the first type where `eltype(x) === typeof(x)`.

This is useful for obtaining a reasonable default `eltype` in deeply nested types.

# Examples
```jldoctest
julia> # `AbstractArrary`
       DynamicPPL.infer_nested_eltype(typeof([1.0]))
Float64

julia> # `NamedTuple` with `Float32`
       DynamicPPL.infer_nested_eltype(typeof((x = [1f0], )))
Float32

julia> # `AbstractDict`
       DynamicPPL.infer_nested_eltype(typeof(Dict(:x => [1.0, ])))
Float64

julia> # Nesting of containers.
       DynamicPPL.infer_nested_eltype(typeof([Dict(:x => 1.0,) ]))
Float64

julia> DynamicPPL.infer_nested_eltype(typeof([Dict(:x => [1.0,],) ]))
Float64

julia> # Empty `Tuple`.
       DynamicPPL.infer_nested_eltype(typeof(()))
Any

julia> # Empty `Dict`.
       DynamicPPL.infer_nested_eltype(typeof(Dict()))
Any
```
"""
function infer_nested_eltype(::Type{T}) where {T}
    ET = eltype(T)
    return ET === T ? T : infer_nested_eltype(ET)
end

# We can do a better job than just `Any` with `Union`.
infer_nested_eltype(::Type{Union{}}) = Any
function infer_nested_eltype(::Type{U}) where {U<:Union}
    return promote_type(U.a, infer_nested_eltype(U.b))
end

# Handle `NamedTuple` and `Tuple` specially given how prolific they are.
function infer_nested_eltype(::Type{<:NamedTuple{<:Any,V}}) where {V}
    return infer_nested_eltype(V)
end

# Recursively deal with `Tuple` so it has the potential of being compiled away.
infer_nested_eltype(::Type{Tuple{T}}) where {T} = infer_nested_eltype(T)
function infer_nested_eltype(::Type{T}) where {T<:Tuple{<:Any,Vararg{Any}}}
    return promote_type(
        infer_nested_eltype(Base.tuple_type_tail(T)),
        infer_nested_eltype(Base.tuple_type_head(T)),
    )
end

# Handle `AbstractDict` differently since `eltype` results in a `Pair`.
infer_nested_eltype(::Type{<:AbstractDict{<:Any,ET}}) where {ET} = infer_nested_eltype(ET)

"""
    varname_leaves(vn::VarName, val)

Return an iterator over all varnames that are represented by `vn` on `val`.

# Examples
```jldoctest
julia> using DynamicPPL: varname_leaves

julia> foreach(println, varname_leaves(@varname(x), rand(2)))
x[1]
x[2]

julia> foreach(println, varname_leaves(@varname(x[1:2]), rand(2)))
x[1:2][1]
x[1:2][2]

julia> x = (y = 1, z = [[2.0], [3.0]]);

julia> foreach(println, varname_leaves(@varname(x), x))
x.y
x.z[1][1]
x.z[2][1]
```
"""
varname_leaves(vn::VarName, ::Real) = [vn]
function varname_leaves(vn::VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (
        VarName(vn, Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_leaves(VarName(vn, Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)), val[I])
        for I in CartesianIndices(val)
    )
end
function varname_leaves(vn::VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        optic = Accessors.PropertyLens{sym}()
        varname_leaves(VarName(vn, optic ∘ getoptic(vn)), optic(val))
    end
    return Iterators.flatten(iter)
end

"""
    varname_and_value_leaves(vn::VarName, val)

Return an iterator over all varname-value pairs that are represented by `vn` on `val`.

# Examples
```jldoctest varname-and-value-leaves
julia> using DynamicPPL: varname_and_value_leaves

julia> foreach(println, varname_and_value_leaves(@varname(x), 1:2))
(x[1], 1)
(x[2], 2)

julia> foreach(println, varname_and_value_leaves(@varname(x[1:2]), 1:2))
(x[1:2][1], 1)
(x[1:2][2], 2)

julia> x = (y = 1, z = [[2.0], [3.0]]);

julia> foreach(println, varname_and_value_leaves(@varname(x), x))
(x.y, 1)
(x.z[1][1], 2.0)
(x.z[2][1], 3.0)
```

There are also some special handling for certain types:

```jldoctest varname-and-value-leaves
julia> using LinearAlgebra

julia> x = reshape(1:4, 2, 2);

julia> # `LowerTriangular`
       foreach(println, varname_and_value_leaves(@varname(x), LowerTriangular(x)))
(x[1, 1], 1)
(x[2, 1], 2)
(x[2, 2], 4)

julia> # `UpperTriangular`
       foreach(println, varname_and_value_leaves(@varname(x), UpperTriangular(x)))
(x[1, 1], 1)
(x[1, 2], 3)
(x[2, 2], 4)

julia> # `Cholesky` with lower-triangular
       foreach(println, varname_and_value_leaves(@varname(x), Cholesky([1.0 0.0; 0.0 1.0], 'L', 0)))
(x.L[1, 1], 1.0)
(x.L[2, 1], 0.0)
(x.L[2, 2], 1.0)

julia> # `Cholesky` with upper-triangular
       foreach(println, varname_and_value_leaves(@varname(x), Cholesky([1.0 0.0; 0.0 1.0], 'U', 0)))
(x.U[1, 1], 1.0)
(x.U[1, 2], 0.0)
(x.U[2, 2], 1.0)
```
"""
function varname_and_value_leaves(vn::VarName, x)
    return Iterators.map(value, Iterators.flatten(varname_and_value_leaves_inner(vn, x)))
end

"""
    Leaf{T}

A container that represents the leaf of a nested structure, implementing
`iterate` to return itself.

This is particularly useful in conjunction with `Iterators.flatten` to
prevent flattening of nested structures.
"""
struct Leaf{T}
    value::T
end

Leaf(xs...) = Leaf(xs)

# Allow us to treat `Leaf` as an iterator containing a single element.
# Something like an `[x]` would also be an iterator with a single element,
# but when we call `flatten` on this, it would also iterate over `x`,
# unflattening that too. By making `Leaf` a single-element iterator, which
# returns itself, we can call `iterate` on this as many times as we like
# without causing any change. The result is that `Iterators.flatten`
# will _not_ unflatten `Leaf`s.
# Note that this is similar to how `Base.iterate` is implemented for `Real`::
#
#    julia> iterate(1)
#    (1, nothing)
#
# One immediate example where this becomes in our scenario is that we might
# have `missing` values in our data, which does _not_ have an `iterate`
# implemented. Calling `Iterators.flatten` on this would cause an error.
Base.iterate(leaf::Leaf) = leaf, nothing
Base.iterate(::Leaf, _) = nothing

# Convenience.
value(leaf::Leaf) = leaf.value

# Leaf-types.
varname_and_value_leaves_inner(vn::VarName, x::Real) = [Leaf(vn, x)]
function varname_and_value_leaves_inner(
    vn::VarName, val::AbstractArray{<:Union{Real,Missing}}
)
    return (
        Leaf(
            VarName(vn, DynamicPPL.Accessors.IndexLens(Tuple(I)) ∘ DynamicPPL.getoptic(vn)),
            val[I],
        ) for I in CartesianIndices(val)
    )
end
# Containers.
function varname_and_value_leaves_inner(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_and_value_leaves_inner(
            VarName(vn, DynamicPPL.Accessors.IndexLens(Tuple(I)) ∘ DynamicPPL.getoptic(vn)),
            val[I],
        ) for I in CartesianIndices(val)
    )
end
function varname_and_value_leaves_inner(vn::DynamicPPL.VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        optic = DynamicPPL.Accessors.PropertyLens{sym}()
        varname_and_value_leaves_inner(
            VarName{getsym(vn)}(optic ∘ getoptic(vn)), optic(val)
        )
    end

    return Iterators.flatten(iter)
end
# Special types.
function varname_and_value_leaves_inner(vn::VarName, x::Cholesky)
    # TODO: Or do we use `PDMat` here?
    return if x.uplo == 'L'
        varname_and_value_leaves_inner(Accessors.PropertyLens{:L}() ∘ vn, x.L)
    else
        varname_and_value_leaves_inner(Accessors.PropertyLens{:U}() ∘ vn, x.U)
    end
end
function varname_and_value_leaves_inner(vn::VarName, x::LinearAlgebra.LowerTriangular)
    return (
        Leaf(
            VarName(vn, DynamicPPL.Accessors.IndexLens(Tuple(I)) ∘ DynamicPPL.getoptic(vn)),
            x[I],
        )
        # Iteration over the lower-triangular indices.
        for I in CartesianIndices(x) if I[1] >= I[2]
    )
end
function varname_and_value_leaves_inner(vn::VarName, x::LinearAlgebra.UpperTriangular)
    return (
        Leaf(
            VarName(vn, DynamicPPL.Accessors.IndexLens(Tuple(I)) ∘ DynamicPPL.getoptic(vn)),
            x[I],
        )
        # Iteration over the upper-triangular indices.
        for I in CartesianIndices(x) if I[1] <= I[2]
    )
end

broadcast_safe(x) = x
broadcast_safe(x::Distribution) = (x,)
broadcast_safe(x::AbstractContext) = (x,)

# Version of `merge` used by `conditioned` and `fixed` to handle
# the scenario where we might try to merge a dict with an empty
# tuple.
# TODO: Maybe replace the default of returning `NamedTuple` with `nothing`?
_merge(left::NamedTuple, right::NamedTuple) = merge(left, right)
_merge(left::AbstractDict, right::AbstractDict) = merge(left, right)
_merge(left::AbstractDict, right::NamedTuple{()}) = left
_merge(left::NamedTuple{()}, right::AbstractDict) = right
