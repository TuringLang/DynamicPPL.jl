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
        $(esc(:(__varinfo__))) = acclogp!!($(esc(:(__varinfo__))), $(esc(ex)))
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

function to_namedtuple_expr(syms, vals=syms)
    length(syms) == 0 && return :(NamedTuple())

    names_expr = Expr(:tuple, QuoteNode.(syms)...)
    vals_expr = Expr(:tuple, vals...)
    return :(NamedTuple{$names_expr}($vals_expr))
end

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

vectorize(d::UnivariateDistribution, r::Real) = [r]
vectorize(d::MultivariateDistribution, r::AbstractVector{<:Real}) = copy(r)
vectorize(d::MatrixDistribution, r::AbstractMatrix{<:Real}) = copy(vec(r))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).
reconstruct(d::UnivariateDistribution, val::Real) = val
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

const Transformable = Union{
    PositiveDistribution,
    UnitDistribution,
    TransformDistribution,
    SimplexDistribution,
    PDMatDistribution,
}
istransformable(dist) = false
istransformable(::Transformable) = true

#################################
# Single-sample initialisations #
#################################

inittrans(rng, dist::UnivariateDistribution) = invlink(dist, randrealuni(rng))
function inittrans(rng, dist::MultivariateDistribution)
    return invlink(dist, randrealuni(rng, size(dist)[1]))
end
inittrans(rng, dist::MatrixDistribution) = invlink(dist, randrealuni(rng, size(dist)...))

################################
# Multi-sample initialisations #
################################

inittrans(rng, dist::UnivariateDistribution, n::Int) = invlink(dist, randrealuni(rng, n))
function inittrans(rng, dist::MultivariateDistribution, n::Int)
    return invlink(dist, randrealuni(rng, size(dist)[1], n))
end
function inittrans(rng, dist::MatrixDistribution, n::Int)
    return invlink(dist, [randrealuni(rng, size(dist)...) for _ in 1:n])
end

#######################
# Convenience methods #
#######################
collectmaybe(x) = x
collectmaybe(x::Base.AbstractSet) = collect(x)

#######################
# BangBang.jl related #
#######################
function set!!(obj, lens::Setfield.Lens, value)
    lensmut = BangBang.prefermutation(lens)
    return Setfield.set(obj, lensmut, value)
end
function set!!(obj, vn::VarName{sym}, value) where {sym}
    lens = BangBang.prefermutation(Setfield.PropertyLens{sym}() ∘ AbstractPPL.getlens(vn))
    return Setfield.set(obj, lens, value)
end

#############################
# AbstractPPL.jl extensions #
#############################
# This is preferable to `haskey` because the order of arguments is different, and
# we're more likely to specialize on the key in these settings rather than the container.
# TODO: I'm not sure about this name.
"""
    canview(lens, container)

Return `true` if `lens` can be used to view `container`, and `false` otherwise.

# Examples
```jldoctest; setup=:(using Setfield; using DynamicPPL: canview)
julia> canview(@lens(_.a), (a = 1.0, ))
true

julia> canview(@lens(_.a), (b = 1.0, )) # property `a` does not exist
false

julia> canview(@lens(_.a[1]), (a = [1.0, 2.0], ))
true

julia> canview(@lens(_.a[3]), (a = [1.0, 2.0], )) # out of bounds
false
```
"""
canview(lens, container) = false
canview(::Setfield.IdentityLens, _) = true
function canview(lens::Setfield.PropertyLens{field}, x) where {field}
    return hasproperty(x, field)
end

# `IndexLens`: only relevant if `x` supports indexing.
canview(lens::Setfield.IndexLens, x) = false
canview(lens::Setfield.IndexLens, x::AbstractArray) = checkbounds(Bool, x, lens.indices...)

# `ComposedLens`: check that we can view `.outer` and `.inner`, but using
# value extracted using `.outer`.
function canview(lens::Setfield.ComposedLens, x)
    return canview(lens.outer, x) && canview(lens.inner, get(x, lens.outer))
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
    p = parent(getlens(vn))
    return p === nothing ? VarName(vn, Setfield.IdentityLens()) : VarName(vn, p)
end

"""
    parent(lens::Setfield.Lens)

Return the parent lens. If `lens` doesn't have a parent,
`nothing` is returned.

See also: [`parent_and_child`].

# Examples
```jldoctest; setup=:(using Setfield; using DynamicPPL: parent)
julia> parent(@lens(_.a[1]))
(@lens _.a)

julia> # Parent of lens without parents results in `nothing`.
       (parent ∘ parent)(@lens(_.a[1])) === nothing
true
```
"""
parent(lens::Setfield.Lens) = first(parent_and_child(lens))

"""
    parent_and_child(lens::Setfield.Lens)

Return a 2-tuple of lenses `(parent, child)` where `parent` is the
parent lens of `lens` and `child` is the child lens of `lens`.

If `lens` does not have a parent, we return `(nothing, lens)`.

See also: [`parent`].

# Examples
```jldoctest; setup=:(using Setfield; using DynamicPPL: parent_and_child)
julia> parent_and_child(@lens(_.a[1]))
((@lens _.a), (@lens _[1]))

julia> parent_and_child(@lens(_.a))
(nothing, (@lens _.a))
```
"""
parent_and_child(lens::Setfield.Lens) = (nothing, lens)
function parent_and_child(lens::Setfield.ComposedLens)
    p, child = parent_and_child(lens.inner)
    parent = p === nothing ? lens.outer : lens.outer ∘ p
    return parent, child
end

"""
    splitlens(condition, lens)

Return a 3-tuple `(parent, child, issuccess)` where, if `issuccess` is `true`,
`parent` is a lens such that `condition(parent)` is `true` and `parent ∘ child == lens`.

If `issuccess` is `false`, then no such split could be found.

# Examples
```jldoctest; setup=:(using Setfield; using DynamicPPL: splitlens)
julia> p, c, issucesss = splitlens(@lens(_.a[1])) do parent
           # Succeeds!
           parent == @lens(_.a)
       end
((@lens _.a), (@lens _[1]), true)

julia> p ∘ c
(@lens _.a[1])

julia> splitlens(@lens(_.a[1])) do parent
           # Fails!
           parent == @lens(_.b)
       end
(nothing, (@lens _.a[1]), false)
```
"""
function splitlens(condition, lens)
    current_parent, current_child = parent_and_child(lens)
    # We stop if either a) `condition` is satisfied, or b) we reached the root.
    while !condition(current_parent) && current_parent !== nothing
        current_parent, c = parent_and_child(current_parent)
        current_child = c ∘ current_child
    end

    return current_parent, current_child, condition(current_parent)
end

# HACK(torfjelde): Avoids type-instability in `dot_assume` for `SimpleVarInfo`.
function BangBang.possible(
    ::typeof(BangBang._setindex!), ::C, ::T, ::Colon, ::Integer
) where {C<:AbstractMatrix,T<:AbstractVector}
    return BangBang.implements(setindex!, C) &&
           promote_type(eltype(C), eltype(T)) <: eltype(C)
end

"""
    nested_getindex(values::AbstractDict, vn::VarName)

Return value corresponding to `vn` in `values` by also looking
in the the actual values of the dict.

# Examples

```jldoctest
julia> DynamicPPL.nested_getindex(Dict(@varname(x) => [1.0]), @varname(x)) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> DynamicPPL.nested_getindex(Dict(@varname(x) => [1.0]), @varname(x[1])) # different from `getindex`
1.0

julia> DynamicPPL.nested_getindex(Dict(@varname(x) => [1.0]), @varname(x[2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]
```
"""
function nested_getindex(values::AbstractDict, vn::VarName)
    maybeval = get(values, vn, nothing)
    if maybeval !== nothing
        return maybeval
    end

    # Split the lens into the key / `parent` and the extraction lens / `child`.
    parent, child, issuccess = splitlens(getlens(vn)) do lens
        l = lens === nothing ? Setfield.IdentityLens() : lens
        haskey(values, VarName(vn, l))
    end
    # When combined with `VarInfo`, `nothing` is equivalent to `IdentityLens`.
    keylens = parent === nothing ? Setfield.IdentityLens() : parent

    # If we found a valid split, then we can extract the value.
    if !issuccess
        # At this point we just throw an error since the key could not be found.
        throw(KeyError(vn))
    end

    # TODO: Should we also check that we `canview` the extracted `value`
    # rather than just let it fail upon `get` call?
    value = values[VarName(vn, keylens)]
    return get(value, child)
end

"""
    nested_haskey(x, vn::VarName)

Determine whether `x` has a mapping for a given `vn`.

# Examples
With `x` as a `NamedTuple`:
```jldoctest
julia> DynamicPPL.nested_haskey((x = 1.0, ), @varname(x))
true

julia> DynamicPPL.nested_haskey((x = 1.0, ), @varname(x[1]))
false

julia> DynamicPPL.nested_haskey((x = [1.0],), @varname(x))
true

julia> DynamicPPL.nested_haskey((x = [1.0],), @varname(x[1]))
true

julia> DynamicPPL.nested_haskey((x = [1.0],), @varname(x[2]))
false
```

With `x` as a `AbstractDict`:
```jldoctest
julia> DynamicPPL.nested_haskey(Dict(@varname(x) => 1.0, ), @varname(x))
true

julia> DynamicPPL.nested_haskey(Dict(@varname(x) => 1.0, ), @varname(x[1]))
false

julia> DynamicPPL.nested_haskey(Dict(@varname(x) => [1.0]), @varname(x))
true

julia> DynamicPPL.nested_haskey(Dict(@varname(x) => [1.0]), @varname(x[1]))
true

julia> DynamicPPL.nested_haskey(Dict(@varname(x) => [1.0]), @varname(x[2]))
false
```
"""
function nested_haskey(nt::NamedTuple, vn::VarName{sym}) where {sym}
    # LHS: Ensure that `nt` indeed has the property we want.
    # RHS: Ensure that the lens can view into `nt`.
    return haskey(nt, sym) && canview(getlens(vn), getproperty(nt, sym))
end

# For `dictlike` we need to check wether `vn` is "immediately" present, or
# if some ancestor of `vn` is present in `dictlike`.
function nested_haskey(dict::AbstractDict, vn::VarName)
    # First we check if `vn` is present as is.
    haskey(dict, vn) && return true

    # If `vn` is not present, we check any parent-varnames by attempting
    # to split the lens into the key / `parent` and the extraction lens / `child`.
    # If `issuccess` is `true`, we found such a split, and hence `vn` is present.
    parent, child, issuccess = splitlens(getlens(vn)) do lens
        l = lens === nothing ? Setfield.IdentityLens() : lens
        haskey(dict, VarName(vn, l))
    end
    # When combined with `VarInfo`, `nothing` is equivalent to `IdentityLens`.
    keylens = parent === nothing ? Setfield.IdentityLens() : parent

    # Return early if no such split could be found.
    issuccess || return false

    # At this point we just need to check that we `canview` the value.
    value = dict[VarName(vn, keylens)]

    return canview(child, value)
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
       DynamicPPL.infer_nested_eltype(Int[])
Int64

julia> # `NamedTuple`
       DynamicPPL.infer_nested_eltype(typeof((x = [1], )))
Int64

julia> # `AbstractDict`
       DynamicPPL.infer_nested_eltype(typeof(Dict(:x => [1, ])))
Int64

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

function infer_nested_eltype(::Type{T}) where {T<:Tuple}
    # Calling `infer_nested_eltype` on the result allows us to handle `Union`, etc.
    return infer_nested_eltype(promote_type(map(infer_nested_eltype, T.parameters)...))
end

# Handle `AbstractDict` differently since `eltype` results in a `Pair`.
infer_nested_eltype(::Type{<:AbstractDict{<:Any,ET}}) where {ET} = infer_nested_eltype(ET)

# No need + causes issues for some AD backends, e.g. Zygote.
ChainRulesCore.@non_differentiable infer_nested_eltype(x)
