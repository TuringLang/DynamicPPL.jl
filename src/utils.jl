# singleton for indicating if no default arguments are present
struct NoDefault end
const NO_DEFAULT = NoDefault()

# A short-hand for a type commonly used in type signatures for VarInfo methods.
VarNameCollection = Union{NTuple{N,VarName} where N,AbstractVector{<:VarName}}

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

"""
    UnwrapSingletonTransform(input_size::InSize)

A transformation that unwraps a singleton array, returning a scalar.

The `input_size` field is the expected size of the input. In practice this only determines
the number of indices, since all dimensions must be 1 for a singleton. `input_size` is used
to check the validity of the input, but also to determine the correct inverse operation.

By default `input_size` is `(1,)`, in which case `tovec` is the inverse.
"""
struct UnwrapSingletonTransform{InSize} <: Bijectors.Bijector
    input_size::InSize
end

UnwrapSingletonTransform() = UnwrapSingletonTransform((1,))

function (f::UnwrapSingletonTransform)(x)
    if size(x) != f.input_size
        throw(DimensionMismatch("Expected input of size $(f.input_size), got $(size(x))"))
    end
    return only(x)
end

Bijectors.with_logabsdet_jacobian(f::UnwrapSingletonTransform, x) = (f(x), 0)
function Bijectors.with_logabsdet_jacobian(
    inv_f::Bijectors.Inverse{<:UnwrapSingletonTransform}, x
)
    f = inv_f.orig
    return (reshape([x], f.input_size), 0)
end

"""
    ReshapeTransform(input_size::InSize, output_size::OutSize)

A `Bijector` that transforms arrays of size `input_size` to arrays of size `output_size`.

`input_size` is not needed for the implementation of the transformation. It is only used to
check that the input is of the expected size, and to determine the correct inverse
operation.

By default `input_size` is the vectorized version of `output_size`. In this case this
transformation is the inverse of `tovec` called on an array.
"""
struct ReshapeTransform{InSize,OutSize} <: Bijectors.Bijector
    input_size::InSize
    output_size::OutSize
end

function ReshapeTransform(output_size::Tuple)
    input_size = (prod(output_size),)
    return ReshapeTransform(input_size, output_size)
end

ReshapeTransform(x::AbstractArray) = ReshapeTransform(size(x))

# TODO: Should we materialize the `reshape`?
function (f::ReshapeTransform)(x)
    if size(x) != f.input_size
        throw(DimensionMismatch("Expected input of size $(f.input_size), got $(size(x))"))
    end
    if f.output_size == ()
        # Specially handle the case where x is a singleton array, see
        # https://github.com/JuliaDiff/ReverseDiff.jl/issues/265 and
        # https://github.com/TuringLang/DynamicPPL.jl/issues/698
        return fill(x[], ())
    else
        # The call to `tovec` is only needed in case `x` is a scalar.
        return reshape(tovec(x), f.output_size)
    end
end

function (inv_f::Bijectors.Inverse{<:ReshapeTransform})(x)
    f = inv_f.orig
    inverse = ReshapeTransform(f.output_size, f.input_size)
    return inverse(x)
end

Bijectors.with_logabsdet_jacobian(f::ReshapeTransform, x) = (f(x), 0)

function Bijectors.with_logabsdet_jacobian(inv_f::Bijectors.Inverse{<:ReshapeTransform}, x)
    return (inv_f(x), 0)
end

struct ToChol <: Bijectors.Bijector
    uplo::Char
end

Bijectors.with_logabsdet_jacobian(f::ToChol, x) = (Cholesky(Matrix(x), f.uplo, 0), 0)
Bijectors.with_logabsdet_jacobian(::Bijectors.Inverse{<:ToChol}, y::Cholesky) = (y.UL, 0)
function Bijectors.with_logabsdet_jacobian(::Bijectors.Inverse{<:ToChol}, y)
    return error(
        "Inverse{ToChol} is only defined for Cholesky factorizations. " *
        "Got a $(typeof(y)) instead.",
    )
end

"""
    from_vec_transform(x)

Return the transformation from the vector representation of `x` to original representation.
"""
from_vec_transform(x::AbstractArray) = from_vec_transform_for_size(size(x))
from_vec_transform(C::Cholesky) = ToChol(C.uplo) ∘ ReshapeTransform(size(C.UL))
from_vec_transform(::Real) = UnwrapSingletonTransform()

"""
    from_vec_transform_for_size(sz::Tuple)

Return the transformation from the vector representation of a realization of size `sz` to
original representation.
"""
from_vec_transform_for_size(sz::Tuple) = ReshapeTransform(sz)
# TODO(mhauru) Is the below used? If not, this function can be removed.
from_vec_transform_for_size(::Tuple{<:Any}) = identity

"""
    from_vec_transform(dist::Distribution)

Return the transformation from the vector representation of a realization from
distribution `dist` to the original representation compatible with `dist`.
"""
from_vec_transform(dist::Distribution) = from_vec_transform_for_size(size(dist))
from_vec_transform(::UnivariateDistribution) = UnwrapSingletonTransform()
from_vec_transform(dist::LKJCholesky) = ToChol(dist.uplo) ∘ ReshapeTransform(size(dist))

"""
    from_vec_transform(f, size::Tuple)

Return the transformation from the vector representation of a realization of size `size` to original representation.

This is useful when the transformation alters the size of the realization, in which case we need to account for the
size of the realization after pushed through the transformation.
"""
from_vec_transform(f, sz) = from_vec_transform_for_size(Bijectors.output_size(f, sz))

"""
    from_linked_vec_transform(dist::Distribution)

Return the transformation from the unconstrained vector to the constrained
realization of distribution `dist`.

By default, this is just `invlink_transform(dist) ∘ from_vec_transform(dist)`.

See also: [`DynamicPPL.invlink_transform`](@ref), [`DynamicPPL.from_vec_transform`](@ref).
"""
function from_linked_vec_transform(dist::Distribution)
    f_invlink = invlink_transform(dist)
    f_vec = from_vec_transform(inverse(f_invlink), size(dist))
    return f_invlink ∘ f_vec
end

# UnivariateDistributions need to be handled as a special case, because size(dist) is (),
# which makes the usual machinery think we are dealing with a 0-dim array, whereas in
# actuality we are dealing with a scalar.
# TODO(mhauru) Hopefully all this can go once the old Gibbs sampler is removed and
# VarNamedVector takes over from Metadata.
function from_linked_vec_transform(dist::UnivariateDistribution)
    f_invlink = invlink_transform(dist)
    f_vec = from_vec_transform(inverse(f_invlink), size(dist))
    f_combined = f_invlink ∘ f_vec
    sz = Bijectors.output_size(f_combined, size(dist))
    return UnwrapSingletonTransform(sz) ∘ f_combined
end

# Specializations that circumvent the `from_vec_transform` machinery.
function from_linked_vec_transform(dist::LKJCholesky)
    return inverse(Bijectors.VecCholeskyBijector(dist.uplo))
end
from_linked_vec_transform(::LKJ) = inverse(Bijectors.VecCorrBijector())

"""
    to_vec_transform(x)

Return the transformation from the original representation of `x` to the vector
representation.
"""
to_vec_transform(x) = inverse(from_vec_transform(x))

"""
    to_linked_vec_transform(dist)

Return the transformation from the constrained realization of distribution `dist`
to the unconstrained vector.
"""
to_linked_vec_transform(x) = inverse(from_linked_vec_transform(x))

# FIXME: When given a `LowerTriangular`, `VarInfo` still stores the full matrix
# flattened, while using `tovec` below flattenes only the necessary entries.
# => Need to either fix how `VarInfo` does things, i.e. use `tovec` everywhere,
# or fix `tovec` to flatten the full matrix instead of using `Bijectors.triu_to_vec`.
tovec(x::Real) = [x]
tovec(x::AbstractArray) = vec(x)
tovec(C::Cholesky) = tovec(Matrix(C.UL))

"""
    recombine(dist::Union{UnivariateDistribution,MultivariateDistribution}, vals::AbstractVector, n::Int)

Recombine `vals`, representing a batch of samples from `dist`, so that it's a compatible with `dist`.

!!! warning
    This only supports `UnivariateDistribution` and `MultivariateDistribution`, which are the only two
    distribution types which are allowed on the right-hand side of a `.~` statement in a model.
"""
function recombine(::UnivariateDistribution, val::AbstractVector, ::Int)
    # This is just a no-op, since we're trying to convert a vector into a vector.
    return copy(val)
end
function recombine(d::MultivariateDistribution, val::AbstractVector, n::Int)
    # Here `val` is of the length `length(d) * n` and so we need to reshape it.
    return copy(reshape(val, length(d), n))
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
"""
    collect_maybe(x)

Return `x` if `x` is an array, otherwise return `collect(x)`.
"""
collect_maybe(x) = collect(x)
collect_maybe(x::AbstractArray) = x

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

Return type corresponding to `float(typeof(x))` if possible; otherwise return `float(Real)`.
"""
float_type_with_fallback(::Type) = float(Real)
float_type_with_fallback(::Type{Union{}}) = float(Real)
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
    varname_and_value_leaves(container)

Return an iterator over all varname-value pairs that are represented by `container`.

This is the same as [`varname_and_value_leaves(vn::VarName, x)`](@ref) but over a container
containing multiple varnames.

See also: [`varname_and_value_leaves(vn::VarName, x)`](@ref).

# Examples
```jldoctest varname-and-value-leaves-container
julia> using DynamicPPL: varname_and_value_leaves

julia> # With an `OrderedDict`
       dict = OrderedDict(@varname(y) => 1, @varname(z) => [[2.0], [3.0]]);

julia> foreach(println, varname_and_value_leaves(dict))
(y, 1)
(z[1][1], 2.0)
(z[2][1], 3.0)

julia> # With a `NamedTuple`
       nt = (y = 1, z = [[2.0], [3.0]]);

julia> foreach(println, varname_and_value_leaves(nt))
(y, 1)
(z[1][1], 2.0)
(z[2][1], 3.0)
```
"""
function varname_and_value_leaves(container::OrderedDict)
    return Iterators.flatten(varname_and_value_leaves(k, v) for (k, v) in container)
end
function varname_and_value_leaves(container::NamedTuple)
    return Iterators.flatten(
        varname_and_value_leaves(VarName{k}(), v) for (k, v) in pairs(container)
    )
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

"""
    unique_syms(vns::T) where {T<:NTuple{N,VarName}}

Return the unique symbols of the variables in `vns`.

Note that `unique_syms` is only defined for `Tuple`s of `VarName`s and, unlike
`Base.unique`, returns a `Tuple`. For an `AbstractVector{<:VarName}` you can use
`Base.unique`. The point of `unique_syms` is that it supports constant propagating
the result, which is possible only when the argument and the return value are `Tuple`s.
"""
@generated function unique_syms(::T) where {T<:NTuple{N,VarName}} where {N}
    retval = Expr(:tuple)
    syms = [first(vn.parameters) for vn in T.parameters]
    for sym in unique(syms)
        push!(retval.args, QuoteNode(sym))
    end
    return retval
end

"""
    varname_namedtuple(vns::NTuple{N,VarName}) where {N}
    varname_namedtuple(vns::AbstractVector{<:VarName})
    varname_namedtuple(vns::NamedTuple)

Return a `NamedTuple` of the variables in `vns` grouped by symbol.

`varname_namedtuple` is type stable for inputs that are `Tuple`s, and for vectors when all
`VarName`s in the vector have the same symbol. For a `NamedTuple` it's a no-op.

Example:
```julia
julia> vns_tuple = (@varname(x), @varname(y[1]), @varname(x.a), @varname(z[15]), @varname(y[2]))
(x, y[1], x.a, z[15], y[2])

julia> vns_nt = (; x=[@varname(x), @varname(x.a)], y=[@varname(y[1]), @varname(y[2])], z=[@varname(z[15])])
(x = VarName{:x}[x, x.a], y = VarName{:y, IndexLens{Tuple{Int64}}}[y[1], y[2]], z = VarName{:z, IndexLens{Tuple{Int64}}}[z[15]])

julia> varname_namedtuple(vns_tuple) == vns_nt
```
"""
function varname_namedtuple(vns::NTuple{N,VarName} where {N})
    syms = unique_syms(vns)
    elements = map(collect, tuple((filter(vn -> getsym(vn) == s, vns) for s in syms)...))
    return NamedTuple{syms}(elements)
end

# This method is type unstable, but that can't be helped: The problem is inherently type
# unstable if there are VarNames with multiple symbols in a Vector.
function varname_namedtuple(vns::AbstractVector{<:VarName})
    syms = tuple(unique(map(getsym, vns))...)
    elements = tuple((filter(vn -> getsym(vn) == s, vns) for s in syms)...)
    return NamedTuple{syms}(elements)
end

# A simpler, type stable implementation when all the VarNames in a Vector have the same
# symbol.
function varname_namedtuple(vns::AbstractVector{<:VarName{T}}) where {T}
    return NamedTuple{(T,)}((vns,))
end

varname_namedtuple(vns::NamedTuple) = vns
