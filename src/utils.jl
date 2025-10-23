# singleton for indicating if no default arguments are present
struct NoDefault end
const NO_DEFAULT = NoDefault()

# A short-hand for a type commonly used in type signatures for VarInfo methods.
VarNameTuple = NTuple{N,VarName} where {N}

# TODO(mhauru) This is currently used in the transformation functions of NoDist,
# ReshapeTransform, and UnwrapSingletonTransform, and in VarInfo. We should also use it in
# SimpleVarInfo and maybe other places.
"""
The type for all log probability variables.

This is Float64 on 64-bit systems and Float32 on 32-bit systems.
"""
const LogProbType = float(Real)

"""
    @addlogprob!(ex)

Add a term to the log joint.

If `ex` evaluates to a `NamedTuple` with keys `:loglikelihood` and/or `:logprior`, the
values are added to the log likelihood and log prior respectively.

If `ex` evaluates to a number it is added to the log likelihood.

# Examples

```jldoctest; setup = :(using Distributions)
julia> mylogjoint(x, μ) = (; loglikelihood=loglikelihood(Normal(μ, 1), x), logprior=1.0);

julia> @model function demo(x)
           μ ~ Normal()
           @addlogprob! mylogjoint(x, μ)
       end;

julia> x = [1.3, -2.1];

julia> loglikelihood(demo(x), (μ=0.2,)) ≈ mylogjoint(x, 0.2).loglikelihood
true

julia> logprior(demo(x), (μ=0.2,)) ≈ logpdf(Normal(), 0.2) + mylogjoint(x, 0.2).logprior
true
```

and to [reject samples](https://github.com/TuringLang/Turing.jl/issues/1328):

```jldoctest; setup = :(using Distributions, LinearAlgebra)
julia> @model function demo(x)
           m ~ MvNormal(zero(x), I)
           if dot(m, x) < 0
               @addlogprob! (; loglikelihood=-Inf)
               # Exit the model evaluation early
               return
           end
           x ~ MvNormal(m, I)
           return
       end;

julia> logjoint(demo([-2.1]), (m=[0.2],)) == -Inf
true
```
"""
macro addlogprob!(ex)
    return quote
        val = $(esc(ex))
        vi = $(esc(:(__varinfo__)))
        if val isa Number
            if hasacc(vi, Val(:LogLikelihood))
                $(esc(:(__varinfo__))) = accloglikelihood!!($(esc(:(__varinfo__))), val)
            end
        elseif val isa NamedTuple
            $(esc(:(__varinfo__))) = acclogp!!(
                $(esc(:(__varinfo__))), val; ignore_missing_accumulator=true
            )
        else
            error("logp must be a Number or a NamedTuple.")
        end
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

function Bijectors.with_logabsdet_jacobian(f::UnwrapSingletonTransform, x)
    return f(x), zero(LogProbType)
end

function Bijectors.with_logabsdet_jacobian(
    inv_f::Bijectors.Inverse{<:UnwrapSingletonTransform}, x
)
    f = inv_f.orig
    result = reshape([x], f.input_size)
    return result, zero(LogProbType)
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

function Bijectors.with_logabsdet_jacobian(f::ReshapeTransform, x)
    return f(x), zero(LogProbType)
end

function Bijectors.with_logabsdet_jacobian(inv_f::Bijectors.Inverse{<:ReshapeTransform}, x)
    return inv_f(x), zero(LogProbType)
end

struct ToChol <: Bijectors.Bijector
    uplo::Char
end

function Bijectors.with_logabsdet_jacobian(f::ToChol, x)
    return Cholesky(Matrix(x), f.uplo, 0), zero(LogProbType)
end

function Bijectors.with_logabsdet_jacobian(::Bijectors.Inverse{<:ToChol}, y::Cholesky)
    return y.UL, zero(LogProbType)
end

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
from_vec_transform_for_size(::Tuple{<:Any}) = identity

"""
    from_vec_transform(dist::Distribution)

Return the transformation from the vector representation of a realization from
distribution `dist` to the original representation compatible with `dist`.
"""
from_vec_transform(dist::Distribution) = from_vec_transform_for_size(size(dist))
from_vec_transform(::UnivariateDistribution) = UnwrapSingletonTransform()
from_vec_transform(dist::LKJCholesky) = ToChol(dist.uplo) ∘ ReshapeTransform(size(dist))

struct ProductNamedTupleUnvecTransform{names,T<:NamedTuple{names}}
    dists::T
    # The `i`-th input range corresponds to the segment of the input vector
    # that belongs to the `i`-th distribution.
    input_ranges::Vector{UnitRange}
    function ProductNamedTupleUnvecTransform(
        d::Distributions.ProductNamedTupleDistribution{names}
    ) where {names}
        offset = 1
        input_ranges = UnitRange[]
        for name in names
            this_dist = d.dists[name]
            this_name_size = _input_length(from_vec_transform(this_dist))
            push!(input_ranges, offset:(offset + this_name_size - 1))
            offset += this_name_size
        end
        return new{names,typeof(d.dists)}(d.dists, input_ranges)
    end
end

@generated function (trf::ProductNamedTupleUnvecTransform{names})(
    x::AbstractVector
) where {names}
    expr = Expr(:tuple)
    for (i, name) in enumerate(names)
        push!(
            expr.args,
            :($name = from_vec_transform(trf.dists.$name)(x[trf.input_ranges[$i]])),
        )
    end
    return expr
end

function from_vec_transform(dist::Distributions.ProductNamedTupleDistribution)
    return ProductNamedTupleUnvecTransform(dist)
end
function Bijectors.with_logabsdet_jacobian(f::ProductNamedTupleUnvecTransform, x)
    return f(x), zero(LogProbType)
end

# This function returns the length of the vector that the function from_vec_transform
# expects. This helps us determine which segment of a concatenated vector belongs to which
# variable.
_input_length(from_vec_trfm::UnwrapSingletonTransform) = 1
_input_length(from_vec_trfm::ReshapeTransform) = prod(from_vec_trfm.output_size)
function _input_length(trfm::ProductNamedTupleUnvecTransform)
    return sum(_input_length ∘ from_vec_transform, values(trfm.dists))
end
function _input_length(
    c::ComposedFunction{<:DynamicPPL.ToChol,<:DynamicPPL.ReshapeTransform}
)
    return _input_length(c.inner)
end

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
function from_linked_vec_transform(dist::Distributions.ProductNamedTupleDistribution)
    return invlink_transform(dist)
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
tovec(t::Tuple) = mapreduce(tovec, vcat, t)
tovec(nt::NamedTuple) = mapreduce(tovec, vcat, values(nt))
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
    return p === nothing ? VarName{getsym(vn)}(identity) : VarName{getsym(vn)}(p)
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
        o == getoptic(vn_parent)
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
    float_type_with_fallback(T::DataType)

Return `T` if it is a Real; otherwise return `float(Real)`.
"""
float_type_with_fallback(::Type) = float(Real)
float_type_with_fallback(::Type{Union{}}) = float(Real)
float_type_with_fallback(::Type{T}) where {T<:Real} = T

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

# Convert (x=1,) to Dict(@varname(x) => 1)
function to_varname_dict(nt::NamedTuple)
    return Dict{VarName,Any}(VarName{k}() => v for (k, v) in pairs(nt))
end
to_varname_dict(d::AbstractDict) = d
# Version of `merge` used by `conditioned` and `fixed` to handle
# the scenario where we might try to merge a dict with an empty
# tuple.
# TODO: Maybe replace the default of returning `NamedTuple` with `nothing`?
_merge(left::NamedTuple, right::NamedTuple) = merge(left, right)
_merge(left::AbstractDict, right::AbstractDict) = merge(left, right)
_merge(left::AbstractDict, ::NamedTuple{()}) = left
_merge(left::AbstractDict, right::NamedTuple) = merge(left, to_varname_dict(right))
_merge(::NamedTuple{()}, right::AbstractDict) = right
_merge(left::NamedTuple, right::AbstractDict) = merge(to_varname_dict(left), right)

"""
    unique_syms(vns::T) where {T<:NTuple{N,VarName}}

Return the unique symbols of the variables in `vns`.

Note that `unique_syms` is only defined for `Tuple`s of `VarName`s and, unlike
`Base.unique`, returns a `Tuple`. The point of `unique_syms` is that it supports constant
propagating the result, which is possible only when the argument and the return value are
`Tuple`s.
"""
@generated function unique_syms(::T) where {T<:VarNameTuple}
    retval = Expr(:tuple)
    syms = [first(vn.parameters) for vn in T.parameters]
    for sym in unique(syms)
        push!(retval.args, QuoteNode(sym))
    end
    return retval
end

"""
    group_varnames_by_symbol(vns::NTuple{N,VarName}) where {N}

Return a `NamedTuple` of the variables in `vns` grouped by symbol.

Note that `group_varnames_by_symbol` only accepts a `Tuple` of `VarName`s. This allows it to
be type stable.

Example:
```julia
julia> vns_tuple = (@varname(x), @varname(y[1]), @varname(x.a), @varname(z[15]), @varname(y[2]))
(x, y[1], x.a, z[15], y[2])

julia> vns_nt = (; x=[@varname(x), @varname(x.a)], y=[@varname(y[1]), @varname(y[2])], z=[@varname(z[15])])
(x = VarName{:x}[x, x.a], y = VarName{:y, IndexLens{Tuple{Int64}}}[y[1], y[2]], z = VarName{:z, IndexLens{Tuple{Int64}}}[z[15]])

julia> group_varnames_by_symbol(vns_tuple) == vns_nt
```
"""
function group_varnames_by_symbol(vns::VarNameTuple)
    syms = unique_syms(vns)
    elements = map(collect, tuple((filter(vn -> getsym(vn) == s, vns) for s in syms)...))
    return NamedTuple{syms}(elements)
end

"""
    basetypeof(x)

Return `typeof(x)` stripped of its type parameters.
"""
basetypeof(x::T) where {T} = Base.typename(T).wrapper
