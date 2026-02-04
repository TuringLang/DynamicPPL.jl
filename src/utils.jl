# subset is defined here to avoid circular dependencies between files. Methods for it are
# defined in other files.
function subset end

"""
The type for all log probability variables.

This is Float64 on 64-bit systems and Float32 on 32-bit systems.
"""
const LogProbType = float(Real)

"""
    typed_identity(x)

Identity function, but with an overload for `with_logabsdet_jacobian` to ensure
that it returns a sensible zero logjac.

The problem with plain old `identity` is that the default definition of
`with_logabsdet_jacobian` for `identity` returns `zero(eltype(x))`:
https://github.com/JuliaMath/ChangesOfVariables.jl/blob/d6a8115fc9b9419decbdb48e2c56ec9675b4c6a4/src/with_ladj.jl#L154

This is fine for most samples `x`, but if `eltype(x)` doesn't return a sensible type (e.g.
if it's `Any`), then using `identity` will error with `zero(Any)`. This can happen with,
for example, `ProductNamedTupleDistribution`:

```julia
julia> using Distributions; d = product_distribution((a = Normal(), b = LKJCholesky(3, 0.5)));

julia> eltype(rand(d))
Any
```

The same problem precludes us from eventually broadening the scope of DynamicPPL.jl to
support distributions with non-numeric samples.

Furthermore, in principle, the type of the log-probability should be separate from the type
of the sample. Thus, instead of using `zero(LogProbType)`, we should use the eltype of the
LogJacobianAccumulator. There's no easy way to thread that through here, but if a way to do
this is discovered, then `typed_identity` is what will allow us to obtain that custom
behaviour.
"""
function typed_identity end
@inline typed_identity(x) = x
@inline Bijectors.with_logabsdet_jacobian(::typeof(typed_identity), x) =
    (x, zero(LogProbType))
@inline Bijectors.inverse(::typeof(typed_identity)) = typed_identity

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

struct ProductNamedTupleUnvecTransform{names,T<:NamedTuple{names}} <: Bijectors.Bijector
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

@generated function (inv_trf::Bijectors.Inverse{<:ProductNamedTupleUnvecTransform{names}})(
    x::NamedTuple{names}
) where {names}
    exprs = Expr[]
    for name in names
        push!(exprs, :(to_vec_transform(inv_trf.orig.dists.$name)(x.$name)))
    end
    return :(vcat($(exprs...)))
end

function from_vec_transform(dist::Distributions.ProductNamedTupleDistribution)
    return ProductNamedTupleUnvecTransform(dist)
end

function Bijectors.with_logabsdet_jacobian(f::ProductNamedTupleUnvecTransform, x)
    return f(x), zero(LogProbType)
end

function Bijectors.with_logabsdet_jacobian(
    inv_f::Bijectors.Inverse{<:ProductNamedTupleUnvecTransform}, x
)
    return inv_f(x), zero(LogProbType)
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
    update_values!!(vi::AbstractVarInfo, vals::NamedTuple, vns)

Return instance similar to `vi` but with `vns` set to values from `vals`.
"""
function update_values!!(vi::AbstractVarInfo, vals::NamedTuple, vns)
    for vn in vns
        vi = DynamicPPL.setindex!!(vi, AbstractPPL.getvalue(vals, vn), vn)
    end
    return vi
end

"""
    float_type_with_fallback(T::DataType)

Return `float(T)` if possible; otherwise return `float(Real)`.
"""
float_type_with_fallback(::Type) = float(Real)
float_type_with_fallback(::Type{Union{}}) = float(Real)
float_type_with_fallback(::Type{T}) where {T<:Real} = float(T)

"""
    basetypeof(x)

Return `typeof(x)` stripped of its type parameters.
"""
basetypeof(::T) where {T} = Base.typename(T).wrapper

const MaybeTypedIdentity = Union{typeof(typed_identity),typeof(identity)}

# TODO(mhauru) Might add another specialisation to _compose_no_identity, where if
# ReshapeTransforms are composed with each other or with a an UnwrapSingeltonTransform, only
# the latter one would be kept.
"""
    _compose_no_identity(f, g)

Like `f ∘ g`, but if `f` or `g` is `identity` it is omitted.

This helps avoid trivial cases of `ComposedFunction` that would cause unnecessary type
conflicts.
"""
_compose_no_identity(f, g) = f ∘ g
_compose_no_identity(::MaybeTypedIdentity, g) = g
_compose_no_identity(f, ::MaybeTypedIdentity) = f
_compose_no_identity(::MaybeTypedIdentity, ::MaybeTypedIdentity) = typed_identity
