# subset is defined here to avoid circular dependencies between files. Methods for it are
# defined in other files.
function subset end

"""
The type for all log probability variables.

This is Float64 on 64-bit systems and Float32 on 32-bit systems.
"""
const LogProbType = Ref(float(Real))

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
    (x, zero(LogProbType[]))
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
