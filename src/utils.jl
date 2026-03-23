# subset is defined here to avoid circular dependencies between files. Methods for it are
# defined in other files.
function subset end

using Preferences: @load_preference, @set_preferences!

"""
    DynamicPPL.NoLogProb <: Real

Singleton type that represents the absence of a log probability value. This is used as the
default type parameter for `LogProbAccumulator` when no log probability value is needed, to
avoid defining a concrete type such as `Float64` that would cause unwanted type promotion
when accumulating log probabilities of other types (e.g., `Float32`).

Adding anything to `NoLogProb()` returns the other thing. In other words, `NoLogProb` is a
true additive identity which additionally preserves types.
"""
struct NoLogProb <: Real end
Base.zero(::Type{NoLogProb}) = NoLogProb()
Base.convert(::Type{T}, ::NoLogProb) where {T<:Number} = zero(T)
Base.promote_rule(::Type{NoLogProb}, ::Type{T}) where {T<:Number} = T
Base.iszero(::NoLogProb) = true
Base.hash(::NoLogProb, h::UInt) = hash(0.0, h)
Base.:(+)(::NoLogProb, ::NoLogProb) = NoLogProb()
(::Type{T})(::NoLogProb) where {T<:Real} = zero(T)

const FLOAT_TYPE_PREF_KEY = "floattype"

"""
    DynamicPPL.LogProbType

The default type used for log-probabilities in DynamicPPL.jl. This is a compile-time constant
that can be set via [`set_logprob_type!`](@ref), which under the hood uses Preferences.jl.

Note that this does not prevent computations within the model from promoting the
log-probability to a different type. In essence, `LogProbType` specifies the *lowest*
possible type that log-probabilities can be, and DynamicPPL promises to not insert any extra
operations that would cause this to be promoted to a higher type. However, DynamicPPL cannot
guard against user code inside models.

For example, in:

```julia
@model f() = x ~ Normal(0.0, 1.0)
```

the log-probability of the model will always be promoted to `Float64`, regardless of the
value of `LogProbType`, because the logpdf of `Normal(0.0, 1.0)` is a `Float64`. On the
other hand, in:

```julia
@model f() = x ~ Normal(0.0f0, 1.0f0)
```

the log-probability of the model will be `Float32` if `LogProbType` is `Float32` or lower.
"""
const LogProbType = let
    logp_pref = @load_preference(FLOAT_TYPE_PREF_KEY, "f64")
    if logp_pref == "f64"
        Float64
    elseif logp_pref == "f32"
        Float32
    elseif logp_pref == "f16"
        Float16
    elseif logp_pref == "min"
        NoLogProb
    else
        error("Unsupported log probability preference: $logp_pref")
    end
end

"""
    set_logprob_type!(::Type{T}) where {T}

Set the log probability type for DynamicPPL.jl, [`DynamicPPL.LogProbType`](@ref), to `T`.
Permitted values are `Float64`, `Float32`, `Float16`, and `NoLogProb`. The default in
DynamicPPL is `Float64`.

`NoLogProb` is a special type that is the "lowest" possible float type. This means that the
log probability will be promoted to whatever type the model dictates. This is a totally
unintrusive option, which can be useful if you do not know in advance what log probability
type you are targeting, or want to troubleshoot a model to see what type the log probability
is being promoted to. However, this can also cause type stability issues and performance
degradations, so we generally recommend setting a specific log probability type if you know
what type you want to target.

This function uses Preferences.jl to set a compile-time constant, so you will need to
restart your Julia session for the change to take effect.
"""
function set_logprob_type!(::Type{T}) where {T}
    new_pref = if T == Float64
        "f64"
    elseif T == Float32
        "f32"
    elseif T == Float16
        "f16"
    elseif T == NoLogProb
        "min"
    else
        throw(ArgumentError("Unsupported log probability type: $T"))
    end
    @set_preferences!(FLOAT_TYPE_PREF_KEY => new_pref)
    @info "DynamicPPL's log probability type has been set to $T.\nPlease note you will need to restart your Julia session for this change to take effect."
end

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
