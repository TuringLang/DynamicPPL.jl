"""
    SimpleVarInfo{NT,T} <: AbstractVarInfo

A simple wrapper of the parameters with a `logp` field for
accumulation of the logdensity.

Currently only implemented for `NT<:NamedTuple` and `NT<:Dict`.

# Notes
The major differences between this and `TypedVarInfo` are:
1. `SimpleVarInfo` does not require linearization.
2. `SimpleVarInfo` can use more efficient bijectors.
3. `SimpleVarInfo` is only type-stable if `NT<:NamedTuple` and either
   a) no indexing is used in tilde-statements, or
   b) the values have been specified with the corret shapes.

# Examples
```jldoctest; setup=:(using Distributions)
julia> using StableRNGs

julia> @model function demo()
           m ~ Normal()
           x = Vector{Float64}(undef, 2)
           for i in eachindex(x)
               x[i] ~ Normal()
           end
           return x
       end
demo (generic function with 1 method)

julia> m = demo();

julia> rng = StableRNG(42);

julia> ### Sampling ###
       ctx = SamplingContext(Random.GLOBAL_RNG, SampleFromPrior(), DefaultContext());

julia> # In the `NamedTuple` version we need to provide the place-holder values for
       # the variablse which are using "containers", e.g. `Array`.
       # In this case, this means that we need to specify `x` but not `m`.
       _, vi = DynamicPPL.evaluate(m, SimpleVarInfo((x = ones(2), )), ctx); vi
SimpleVarInfo{NamedTuple{(:x, :m), Tuple{Vector{Float64}, Float64}}, Float64}((x = [1.6642061055583879, 1.796319600944139], m = -0.16796295277202952), -5.769094411622931)

julia> # (✓) Vroom, vroom! FAST!!!
       DynamicPPL.getval(vi, @varname(x[1]))
1.6642061055583879

julia> # We can also access arbitrary varnames pointing to `x`, e.g.
       DynamicPPL.getval(vi, @varname(x))
2-element Vector{Float64}:
 1.6642061055583879
 1.796319600944139

julia> DynamicPPL.getval(vi, @varname(x[1:2]))
2-element view(::Vector{Float64}, 1:2) with eltype Float64:
 1.6642061055583879
 1.796319600944139

julia> # (×) If we don't provide the container...
       _, vi = DynamicPPL.evaluate(m, SimpleVarInfo(), ctx); vi
ERROR: type NamedTuple has no field x
[...]

julia> # If one does not know the varnames, we can use a `Dict` instead.
       _, vi = DynamicPPL.evaluate(m, SimpleVarInfo{Float64}(Dict()), ctx); vi
SimpleVarInfo{Dict{Any, Any}, Float64}(Dict{Any, Any}(x[1] => 1.192696983568277, x[2] => 0.4914514300738121, m => 0.25572200616753643), -3.6215377732004237)

julia> # (✓) Sort of fast, but only possible at runtime.
       DynamicPPL.getval(vi, @varname(x[1]))
1.192696983568277

julia> # In addtion, we can only access varnames as they appear in the model!
       DynamicPPL.getval(vi, @varname(x))
ERROR: KeyError: key x not found
[...]

julia> julia> DynamicPPL.getval(vi, @varname(x[1:2]))
ERROR: KeyError: key x[1:2] not found
[...]
```
"""
struct SimpleVarInfo{NT,T} <: AbstractVarInfo
    θ::NT
    logp::T
end

SimpleVarInfo{T}(θ) where {T<:Real} = SimpleVarInfo{typeof(θ),T}(θ, zero(T))
SimpleVarInfo(θ) = SimpleVarInfo{eltype(first(θ))}(θ)
SimpleVarInfo{T}() where {T<:Real} = SimpleVarInfo{T}(NamedTuple())
SimpleVarInfo() = SimpleVarInfo{Float64}()

# Constructor from `Model`.
SimpleVarInfo(model::Model, args...) = SimpleVarInfo{Float64}(model, args...)
function SimpleVarInfo{T}(model::Model, args...) where {T<:Real}
    _, svi = DynamicPPL.evaluate(model, SimpleVarInfo{T}(), args...)
    return svi
end

# Constructor from `VarInfo`.
function SimpleVarInfo(vi::TypedVarInfo, ::Type{D}=NamedTuple; kwargs...) where {D}
    return SimpleVarInfo{eltype(getlogp(vi))}(vi, D; kwargs...)
end
function SimpleVarInfo{T}(
    vi::VarInfo{<:NamedTuple{names}}, ::Type{D}
) where {T<:Real,names,D}
    values = values_as(vi, D)
    return SimpleVarInfo{T}(values)
end

getlogp(vi::SimpleVarInfo) = vi.logp
setlogp!!(vi::SimpleVarInfo, logp) = SimpleVarInfo(vi.θ, logp)
acclogp!!(vi::SimpleVarInfo, logp) = SimpleVarInfo(vi.θ, getlogp(vi) + logp)

function setlogp!!(vi::SimpleVarInfo{<:Any,<:Ref}, logp)
    vi.logp[] = logp
    return vi
end

function acclogp!!(vi::SimpleVarInfo{<:Any,<:Ref}, logp)
    vi.logp[] += logp
    return vi
end

# TODO: Get rid of this once we have lenses.
_getindex_view(x, inds::Tuple) = _getindex(view(x, first(inds)...), Base.tail(inds))
_getindex_view(x, inds::Tuple{}) = x

# TODO: Get rid of this once we have lenses.
function _setvalue!!(nt::NamedTuple, val, vn::VarName{sym,Tuple{}}) where {sym}
    return merge(nt, NamedTuple{(sym, )}((val, )))
end
function _setvalue!!(nt::NamedTuple, val, vn::VarName{sym}) where {sym}
    # Use `getproperty` instead of `getfield`
    value = getproperty(nt, sym)
    # Note that this will return a `view`, even if the resulting value is 0-dim.
    # This makes it possible to call `setindex!` on the result later to update
    # in place even in the case where are retrieving a single element, e.g. `x[1]`.
    dest_view = _getindex_view(value, vn.indexing)
    dest_view .= val

    return nt
end

# `NamedTuple`
function getval(vi::SimpleVarInfo{<:NamedTuple}, vn::VarName{sym}) where {sym}
    return _getvalue(vi.θ, Val{sym}(), vn.indexing)
end

# `Dict`
function getval(vi::SimpleVarInfo{<:Dict}, vn::VarName)
    return vi.θ[vn]
end

# `SimpleVarInfo` doesn't necessarily vectorize, so we can have arrays other than
# just `Vector`.
getval(vi::SimpleVarInfo, vns::AbstractArray{<:VarName}) = map(vn -> getval(vi, vn), vns)
# To disambiguiate.
getval(vi::SimpleVarInfo, vns::Vector{<:VarName}) = map(vn -> getval(vi, vn), vns)

haskey(vi::SimpleVarInfo, vn) = haskey(vi.θ, getsym(vn))

istrans(::SimpleVarInfo, vn::VarName) = false

getindex(vi::SimpleVarInfo, spl::SampleFromPrior) = vi.θ
getindex(vi::SimpleVarInfo, spl::SampleFromUniform) = vi.θ
# TODO: Should we do better?
getindex(vi::SimpleVarInfo, spl::Sampler) = vi.θ
getindex(vi::SimpleVarInfo, vn::VarName) = getval(vi, vn)
getindex(vi::SimpleVarInfo, vns::AbstractArray{<:VarName}) = getval(vi, vns)
# HACK: Need to disambiguiate.
getindex(vi::SimpleVarInfo, vns::Vector{<:VarName}) = getval(vi, vns)

# Necessary for `matchingvalue` to work properly.
function Base.eltype(
    vi::SimpleVarInfo{<:Any,T}, spl::Union{AbstractSampler,SampleFromPrior}
) where {T}
    return T
end

# `NamedTuple`
function push!!(
    vi::SimpleVarInfo{<:NamedTuple},
    vn::VarName{sym,Tuple{}},
    value,
    dist::Distribution,
    gidset::Set{Selector},
) where {sym}
    Setfield.@set vi.θ = merge(vi.θ, NamedTuple{(sym,)}((value,)))
end
function push!!(
    vi::SimpleVarInfo{<:NamedTuple},
    vn::VarName{sym},
    value,
    dist::Distribution,
    gidset::Set{Selector},
) where {sym}
    # We update in place.
    # We need a view into the array, hence we call `_getvalue` directly
    # rather than `getval`.
    _setvalue!!(vi.θ, value, vn)
    return vi
end

# `Dict`
function push!!(
    vi::SimpleVarInfo{<:Dict}, vn::VarName, r, dist::Distribution, gidset::Set{Selector}
)
    vi.θ[vn] = r
    return vi
end

# Context implementations
function tilde_assume!!(context, right, vn, inds, vi::SimpleVarInfo)
    value, logp, vi_new = tilde_assume(context, right, vn, inds, vi)
    return value, acclogp!!(vi_new, logp)
end

function assume(dist::Distribution, vn::VarName, vi::SimpleVarInfo)
    left = vi[vn]
    return left, Distributions.loglikelihood(dist, left), vi
end

function assume(
    rng::Random.AbstractRNG,
    sampler::SampleFromPrior,
    dist::Distribution,
    vn::VarName,
    vi::SimpleVarInfo,
)
    value = init(rng, dist, sampler)
    vi = push!!(vi, vn, value, dist, sampler)
    vi = settrans!!(vi, false, vn)
    return value, Distributions.loglikelihood(dist, value), vi
end

# function dot_tilde_assume!!(context, right, left, vn, inds, vi::SimpleVarInfo)
#     throw(MethodError(dot_tilde_assume!!, (context, right, left, vn, inds, vi)))
# end

function dot_tilde_assume!!(context, right, left, vn, inds, vi::SimpleVarInfo)
    value, logp, vi_new = dot_tilde_assume(context, right, left, vn, inds, vi)
    # Mutation of `value` no longer occurs in main body, so we do it here.
    left .= value
    return value, acclogp!!(vi_new, logp)
end

function dot_assume(
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    vi::SimpleVarInfo,
)
    @assert length(dist) == size(var, 1)
    # NOTE: We cannot work with `var` here because we might have a model of the form
    #
    #     m = Vector{Float64}(undef, n)
    #     m .~ Normal()
    #
    # in which case `var` will have `undef` elements, even if `m` is present in `vi`.
    value = vi[vns]
    lp = sum(zip(vns, eachcol(value))) do vn, val
        return Distributions.logpdf(dist, val)
    end
    return value, lp, vi
end

function dot_assume(
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    vi::SimpleVarInfo{<:NamedTuple},
)
    # NOTE: We cannot work with `var` here because we might have a model of the form
    #
    #     m = Vector{Float64}(undef, n)
    #     m .~ Normal()
    #
    # in which case `var` will have `undef` elements, even if `m` is present in `vi`.
    value = vi[vns]
    lp = sum(Distributions.logpdf.(dists, value))
    return value, lp, vi
end

# HACK: Allows us to re-use the implementation of `dot_tilde`, etc. for literals.
increment_num_produce!(::SimpleVarInfo) = nothing
settrans!!(vi::SimpleVarInfo, trans::Bool, vn::VarName) = vi

values_as(vi::SimpleVarInfo, ::Type{Dict}) = Dict(pairs(vi.θ))
values_as(vi::SimpleVarInfo, ::Type{NamedTuple}) = NamedTuple(pairs(vi.θ))
values_as(vi::SimpleVarInfo{<:NamedTuple}, ::Type{NamedTuple}) = vi.θ
