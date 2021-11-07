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

julia> using OrderedCollections: OrderedDict # ensures consisent output

julia> @model function demo()
           m ~ Normal()
           x = Vector{Float64}(undef, 2)
           for i in eachindex(x)
               x[i] ~ Normal()
           end
           return x
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> rng = StableRNG(42);

julia> ### Sampling ###
       ctx = SamplingContext(rng, SampleFromPrior(), DefaultContext());

julia> # In the `NamedTuple` version we need to provide the place-holder values for
       # the variablse which are using "containers", e.g. `Array`.
       # In this case, this means that we need to specify `x` but not `m`.
       _, vi = DynamicPPL.evaluate(m, SimpleVarInfo((x = ones(2), )), ctx); vi
SimpleVarInfo((x = [0.4471218424633827, 1.3736306979834252],), -4.024823883230379)

julia> # (✓) Vroom, vroom! FAST!!!
       DynamicPPL.getval(vi, @varname(x[1]))
0.4471218424633827

julia> # We can also access arbitrary varnames pointing to `x`, e.g.
       DynamicPPL.getval(vi, @varname(x))
2-element Vector{Float64}:
 0.4471218424633827
 1.3736306979834252

julia> DynamicPPL.getval(vi, @varname(x[1:2]))
2-element Vector{Float64}:
 0.4471218424633827
 1.3736306979834252

julia> # (×) If we don't provide the container...
       _, vi = DynamicPPL.evaluate(m, SimpleVarInfo(), ctx); vi
ERROR: type NamedTuple has no field x
[...]

julia> # If one does not know the varnames, we can use a `OrderedDict` instead.
       _, vi = DynamicPPL.evaluate(m, SimpleVarInfo{Float64}(OrderedDict()), ctx); vi
SimpleVarInfo(OrderedCollections.OrderedDict{Any, Any}(m => 0.683947930996541, x[1] => -1.019202452456547, x[2] => -0.7935128416361353), -3.8249261202386906)

julia> # (✓) Sort of fast, but only possible at runtime.
       DynamicPPL.getval(vi, @varname(x[1]))
-1.019202452456547

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

function Base.show(io::IO, ::MIME"text/plain", svi::SimpleVarInfo)
    print(io, "SimpleVarInfo(")
    print(io, svi.θ)
    print(io, ", ")
    print(io, svi.logp)
    return print(io, ")")
end

# `NamedTuple`
function getval(vi::SimpleVarInfo{<:NamedTuple}, vn::VarName)
    return get(vi.θ, vn)
end

# `Dict`
function getval(vi::SimpleVarInfo{<:AbstractDict}, vn::VarName{sym}) where {sym}
    # TODO: Should we maybe allow indexing of sub-keys, etc. too? E.g.
    # if `x` is present and it has an array, maybe we should allow indexing `x[1]`, etc.
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
    vn::VarName{sym,Setfield.IdentityLens},
    value,
    dist::Distribution,
    gidset::Set{Selector},
) where {sym}
    return Setfield.@set vi.θ = merge(vi.θ, NamedTuple{(sym,)}((value,)))
end
function push!!(
    vi::SimpleVarInfo{<:NamedTuple},
    vn::VarName{sym},
    value,
    dist::Distribution,
    gidset::Set{Selector},
) where {sym}
    return Setfield.@set vi.θ = set!!(vi.θ, vn, value)
end

# `Dict`
function push!!(
    vi::SimpleVarInfo{<:AbstractDict},
    vn::VarName,
    r,
    dist::Distribution,
    gidset::Set{Selector},
)
    vi.θ[vn] = r
    return vi
end

const SimpleOrThreadSafeSimple{T} = Union{
    SimpleVarInfo{T},ThreadSafeVarInfo{<:SimpleVarInfo{T}}
}

# Context implementations
function assume(dist::Distribution, vn::VarName, vi::SimpleOrThreadSafeSimple)
    left = vi[vn]
    return left, Distributions.loglikelihood(dist, left), vi
end

function assume(
    rng::Random.AbstractRNG,
    sampler::SampleFromPrior,
    dist::Distribution,
    vn::VarName,
    vi::SimpleOrThreadSafeSimple,
)
    value = init(rng, dist, sampler)
    vi = push!!(vi, vn, value, dist, sampler)
    return value, Distributions.loglikelihood(dist, value), vi
end

function dot_assume(
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    vi::SimpleOrThreadSafeSimple,
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
    vi::SimpleOrThreadSafeSimple,
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
increment_num_produce!(::SimpleOrThreadSafeSimple) = nothing
settrans!(vi::SimpleOrThreadSafeSimple, trans::Bool, vn::VarName) = nothing

values_as(vi::SimpleVarInfo, ::Type{Dict}) = Dict(pairs(vi.θ))
values_as(vi::SimpleVarInfo, ::Type{NamedTuple}) = NamedTuple(pairs(vi.θ))
values_as(vi::SimpleVarInfo{<:NamedTuple}, ::Type{NamedTuple}) = vi.θ
