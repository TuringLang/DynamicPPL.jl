abstract type AbstractTransformation end

struct NoTransformation <: AbstractTransformation end
struct DefaultTransformation <: AbstractTransformation end

"""
    $(TYPEDEF)

A simple wrapper of the parameters with a `logp` field for
accumulation of the logdensity.

Currently only implemented for `NT<:NamedTuple` and `NT<:AbstractDict`.

# Fields
$(FIELDS)

# Notes
The major differences between this and `TypedVarInfo` are:
1. `SimpleVarInfo` does not require linearization.
2. `SimpleVarInfo` can use more efficient bijectors.
3. `SimpleVarInfo` is only type-stable if `NT<:NamedTuple` and either
   a) no indexing is used in tilde-statements, or
   b) the values have been specified with the correct shapes.

# Examples
## General usage
```jldoctest simplevarinfo-general; setup=:(using Distributions)
julia> using StableRNGs

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
       # the variables which are using "containers", e.g. `Array`.
       # In this case, this means that we need to specify `x` but not `m`.
       _, vi = DynamicPPL.evaluate!!(m, SimpleVarInfo((x = ones(2), )), ctx);

julia> # (✓) Vroom, vroom! FAST!!!
       vi[@varname(x[1])]
0.4471218424633827

julia> # We can also access arbitrary varnames pointing to `x`, e.g.
       vi[@varname(x)]
2-element Vector{Float64}:
 0.4471218424633827
 1.3736306979834252

julia> vi[@varname(x[1:2])]
2-element Vector{Float64}:
 0.4471218424633827
 1.3736306979834252

julia> # (×) If we don't provide the container...
       _, vi = DynamicPPL.evaluate!!(m, SimpleVarInfo(), ctx); vi
ERROR: type NamedTuple has no field x
[...]

julia> # If one does not know the varnames, we can use a `OrderedDict` instead.
       _, vi = DynamicPPL.evaluate!!(m, SimpleVarInfo{Float64}(OrderedDict()), ctx);

julia> # (✓) Sort of fast, but only possible at runtime.
       vi[@varname(x[1])]
-1.019202452456547

julia> # In addtion, we can only access varnames as they appear in the model!
       vi[@varname(x)]
ERROR: KeyError: key x not found
[...]

julia> vi[@varname(x[1:2])]
ERROR: KeyError: key x[1:2] not found
[...]
```

_Technically_, it's possible to use any implementation of `AbstractDict` in place of
`OrderedDict`, but `OrderedDict` ensures that certain operations, e.g. linearization/flattening
of the values in the varinfo, are consistent between evaluations. Hence `OrderedDict` is
the preferred implementation of `AbstractDict` to use here.

You can also sample in _transformed_ space:

```jldoctest simplevarinfo-general
julia> @model demo_constrained() = x ~ Exponential()
demo_constrained (generic function with 2 methods)

julia> m = demo_constrained();

julia> _, vi = DynamicPPL.evaluate!!(m, SimpleVarInfo(), ctx);

julia> vi[@varname(x)] # (✓) 0 ≤ x < ∞
1.8632965762164932

julia> _, vi = DynamicPPL.evaluate!!(m, DynamicPPL.settrans!!(SimpleVarInfo(), true), ctx);

julia> vi[@varname(x)] # (✓) -∞ < x < ∞
-0.21080155351918753

julia> xs = [last(DynamicPPL.evaluate!!(m, DynamicPPL.settrans!!(SimpleVarInfo(), true), ctx))[@varname(x)] for i = 1:10];

julia> any(xs .< 0)  # (✓) Positive probability mass on negative numbers!
true

julia> # And with `OrderedDict` of course!
       _, vi = DynamicPPL.evaluate!!(m, DynamicPPL.settrans!!(SimpleVarInfo(OrderedDict()), true), ctx);

julia> vi[@varname(x)] # (✓) -∞ < x < ∞
0.6225185067787314

julia> xs = [last(DynamicPPL.evaluate!!(m, DynamicPPL.settrans!!(SimpleVarInfo(), true), ctx))[@varname(x)] for i = 1:10];

julia> any(xs .< 0) # (✓) Positive probability mass on negative numbers!
true
```

Evaluation in transformed space of course also works:

```jldoctest simplevarinfo-general
julia> vi = DynamicPPL.settrans!!(SimpleVarInfo((x = -1.0,)), true)
Transformed SimpleVarInfo((x = -1.0,), 0.0)

julia> # (✓) Positive probability mass on negative numbers!
       getlogp(last(DynamicPPL.evaluate!!(m, vi, DynamicPPL.DefaultContext())))
-1.3678794411714423

julia> # While if we forget to make indicate that it's transformed:
       vi = DynamicPPL.settrans!!(SimpleVarInfo((x = -1.0,)), false)
SimpleVarInfo((x = -1.0,), 0.0)

julia> # (✓) No probability mass on negative numbers!
       getlogp(last(DynamicPPL.evaluate!!(m, vi, DynamicPPL.DefaultContext())))
-Inf
```

## Indexing
Using `NamedTuple` as underlying storage.

```jldoctest
julia> svi_nt = SimpleVarInfo((m = (a = [1.0], ), ));

julia> svi_nt[@varname(m)]
(a = [1.0],)

julia> svi_nt[@varname(m.a)]
1-element Vector{Float64}:
 1.0

julia> svi_nt[@varname(m.a[1])]
1.0

julia> svi_nt[@varname(m.a[2])]
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]

julia> svi_nt[@varname(m.b)]
ERROR: type NamedTuple has no field b
[...]
```

Using `OrderedDict` as underlying storage.
```jldoctest
julia> svi_dict = SimpleVarInfo(OrderedDict(@varname(m) => (a = [1.0], )));

julia> svi_dict[@varname(m)]
(a = [1.0],)

julia> svi_dict[@varname(m.a)]
1-element Vector{Float64}:
 1.0

julia> svi_dict[@varname(m.a[1])]
1.0

julia> svi_dict[@varname(m.a[2])]
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]

julia> svi_dict[@varname(m.b)]
ERROR: type NamedTuple has no field b
[...]
```
"""
struct SimpleVarInfo{NT,T,C<:AbstractTransformation} <: AbstractVarInfo
    "underlying representation of the realization represented"
    values::NT
    "holds the accumulated log-probability"
    logp::T
    "represents whether it assumes variables to be transformed"
    transformation::C
end

# Makes things a bit more readable vs. putting `Float64` everywhere.
const SIMPLEVARINFO_DEFAULT_ELTYPE = Float64

function SimpleVarInfo{NT,T}(values, logp) where {NT,T}
    return SimpleVarInfo{NT,T,NoTransformation}(values, logp, NoTransformation())
end
function SimpleVarInfo{T}(θ) where {T<:Real}
    return SimpleVarInfo{typeof(θ),T}(θ, zero(T))
end

# Constructors without type-specification.
SimpleVarInfo(θ) = SimpleVarInfo{SIMPLEVARINFO_DEFAULT_ELTYPE}(θ)
function SimpleVarInfo(θ::Union{<:NamedTuple,<:AbstractDict})
    return if isempty(θ)
        # Can't infer from values, so we just use default.
        SimpleVarInfo{SIMPLEVARINFO_DEFAULT_ELTYPE}(θ)
    else
        # Infer from `values`.
        SimpleVarInfo{float_type_with_fallback(infer_nested_eltype(typeof(θ)))}(θ)
    end
end

SimpleVarInfo(values, logp) = SimpleVarInfo{typeof(values),typeof(logp)}(values, logp)

# Using `kwargs` to specify the values.
function SimpleVarInfo{T}(; kwargs...) where {T<:Real}
    return SimpleVarInfo{T}(NamedTuple(kwargs))
end
function SimpleVarInfo(; kwargs...)
    return SimpleVarInfo(NamedTuple(kwargs))
end

# Constructor from `Model`.
SimpleVarInfo(model::Model, args...) = SimpleVarInfo{Float64}(model, args...)
function SimpleVarInfo{T}(model::Model, args...) where {T<:Real}
    return last(evaluate!!(model, SimpleVarInfo{T}(), args...))
end

# Constructor from `VarInfo`.
function SimpleVarInfo(vi::TypedVarInfo, ::Type{D}=NamedTuple; kwargs...) where {D}
    return SimpleVarInfo{eltype(getlogp(vi))}(vi, D; kwargs...)
end
function SimpleVarInfo{T}(
    vi::VarInfo{<:NamedTuple{names}}, ::Type{D}
) where {T<:Real,names,D}
    values = values_as(vi, D)
    return SimpleVarInfo(values, convert(T, getlogp(vi)))
end

function BangBang.empty!!(vi::SimpleVarInfo)
    Setfield.@set resetlogp!!(vi).values = empty!!(vi.values)
end

getlogp(vi::SimpleVarInfo) = vi.logp
setlogp!!(vi::SimpleVarInfo, logp) = Setfield.@set vi.logp = logp
acclogp!!(vi::SimpleVarInfo, logp) = Setfield.@set vi.logp = getlogp(vi) + logp

"""
    keys(vi::SimpleVarInfo)

Return an iterator of keys present in `vi`.
"""
Base.keys(vi::SimpleVarInfo) = keys(vi.values)
Base.keys(vi::SimpleVarInfo{<:NamedTuple}) = map(k -> VarName{k}(), keys(vi.values))

function setlogp!!(vi::SimpleVarInfo{<:Any,<:Ref}, logp)
    vi.logp[] = logp
    return vi
end

function acclogp!!(vi::SimpleVarInfo{<:Any,<:Ref}, logp)
    vi.logp[] += logp
    return vi
end

function Base.show(io::IO, ::MIME"text/plain", svi::SimpleVarInfo)
    if !(svi.transformation isa NoTransformation)
        print(io, "Transformed ")
    end

    return print(io, "SimpleVarInfo(", svi.values, ", ", svi.logp, ")")
end

# `NamedTuple`
function Base.getindex(vi::SimpleVarInfo, vn::VarName, dist::Distribution)
    return maybe_invlink(vi, vn, dist, getindex(vi, vn))
end
function Base.getindex(vi::SimpleVarInfo, vns::Vector{<:VarName}, dist::Distribution)
    vals_linked = mapreduce(vcat, vns) do vn
        getindex(vi, vn, dist)
    end
    return reconstruct(dist, vals_linked, length(vns))
end

Base.getindex(vi::SimpleVarInfo, vn::VarName) = get(vi.values, vn)

# `AbstractDict`
function Base.getindex(vi::SimpleVarInfo{<:AbstractDict}, vn::VarName)
    return nested_getindex(vi.values, vn)
end

# `SimpleVarInfo` doesn't necessarily vectorize, so we can have arrays other than
# just `Vector`.
function Base.getindex(vi::SimpleVarInfo, vns::AbstractArray{<:VarName})
    return map(Base.Fix1(getindex, vi), vns)
end
# HACK: Needed to disambiguiate.
Base.getindex(vi::SimpleVarInfo, vns::Vector{<:VarName}) = map(Base.Fix1(getindex, vi), vns)

Base.getindex(vi::SimpleVarInfo, spl::SampleFromPrior) = vi.values
Base.getindex(vi::SimpleVarInfo, spl::SampleFromUniform) = vi.values

# TODO: Should we do better?
Base.getindex(vi::SimpleVarInfo, spl::Sampler) = vi.values

# Since we don't perform any transformations in `getindex` for `SimpleVarInfo`
# we simply call `getindex` in `getindex_raw`.
getindex_raw(vi::SimpleVarInfo, vn::VarName) = vi[vn]
function getindex_raw(vi::SimpleVarInfo, vn::VarName, dist::Distribution)
    return reconstruct(dist, getindex_raw(vi, vn))
end
getindex_raw(vi::SimpleVarInfo, vns::Vector{<:VarName}) = vi[vns]
function getindex_raw(vi::SimpleVarInfo, vns::Vector{<:VarName}, dist::Distribution)
    # `reconstruct` expects a flattened `Vector` regardless of the type of `dist`, so we `vcat` everything.
    vals = mapreduce(Base.Fix1(getindex_raw, vi), vcat, vns)
    return reconstruct(dist, vals, length(vns))
end

Base.haskey(vi::SimpleVarInfo, vn::VarName) = nested_haskey(vi.values, vn)

function BangBang.setindex!!(vi::SimpleVarInfo, val, vn::VarName)
    # For `NamedTuple` we treat the symbol in `vn` as the _property_ to set.
    return Setfield.@set vi.values = set!!(vi.values, vn, val)
end

# TODO: Specialize to handle certain cases, e.g. a collection of `VarName` with
# same symbol and same type of, say, `IndexLens`, for improved `.~` performance.
function BangBang.setindex!!(vi::SimpleVarInfo, vals, vns::AbstractVector{<:VarName})
    for (vn, val) in zip(vns, vals)
        vi = BangBang.setindex!!(vi, val, vn)
    end
    return vi
end

function BangBang.setindex!!(vi::SimpleVarInfo{<:AbstractDict}, val, vn::VarName)
    # For dictlike objects, we treat the entire `vn` as a _key_ to set.
    dict = values_as(vi)
    # Attempt to split into `parent` and `child` lenses.
    parent, child, issuccess = splitlens(getlens(vn)) do lens
        l = lens === nothing ? Setfield.IdentityLens() : lens
        haskey(dict, VarName(vn, l))
    end
    # When combined with `VarInfo`, `nothing` is equivalent to `IdentityLens`.
    keylens = parent === nothing ? Setfield.IdentityLens() : parent

    dict_new = if !issuccess
        # Split doesn't exist ⟹ we're working with a new key.
        BangBang.setindex!!(dict, val, vn)
    else
        # Split exists ⟹ trying to set an existing key.
        vn_key = VarName(vn, keylens)
        BangBang.setindex!!(dict, set!!(dict[vn_key], child, val), vn_key)
    end
    return Setfield.@set vi.values = dict_new
end

# `NamedTuple`
function BangBang.push!!(
    vi::SimpleVarInfo{<:NamedTuple},
    vn::VarName{sym,Setfield.IdentityLens},
    value,
    dist::Distribution,
    gidset::Set{Selector},
) where {sym}
    return Setfield.@set vi.values = merge(vi.values, NamedTuple{(sym,)}((value,)))
end
function BangBang.push!!(
    vi::SimpleVarInfo{<:NamedTuple},
    vn::VarName{sym},
    value,
    dist::Distribution,
    gidset::Set{Selector},
) where {sym}
    return Setfield.@set vi.values = set!!(vi.values, vn, value)
end

# `AbstractDict`
function BangBang.push!!(
    vi::SimpleVarInfo{<:AbstractDict},
    vn::VarName,
    r,
    dist::Distribution,
    gidset::Set{Selector},
)
    vi.values[vn] = r
    return vi
end

const SimpleOrThreadSafeSimple{T,V,C} = Union{
    SimpleVarInfo{T,V,C},ThreadSafeVarInfo{<:SimpleVarInfo{T,V,C}}
}

# Necessary for `matchingvalue` to work properly.
function Base.eltype(
    vi::SimpleOrThreadSafeSimple{<:Any,V}, spl::Union{AbstractSampler,SampleFromPrior}
) where {V}
    return V
end

# Context implementations
# NOTE: Evaluations, i.e. those without `rng` are shared with other
# implementations of `AbstractVarInfo`.
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi::SimpleOrThreadSafeSimple,
)
    value = init(rng, dist, sampler)
    # Transform if we're working in unconstrained space.
    value_raw = maybe_link(vi, vn, dist, value)
    vi = BangBang.push!!(vi, vn, value_raw, dist, sampler)
    return value, Bijectors.logpdf_with_trans(dist, value, istrans(vi, vn)), vi
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::SimpleOrThreadSafeSimple,
)
    f = (vn, dist) -> init(rng, dist, spl)
    value = f.(vns, dists)

    # Transform if we're working in transformed space.
    value_raw = if dists isa Distribution
        maybe_link.((vi,), vns, (dists,), value)
    else
        maybe_link.((vi,), vns, dists, value)
    end

    # Update `vi`
    vi = BangBang.setindex!!(vi, value_raw, vns)

    # Compute logp.
    lp = sum(Bijectors.logpdf_with_trans.(dists, value, istrans.((vi,), vns)))
    return value, lp, vi
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi::SimpleOrThreadSafeSimple,
)
    @assert length(dist) == size(var, 1) "dimensionality of `var` ($(size(var, 1))) is incompatible with dimensionality of `dist` $(length(dist))"

    # r = get_and_set_val!(rng, vi, vns, dist, spl)
    n = length(vns)
    value = init(rng, dist, spl, n)

    # Update `vi`.
    for (vn, val) in zip(vns, eachcol(value))
        val_linked = maybe_link(vi, vn, dist, val)
        vi = BangBang.setindex!!(vi, val_linked, vn)
    end

    # Compute logp.
    lp = sum(Bijectors.logpdf_with_trans(dist, value, istrans(vi)))
    return value, lp, vi
end

# HACK: Allows us to re-use the implementation of `dot_tilde`, etc. for literals.
increment_num_produce!(::SimpleOrThreadSafeSimple) = nothing

# NOTE: We don't implement `settrans!!(vi, trans, vn)`.
function settrans!!(vi::SimpleVarInfo, trans)
    return settrans!!(vi, trans ? DefaultTransformation() : NoTransformation())
end
function settrans!!(vi::SimpleVarInfo, transformation::AbstractTransformation)
    return Setfield.@set vi.transformation = transformation
end
function settrans!!(vi::ThreadSafeVarInfo{<:SimpleVarInfo}, trans)
    return Setfield.@set vi.varinfo = settrans!!(vi.varinfo, trans)
end

istrans(vi::SimpleVarInfo) = !(vi.transformation isa NoTransformation)
istrans(vi::SimpleVarInfo, vn::VarName) = istrans(vi)
istrans(vi::ThreadSafeVarInfo{<:SimpleVarInfo}, vn::VarName) = istrans(vi.varinfo, vn)

values_as(vi::SimpleVarInfo) = vi.values
values_as(vi::SimpleVarInfo{<:T}, ::Type{T}) where {T} = vi.values
function values_as(vi::SimpleVarInfo, ::Type{D}) where {D<:AbstractDict}
    return ConstructionBase.constructorof(D)(zip(keys(vi), values(vi.values)))
end
function values_as(vi::SimpleVarInfo{<:AbstractDict}, ::Type{NamedTuple})
    return NamedTuple((Symbol(k), v) for (k, v) in vi.values)
end

"""
    logjoint(model::Model, θ)

Return the log joint probability of variables `θ` for the probabilistic `model`.

See [`logjoint`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 2 methods)

julia> # Using a `NamedTuple`.
       logjoint(demo([1.0]), (m = 100.0, ))
-9902.33787706641

julia> # Using a `Dict`.
       logjoint(demo([1.0]), Dict(@varname(m) => 100.0))
-9902.33787706641

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0) + logpdf(Normal(), 100.0)
-9902.33787706641
```
"""
logjoint(model::Model, θ) = logjoint(model, SimpleVarInfo(θ))

"""
    logprior(model::Model, θ)

Return the log prior probability of variables `θ` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 2 methods)

julia> # Using a `NamedTuple`.
       logprior(demo([1.0]), (m = 100.0, ))
-5000.918938533205

julia> # Using a `Dict`.
       logprior(demo([1.0]), Dict(@varname(m) => 100.0))
-5000.918938533205

julia> # Truth.
       logpdf(Normal(), 100.0)
-5000.918938533205
```
"""
logprior(model::Model, θ) = logprior(model, SimpleVarInfo(θ))

"""
    loglikelihood(model::Model, θ)

Return the log likelihood of variables `θ` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`logprior`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 2 methods)

julia> # Using a `NamedTuple`.
       loglikelihood(demo([1.0]), (m = 100.0, ))
-4901.418938533205

julia> # Using a `Dict`.
       loglikelihood(demo([1.0]), Dict(@varname(m) => 100.0))
-4901.418938533205

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0)
-4901.418938533205
```
"""
Distributions.loglikelihood(model::Model, θ) = loglikelihood(model, SimpleVarInfo(θ))

# Threadsafe stuff.
# For `SimpleVarInfo` we don't really need `Ref` so let's not use it.
function ThreadSafeVarInfo(vi::SimpleVarInfo)
    return ThreadSafeVarInfo(vi, zeros(typeof(getlogp(vi)), Threads.nthreads()))
end
function ThreadSafeVarInfo(vi::SimpleVarInfo{<:Any,<:Ref})
    return ThreadSafeVarInfo(vi, [Ref(zero(getlogp(vi))) for _ in 1:Threads.nthreads()])
end
