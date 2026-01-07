"""
    $(TYPEDEF)

A simple wrapper of the parameters with a `logp` field for
accumulation of the logdensity.

Currently only implemented for `NT<:NamedTuple` and `NT<:AbstractDict`.

# Fields
$(FIELDS)

# Notes
The major differences between this and `NTVarInfo` are:
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

julia> # In the `NamedTuple` version we need to provide the place-holder values for
       # the variables which are using "containers", e.g. `Array`.
       # In this case, this means that we need to specify `x` but not `m`.
       _, vi = DynamicPPL.init!!(rng, m, SimpleVarInfo((x = ones(2), )));

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
       _, vi = DynamicPPL.init!!(rng, m, SimpleVarInfo());
ERROR: FieldError: type NamedTuple has no field `x`, available fields: `m`
[...]

julia> # If one does not know the varnames, we can use a `OrderedDict` instead.
       _, vi = DynamicPPL.init!!(rng, m, SimpleVarInfo{Float64}(OrderedDict{VarName,Any}()));

julia> # (✓) Sort of fast, but only possible at runtime.
       vi[@varname(x[1])]
-1.019202452456547

julia> # In addtion, we can only access varnames as they appear in the model!
       vi[@varname(x)]
ERROR: x was not found in the dictionary provided
[...]

julia> vi[@varname(x[1:2])]
ERROR: x[1:2] was not found in the dictionary provided
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

julia> _, vi = DynamicPPL.init!!(rng, m, SimpleVarInfo());

julia> vi[@varname(x)] # (✓) 0 ≤ x < ∞
1.8632965762164932

julia> _, vi = DynamicPPL.init!!(rng, m, DynamicPPL.set_transformed!!(SimpleVarInfo(), true));

julia> vi[@varname(x)] # (✓) -∞ < x < ∞
-0.21080155351918753

julia> xs = [last(DynamicPPL.init!!(rng, m, DynamicPPL.set_transformed!!(SimpleVarInfo(), true)))[@varname(x)] for i = 1:10];

julia> any(xs .< 0)  # (✓) Positive probability mass on negative numbers!
true

julia> # And with `OrderedDict` of course!
       _, vi = DynamicPPL.init!!(rng, m, DynamicPPL.set_transformed!!(SimpleVarInfo(OrderedDict{VarName,Any}()), true));

julia> vi[@varname(x)] # (✓) -∞ < x < ∞
0.6225185067787314

julia> xs = [last(DynamicPPL.init!!(rng, m, DynamicPPL.set_transformed!!(SimpleVarInfo(), true)))[@varname(x)] for i = 1:10];

julia> any(xs .< 0) # (✓) Positive probability mass on negative numbers!
true
```

Evaluation in transformed space of course also works:

```jldoctest simplevarinfo-general
julia> vi = DynamicPPL.set_transformed!!(SimpleVarInfo((x = -1.0,)), true)
Transformed SimpleVarInfo((x = -1.0,), (LogPrior = LogPriorAccumulator(0.0), LogJacobian = LogJacobianAccumulator(0.0), LogLikelihood = LogLikelihoodAccumulator(0.0)))

julia> # (✓) Positive probability mass on negative numbers!
       getlogjoint_internal(last(DynamicPPL.evaluate!!(m, vi)))
-1.3678794411714423

julia> # While if we forget to indicate that it's transformed:
       vi = DynamicPPL.set_transformed!!(SimpleVarInfo((x = -1.0,)), false)
SimpleVarInfo((x = -1.0,), (LogPrior = LogPriorAccumulator(0.0), LogJacobian = LogJacobianAccumulator(0.0), LogLikelihood = LogLikelihoodAccumulator(0.0)))

julia> # (✓) No probability mass on negative numbers!
       getlogjoint_internal(last(DynamicPPL.evaluate!!(m, vi)))
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
ERROR: FieldError: type NamedTuple has no field `b`, available fields: `a`
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
ERROR: m.a[2] was not found in the dictionary provided
[...]

julia> svi_dict[@varname(m.b)]
ERROR: m.b was not found in the dictionary provided
[...]
```
"""
struct SimpleVarInfo{NT,Accs<:AccumulatorTuple where {N},C<:AbstractTransformation} <:
       AbstractVarInfo
    "underlying representation of the realization represented"
    values::NT
    "tuple of accumulators for things like log prior and log likelihood"
    accs::Accs
    "represents whether it assumes variables to be transformed"
    transformation::C
end

function Base.:(==)(vi1::SimpleVarInfo, vi2::SimpleVarInfo)
    return vi1.values == vi2.values &&
           vi1.accs == vi2.accs &&
           vi1.transformation == vi2.transformation
end

transformation(vi::SimpleVarInfo) = vi.transformation

function SimpleVarInfo(values, accs)
    return SimpleVarInfo(values, accs, NoTransformation())
end
function SimpleVarInfo{T}(values) where {T<:Real}
    return SimpleVarInfo(values, default_accumulators(T))
end
function SimpleVarInfo(values)
    return SimpleVarInfo{LogProbType}(values)
end
function SimpleVarInfo(values::Union{<:NamedTuple,<:AbstractDict{<:VarName}})
    return if isempty(values)
        # Can't infer from values, so we just use default.
        SimpleVarInfo{LogProbType}(values)
    else
        # Infer from `values`.
        SimpleVarInfo{float_type_with_fallback(infer_nested_eltype(typeof(values)))}(values)
    end
end

# Using `kwargs` to specify the values.
function SimpleVarInfo{T}(; kwargs...) where {T<:Real}
    return SimpleVarInfo{T}(NamedTuple(kwargs))
end
function SimpleVarInfo(; kwargs...)
    return SimpleVarInfo(NamedTuple(kwargs))
end

# Constructor from `Model`.
function SimpleVarInfo{T}(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
) where {T<:Real}
    return last(init!!(rng, model, SimpleVarInfo{T}(), init_strategy))
end
function SimpleVarInfo{T}(
    model::Model, init_strategy::AbstractInitStrategy=InitFromPrior()
) where {T<:Real}
    return SimpleVarInfo{T}(Random.default_rng(), model, init_strategy)
end
# Constructors without type param
function SimpleVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return SimpleVarInfo{LogProbType}(rng, model, init_strategy)
end
function SimpleVarInfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return SimpleVarInfo{LogProbType}(Random.default_rng(), model, init_strategy)
end

# Constructor from `VarInfo`.
function SimpleVarInfo(vi::NTVarInfo, ::Type{D}) where {D}
    values = values_as(vi, D)
    return SimpleVarInfo(values, copy(getaccs(vi)))
end
function SimpleVarInfo{T}(vi::NTVarInfo, ::Type{D}) where {T<:Real,D}
    values = values_as(vi, D)
    accs = map(acc -> convert_eltype(T, acc), getaccs(vi))
    return SimpleVarInfo(values, accs)
end

function untyped_simple_varinfo(model::Model)
    varinfo = SimpleVarInfo(OrderedDict{VarName,Any}())
    return last(init!!(model, varinfo))
end

function typed_simple_varinfo(model::Model)
    varinfo = SimpleVarInfo{Float64}()
    return last(init!!(model, varinfo))
end

function unflatten(svi::SimpleVarInfo, x::AbstractVector)
    vals = unflatten(svi.values, x)
    return SimpleVarInfo(vals, svi.accs, svi.transformation)
end

function BangBang.empty!!(vi::SimpleVarInfo)
    return resetaccs!!(Accessors.@set vi.values = empty!!(vi.values))
end
Base.isempty(vi::SimpleVarInfo) = isempty(vi.values)

getaccs(vi::SimpleVarInfo) = vi.accs
setaccs!!(vi::SimpleVarInfo, accs::AccumulatorTuple) = Accessors.@set vi.accs = accs

"""
    keys(vi::SimpleVarInfo)

Return an iterator of keys present in `vi`.
"""
Base.keys(vi::SimpleVarInfo) = keys(vi.values)
Base.keys(vi::SimpleVarInfo{<:NamedTuple}) = map(k -> VarName{k}(), keys(vi.values))

function Base.show(io::IO, mime::MIME"text/plain", svi::SimpleVarInfo)
    if !(svi.transformation isa NoTransformation)
        print(io, "Transformed ")
    end

    return print(io, "SimpleVarInfo(", svi.values, ", ", repr(mime, getaccs(svi)), ")")
end

function Base.getindex(vi::SimpleVarInfo, vn::VarName, dist::Distribution)
    return from_maybe_linked_internal(vi, vn, dist, getindex(vi, vn))
end
function Base.getindex(vi::SimpleVarInfo, vns::Vector{<:VarName}, dist::Distribution)
    vals_linked = mapreduce(vcat, vns) do vn
        getindex(vi, vn, dist)
    end
    return recombine(dist, vals_linked, length(vns))
end

Base.getindex(vi::SimpleVarInfo, vn::VarName) = getindex_internal(vi, vn)

# `SimpleVarInfo` doesn't necessarily vectorize, so we can have arrays other than
# just `Vector`.
function Base.getindex(vi::SimpleVarInfo, vns::AbstractArray{<:VarName})
    return map(Base.Fix1(getindex, vi), vns)
end
# HACK: Needed to disambiguate.
Base.getindex(vi::SimpleVarInfo, vns::Vector{<:VarName}) = map(Base.Fix1(getindex, vi), vns)

Base.getindex(svi::SimpleVarInfo, ::Colon) = values_as(svi, Vector)

getindex_internal(vi::SimpleVarInfo, vn::VarName) = get(vi.values, vn)
# `AbstractDict`
function getindex_internal(
    vi::SimpleVarInfo{<:Union{AbstractDict,VarNamedVector}}, vn::VarName
)
    return getvalue(vi.values, vn)
end

Base.haskey(vi::SimpleVarInfo, vn::VarName) = hasvalue(vi.values, vn)

function BangBang.setindex!!(vi::SimpleVarInfo, val, vn::VarName)
    # For `NamedTuple` we treat the symbol in `vn` as the _property_ to set.
    return Accessors.@set vi.values = set!!(vi.values, vn, val)
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
    # Attempt to split into `parent` and `child` optic.
    parent, child, issuccess = splitoptic(getoptic(vn)) do optic
        o = optic === nothing ? identity : optic
        haskey(dict, VarName{getsym(vn)}(o))
    end
    # When combined with `VarInfo`, `nothing` is equivalent to `identity`.
    keyoptic = parent === nothing ? identity : parent

    dict_new = if !issuccess
        # Split doesn't exist ⟹ we're working with a new key.
        BangBang.setindex!!(dict, val, vn)
    else
        # Split exists ⟹ trying to set an existing key.
        vn_key = VarName{getsym(vn)}(keyoptic)
        BangBang.setindex!!(dict, set!!(dict[vn_key], child, val), vn_key)
    end
    return Accessors.@set vi.values = dict_new
end

# `NamedTuple`
function BangBang.push!!(
    vi::SimpleVarInfo{<:NamedTuple}, ::VarName{sym,typeof(identity)}, value, ::Distribution
) where {sym}
    return Accessors.@set vi.values = merge(vi.values, NamedTuple{(sym,)}((value,)))
end
function BangBang.push!!(
    vi::SimpleVarInfo{<:NamedTuple}, vn::VarName{sym}, value, ::Distribution
) where {sym}
    return Accessors.@set vi.values = set!!(vi.values, vn, value)
end

# `AbstractDict`
function BangBang.push!!(
    vi::SimpleVarInfo{<:AbstractDict}, vn::VarName, value, ::Distribution
)
    vi.values[vn] = value
    return vi
end

function BangBang.push!!(
    vi::SimpleVarInfo{<:VarNamedVector}, vn::VarName, value, ::Distribution
)
    # The semantics of push!! for SimpleVarInfo and VarNamedVector are different. For
    # SimpleVarInfo, push!! allows the key to exist already, for VarNamedVector it does not.
    # Hence we need to call update!! here, which has the same semantics as push!! does for
    # SimpleVarInfo.
    return Accessors.@set vi.values = setindex!!(vi.values, value, vn)
end

const SimpleOrThreadSafeSimple{T,V,C} = Union{
    SimpleVarInfo{T,V,C},ThreadSafeVarInfo{<:SimpleVarInfo{T,V,C}}
}

# Necessary for `matchingvalue` to work properly.
Base.eltype(svi::SimpleVarInfo) = infer_nested_eltype(typeof(svi.values))
Base.eltype(tsvi::ThreadSafeVarInfo{<:SimpleVarInfo}) = eltype(tsvi.varinfo)

# `subset`
function subset(varinfo::SimpleVarInfo, vns::AbstractVector{<:VarName})
    return SimpleVarInfo(
        _subset(varinfo.values, vns), map(copy, getaccs(varinfo)), varinfo.transformation
    )
end

function _subset(x::AbstractDict, vns::AbstractVector{VN}) where {VN<:VarName}
    vns_present = collect(keys(x))
    vns_found = filter(
        vn_present -> any(subsumes(vn, vn_present) for vn in vns), vns_present
    )
    C = ConstructionBase.constructorof(typeof(x))
    if isempty(vns_found)
        return C()
    else
        return C(vn => x[vn] for vn in vns_found)
    end
end

function _subset(x::NamedTuple, vns)
    # NOTE: Here we can only handle `vns` that contain `identity` as optic.
    if any(Base.Fix1(!==, identity) ∘ getoptic, vns)
        throw(
            ArgumentError(
                "Cannot subset `NamedTuple` with non-`identity` `VarName`. " *
                "For example, `@varname(x)` is allowed, but `@varname(x[1])` is not.",
            ),
        )
    end

    syms = map(getsym, vns)
    x_syms = filter(Base.Fix2(in, syms), keys(x))
    return NamedTuple{Tuple(x_syms)}(Tuple(map(Base.Fix1(getindex, x), x_syms)))
end

_subset(x::VarNamedVector, vns) = subset(x, vns)

# `merge`
function Base.merge(varinfo_left::SimpleVarInfo, varinfo_right::SimpleVarInfo)
    values = merge(varinfo_left.values, varinfo_right.values)
    accs = map(copy, getaccs(varinfo_right))
    transformation = merge_transformations(
        varinfo_left.transformation, varinfo_right.transformation
    )
    return SimpleVarInfo(values, accs, transformation)
end

function set_transformed!!(vi::SimpleVarInfo, trans)
    return set_transformed!!(vi, trans ? DynamicTransformation() : NoTransformation())
end
function set_transformed!!(vi::SimpleVarInfo, transformation::AbstractTransformation)
    return Accessors.@set vi.transformation = transformation
end
function set_transformed!!(vi::ThreadSafeVarInfo{<:SimpleVarInfo}, trans)
    return Accessors.@set vi.varinfo = set_transformed!!(vi.varinfo, trans)
end
function set_transformed!!(vi::SimpleOrThreadSafeSimple, trans::Bool, ::VarName)
    # We keep this method around just to obey the AbstractVarInfo interface.
    # However, note that this would only be a valid operation if it would be a
    # no-op, which we check here.
    if trans != is_transformed(vi)
        error(
            "Individual variables in SimpleVarInfo cannot have different `set_transformed` statuses.",
        )
    end
    return vi
end

is_transformed(vi::SimpleVarInfo) = !(vi.transformation isa NoTransformation)
is_transformed(vi::SimpleVarInfo, ::VarName) = is_transformed(vi)
function is_transformed(vi::ThreadSafeVarInfo{<:SimpleVarInfo}, vn::VarName)
    return is_transformed(vi.varinfo, vn)
end
is_transformed(vi::ThreadSafeVarInfo{<:SimpleVarInfo}) = is_transformed(vi.varinfo)

values_as(vi::SimpleVarInfo) = vi.values
values_as(vi::SimpleVarInfo{<:T}, ::Type{T}) where {T} = vi.values
function values_as(vi::SimpleVarInfo, ::Type{Vector})
    isempty(vi) && return Any[]
    return mapreduce(tovec, vcat, values(vi.values))
end
function values_as(vi::SimpleVarInfo, ::Type{D}) where {D<:AbstractDict}
    return ConstructionBase.constructorof(D)(zip(keys(vi), values(vi.values)))
end
function values_as(vi::SimpleVarInfo{<:AbstractDict}, ::Type{NamedTuple})
    return NamedTuple((Symbol(k), v) for (k, v) in vi.values)
end
function values_as(vi::SimpleVarInfo, ::Type{T}) where {T}
    return values_as(vi.values, T)
end

# Allow usage of `NamedBijector` too.
function link!!(
    t::StaticTransformation{<:Bijectors.NamedTransform},
    vi::SimpleVarInfo{<:NamedTuple},
    ::Model,
)
    b = inverse(t.bijector)
    x = vi.values
    y, logjac = with_logabsdet_jacobian(b, x)
    vi_new = Accessors.@set(vi.values = y)
    if hasacc(vi_new, Val(:LogJacobian))
        vi_new = acclogjac!!(vi_new, logjac)
    end
    return set_transformed!!(vi_new, t)
end

function invlink!!(
    t::StaticTransformation{<:Bijectors.NamedTransform},
    vi::SimpleVarInfo{<:NamedTuple},
    ::Model,
)
    b = t.bijector
    y = vi.values
    x, inv_logjac = with_logabsdet_jacobian(b, y)
    vi_new = Accessors.@set(vi.values = x)
    # Mildly confusing: we need to _add_ the logjac of the inverse transform,
    # because we are trying to remove the logjac of the forward transform
    # that was previously accumulated when linking.
    if hasacc(vi_new, Val(:LogJacobian))
        vi_new = acclogjac!!(vi_new, inv_logjac)
    end
    return set_transformed!!(vi_new, NoTransformation())
end

# With `SimpleVarInfo`, when we're not working with linked variables, there's no need to do anything.
from_internal_transform(vi::SimpleVarInfo, ::VarName) = identity
from_internal_transform(vi::SimpleVarInfo, ::VarName, dist) = identity
# TODO: Should the following methods specialize on the case where we have a `StaticTransformation{<:Bijectors.NamedTransform}`?
from_linked_internal_transform(vi::SimpleVarInfo, ::VarName) = identity
function from_linked_internal_transform(vi::SimpleVarInfo, ::VarName, dist)
    return invlink_transform(dist)
end

has_varnamedvector(vi::SimpleVarInfo) = vi.values isa VarNamedVector
