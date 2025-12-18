struct VNTVarInfo{T<:VarNamedTuple,Accs<:AccumulatorTuple} <: AbstractVarInfo
    values::T
    accs::Accs
end

# TODO(mhauru) Make this renaming permanent.
const VarInfo = VNTVarInfo

struct TransformedValue{ValType,TransformType}
    val::ValType
    linked::Bool
    transform::TransformType
end

VNTVarInfo() = VNTVarInfo(VarNamedTuple(), default_accumulators())

function VNTVarInfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return VNTVarInfo(Random.default_rng(), model, init_strategy)
end

function VNTVarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return last(init!!(rng, model, VNTVarInfo(), init_strategy))
end

getaccs(vi::VNTVarInfo) = vi.accs
setaccs!!(vi::VNTVarInfo, accs::AccumulatorTuple) = VNTVarInfo(vi.values, accs)

transformation(::VNTVarInfo) = DynamicTransformation()

Base.haskey(vi::VNTVarInfo, vn::VarName) = haskey(vi.values, vn)

Base.length(vi::VNTVarInfo) = length(vi.values)

function Base.getindex(vi::VNTVarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.transform(tv.val)
end

Base.isempty(vi::VNTVarInfo) = isempty(vi.values)

# TODO(mhauru) This should be called setindex_internal!!, but that's not the current
# convention.
function BangBang.setindex!!(vi::VNTVarInfo, val, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = TransformedValue(val, old_tv.linked, old_tv.transform)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VNTVarInfo(new_values, vi.accs)
end

# TODO(mhauru) The arguments are in the wrong order, but this is the current convetion.
function BangBang.push!!(vi::VNTVarInfo, vn::VarName, val, transform=typed_identity)
    new_tv = TransformedValue(val, false, transform)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VNTVarInfo(new_values, vi.accs)
end

Base.keys(vi::VNTVarInfo) = keys(vi.values)

function set_transformed!!(vi::VNTVarInfo, linked::Bool, vn::VarName)
    old_tv = getindex(vi.values, vn)
    new_tv = TransformedValue(old_tv.val, linked, old_tv.transform)
    new_values = setindex!!(vi.values, new_tv, vn)
    return VNTVarInfo(new_values, vi.accs)
end

function set_transformed!!(vi::VNTVarInfo, linked::Bool)
    new_values = map!!(vi.values) do tv
        TransformedValue(tv.val, linked, tv.transform)
    end
    return VNTVarInfo(new_values, vi.accs)
end

function getindex_internal(vi::VNTVarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.val
end

getindex_internal(vi::VNTVarInfo, ::Colon) = values_as(vi, Vector)

function is_transformed(vi::VNTVarInfo, vn::VarName)
    tv = getindex(vi.values, vn)
    return tv.linked
end

# TODO(mhauru) Other VarInfos have something like this. Do we need it?
# function from_internal_transform(::VNTVarInfo, ::VarName, dist::Distribution)
#     return from_vec_transform(dist)
# end

function from_internal_transform(vi::VNTVarInfo, vn::VarName, ::Distribution)
    return getindex(vi.values, vn).transform
end

function from_linked_internal_transform(::VNTVarInfo, ::VarName, dist::Distribution)
    return from_linked_vec_transform(dist)
end

function from_linked_internal_transform(vi::VNTVarInfo, vn::VarName)
    return getindex(vi.values, vn).transform
end

function change_transform(tv::TransformedValue, new_transform, linked)
    val_untransformed, logjac1 = with_logabsdet_jacobian(tv.transform, tv.val)
    val_new, logjac2 = with_logabsdet_jacobian(inverse(new_transform), val_untransformed)
    return TransformedValue(val_new, linked, new_transform), logjac1 + logjac2
end

function link!!(::DynamicTransformation, vi::VNTVarInfo, vns, model::Model)
    dists = extract_priors(model, vi)
    cumulative_logjac = zero(LogProbType)
    new_values = vi.values
    for vn in vns
        new_values = apply!!(new_values, vn) do tv
            dist = getindex(dists, vn)
            transform = from_linked_vec_transform(dist)
            new_tv, logjac = change_transform(tv, transform, true)
            cumulative_logjac += logjac
            return new_tv
        end
    end
    vi = VNTVarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function link!!(::DynamicTransformation, vi::VNTVarInfo, model::Model)
    # TODO(mhauru) This is probably pretty inefficient. Do this better. Would like to use
    # map!!, but it doesn't have access to the VarName.
    dists = extract_priors(model, vi)
    cumulative_logjac = zero(LogProbType)
    new_values = vi.values
    vns = keys(vi)
    for vn in vns
        new_values = apply!!(new_values, vn) do tv
            dist = getindex(dists, vn)
            transform = from_linked_vec_transform(dist)
            new_tv, logjac = change_transform(tv, transform, true)
            cumulative_logjac += logjac
            return new_tv
        end
    end
    vi = VNTVarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function invlink!!(::DynamicTransformation, vi::VNTVarInfo, vns, model::Model)
    cumulative_logjac = zero(LogProbType)
    new_values = vi.values
    for vn in vns
        new_values = apply!!(new_values, vn) do tv
            transform = typed_identity
            new_tv, logjac = change_transform(tv, transform, false)
            cumulative_logjac += logjac
            return new_tv
        end
    end
    vi = VNTVarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function invlink!!(::DynamicTransformation, vi::VNTVarInfo, model::Model)
    # TODO(mhauru) This is probably pretty inefficient. Do this better. Would like to use
    # map!!, but it doesn't have access to the VarName.
    cumulative_logjac = zero(LogProbType)
    new_values = vi.values
    vns = keys(vi)
    for vn in vns
        new_values = apply!!(new_values, vn) do tv
            transform = typed_identity
            new_tv, logjac = change_transform(tv, transform, false)
            cumulative_logjac += logjac
            return new_tv
        end
    end
    vi = VNTVarInfo(new_values, vi.accs)
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

# TODO(mhauru) I don't think this should return the internal values, but that's the current
# convention.
function values_as(vi::VNTVarInfo, ::Type{Vector})
    return mapreduce(tv -> tovec(tv.val), vcat, vi.values; init=Union{}[])
end

# TODO(mhauru) These two are now redundant, just conforming to the old interface
# temporarily.
function untyped_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return VNTVarInfo(rng, model, init_strategy)
end

function typed_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return VNTVarInfo(rng, model, init_strategy)
end

typed_varinfo(vi::VNTVarInfo) = vi

function typed_varinfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return typed_varinfo(Random.default_rng(), model, init_strategy)
end

function untyped_varinfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return untyped_varinfo(Random.default_rng(), model, init_strategy)
end

function unflatten(vi::VNTVarInfo, vec::AbstractVector)
    index = 1
    new_values = map!!(vi.values) do tv
        # TODO(mhauru) This is quite crude, assuming that the value stored currently is
        # an AbstractArray of some kind that has a size, and that reshape makes sense here.
        # I may fix this later, but I'm also tempted to just get rid of unflatten entirely.
        # This works for now for making most tests pass.
        old_val = tv.val
        len = length(old_val)
        new_val = reshape(vec[index:(index + len - 1)], size(old_val))
        # If the old_val was a scalar then new_val is a 0-dimensional array.
        # Convert it to a scalar.
        if !(old_val isa AbstractArray) && length(old_val) == 1
            new_val = new_val[1]
        end
        index += len
        return TransformedValue(new_val, tv.linked, tv.transform)
    end
    return VNTVarInfo(new_values, vi.accs)
end
