struct MixedVarInfo{
    Ttvi <: Union{TypedVarInfo, Nothing},
    Tuvi <: UntypedVarInfo,
} <: AbstractVarInfo
    tvi::Ttvi
    uvi::Tuvi
    is_uvi_empty::Base.RefValue{Bool}
end
MixedVarInfo(vi::TypedVarInfo) = MixedVarInfo(vi, VarInfo(), Ref(true))
function MixedVarInfo(vi::UntypedVarInfo)
    MixedVarInfo(TypedVarInfo(vi), empty!(deepcopy(vi)), Ref(true))
end
function VarInfo(model::Model, ctx = DefaultContext())
    vi = VarInfo()
    model(vi, SampleFromPrior(), ctx)
    return MixedVarInfo(TypedVarInfo(vi))
end
function VarInfo(model::Model, n::Integer, ctx = DefaultContext())
    if n == 0
        vi = VarInfo()
        model(vi)
        return vi
    else
        tvi = TypedVarInfo(model, n, ctx)
        return MixedVarInfo(tvi)
    end
end
function VarInfo(old_vi::MixedVarInfo, spl, x::AbstractVector)
    new_tvi = VarInfo(old_vi.tvi, spl, x)
    return MixedVarInfo(new_tvi, old_vi.uvi, old_vi.is_uvi_empty)
end
function TypedVarInfo(vi::MixedVarInfo)
    @assert getmode(vi.tvi) === getmode(vi.uvi)
    mode = getmode(vi.tvi)
    fixed_support = has_fixed_support(vi.tvi) && has_fixed_support(vi.uvi)
    synced = issynced(vi.tvi) && issynced(vi.uvi)
    if vi.is_uvi_empty[]
        return vi.tvi
    else
        return VarInfo(
            merge(vi.tvi.metadata, TypedVarInfo(vi.uvi).metadata),
            Ref(getlogp(vi.tvi)),
            Ref(get_num_produce(vi.tvi)),
            mode,
            Ref(fixed_support),
            Ref(synced),
        )
    end
end

function Base.merge(t1::MixedVarInfo, t2::MixedVarInfo)
    return MixedVarInfo(merge(TypedVarInfo(t1), TypedVarInfo(t2)), VarInfo(), Ref(true))
end
function Base.merge(t1::TypedVarInfo, t2::MixedVarInfo)
    return MixedVarInfo(merge(t1, TypedVarInfo(t2)), VarInfo(), Ref(true))
end
Base.merge(t1::MixedVarInfo, t2::TypedVarInfo) = merge(t2, t1)
function Base.merge(t1::UntypedVarInfo, t2::MixedVarInfo)
    return MixedVarInfo(merge(TypedVarInfo(t1), TypedVarInfo(t2)), VarInfo(), Ref(true))
end
Base.merge(t1::MixedVarInfo, t2::UntypedVarInfo) = merge(t2, t1)

function getvns(vi::MixedVarInfo, s::Selector, ::Val{space}) where {space}
    if space !== () && all(haskey.(Ref(vi.tvi.metadata), space))
        return getvns(vi.tvi, s, Val(space))
    else
        return getvns(TypedVarInfo(vi), s, Val(space))
    end
end
getmode(vi::MixedVarInfo) = getmode(vi.tvi)

function getmetadata(vi::MixedVarInfo, vn::VarName)
    if haskey(vi.tvi, vn)
        return getmetadata(vi.tvi, vn)
    else
        return getmetadata(vi.uvi, vn)
    end
end
function Base.show(io::IO, vi::MixedVarInfo)
    print(io, "Instance of MixedVarInfo")
end

function fullyinspace(spl::AbstractSampler, vi::TypedVarInfo)
    space = getspace(spl)
    return space !== () && all(haskey.(Ref(vi.metadata), space))
end

acclogp!(vi::MixedVarInfo, logp) = acclogp!(vi.tvi, logp)
getlogp(vi::MixedVarInfo) = getlogp(vi.tvi)
resetlogp!(vi::MixedVarInfo) = resetlogp!(vi.tvi)
setlogp!(vi::MixedVarInfo, logp) = setlogp!(vi.tvi, logp)

get_num_produce(vi::MixedVarInfo) = get_num_produce(vi.tvi)
increment_num_produce!(vi::MixedVarInfo) = increment_num_produce!(vi.tvi)
reset_num_produce!(vi::MixedVarInfo) = reset_num_produce!(vi.tvi)
set_num_produce!(vi::MixedVarInfo, n::Int) = set_num_produce!(vi.tvi, n)

syms(vi::MixedVarInfo) = (syms(vi.tvi)..., syms(vi.uvi)...)

function setgid!(vi::MixedVarInfo, gid::Selector, vn::VarName; overwrite=false)
    hassymbol(vi.tvi, vn) ? setgid!(vi.tvi, gid, vn; overwrite=overwrite) : setgid!(vi.uvi, gid, vn; overwrite=overwrite)
    return vi
end
function setorder!(vi::MixedVarInfo, vn::VarName, index::Int)
    hassymbol(vi.tvi, vn) ? setorder!(vi.tvi, vn, index) : setorder!(vi.uvi, vn, index)
    return vi
end
function setval!(vi::MixedVarInfo, val, vn::VarName)
    hassymbol(vi.tvi, vn) ? setval!(vi.tvi, val, vn) : setval!(vi.uvi, val, vn)
    return vi
end

function haskey(vi::MixedVarInfo, vn::VarName)
    return hassymbol(vi.tvi, vn) ? haskey(vi.tvi, vn) : haskey(vi.uvi, vn)
end

Bijectors.link(vi::MixedVarInfo) = MixedVarInfo(link(vi.tvi), link(vi.uvi), vi.is_uvi_empty)
Bijectors.invlink(vi::MixedVarInfo) = MixedVarInfo(invlink(vi.tvi), invlink(vi.uvi), vi.is_uvi_empty)
initlink(vi::MixedVarInfo) = MixedVarInfo(initlink(vi.tvi), initlink(vi.uvi), vi.is_uvi_empty)
has_fixed_support(vi::MixedVarInfo) = has_fixed_support(vi.tvi) && has_fixed_support(vi.uvi)
function set_fixed_support!(vi::MixedVarInfo, b::Bool)
    set_fixed_support!(vi.tvi, b)
    return vi
end

issynced(vi::MixedVarInfo) = issynced(vi.tvi) && issynced(vi.uvi)
function setsynced!(vi::MixedVarInfo, b::Bool)
    setsynced!(vi.tvi, b)
    setsynced!(vi.uvi, b)
    return vi
end

function removedel!(vi::MixedVarInfo)
    if vi.is_uvi_empty[]
        return MixedVarInfo(removedel!(vi.tvi), vi.uvi, vi.is_uvi_empty)
    else
        removedel!(vi.uvi)
        if isempty(vi.uvi)
            return MixedVarInfo(removedel!(vi.tvi), vi.uvi, Ref(true))
        else
            return MixedVarInfo(removedel!(vi.tvi), vi.uvi, Ref(false))
        end
    end
end

function link!(vi::MixedVarInfo, spl::AbstractSampler, model)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        link!(vi.tvi, spl, model)
    else
        link!(vi.tvi, spl, model)
        link!(vi.uvi, spl, model)
    end
    return vi
end
function invlink!(vi::MixedVarInfo, spl::AbstractSampler, model)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        invlink!(vi.tvi, spl, model)
    else
        invlink!(vi.tvi, spl, model)
        invlink!(vi.uvi, spl, model)
    end
    return vi
end
function islinked(vi::MixedVarInfo, spl::AbstractSampler)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        return islinked(vi.tvi, spl)
    else
        return islinked(vi.tvi, spl) || islinked(vi.uvi, spl)
    end
end

function getindex(vi::MixedVarInfo, vn::VarName)
    return hassymbol(vi.tvi, vn) ? getindex(vi.tvi, vn) : getindex(vi.uvi, vn)
end
# All the VarNames have the same symbol
function getindex(vi::MixedVarInfo, vns::Vector{<:VarName{s}}) where {s}
    return hassymbol(vi.tvi, vns[1]) ? getindex(vi.tvi, vns) : getindex(vi.uvi, vns)
end

for splT in (:SampleFromPrior, :SampleFromUniform, :AbstractSampler)
    @eval begin
        function getindex(vi::MixedVarInfo, spl::$splT)
            if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
                return vi.tvi[spl]
            else
                return vcat(vi.tvi[spl], copy.(vi.uvi[spl]))
            end
        end

        function setindex!(vi::MixedVarInfo, val, spl::$splT)
            if fullyinspace(spl, vi.tvi)
                setindex!(vi.tvi, val, spl)
            else
                # TODO: define length(vi::TypedVarInfo, spl)
                n = length(vi.tvi[spl])
                setindex!(vi.tvi, val[1:n], spl)
                if n < length(val)
                    setindex!(vi.uvi, val[n+1:end], spl)
                end
            end
            return vi
        end
    end
end

function getall(vi::MixedVarInfo)
    if vi.is_uvi_empty[]
        return getall(vi.tvi)
    else
        return vcat(getall(vi.tvi), copy.(getall(vi.uvi)))
    end
end

function set_retained_vns_del_by_spl!(vi::MixedVarInfo, spl::Sampler)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        set_retained_vns_del_by_spl!(vi.tvi, spl)
    else
        set_retained_vns_del_by_spl!(vi.tvi, spl)
        set_retained_vns_del_by_spl!(vi.uvi, spl)
    end
    return vi
end

isempty(vi::MixedVarInfo) = isempty(vi.tvi) && vi.is_uvi_empty[]
function empty!(vi::MixedVarInfo)
    empty!(vi.tvi)
    vi.is_uvi_empty[] || empty!(vi.uvi)
    vi.is_uvi_empty[] = true
    return vi
end

function push!(
    vi::MixedVarInfo,
    vn::VarName,
    r,
    dist::Distribution,
    gidset::Set{Selector}
)
    if hassymbol(vi.tvi, vn)
        push!(vi.tvi, vn, r, dist, gidset)
    else
        push!(vi.uvi, vn, r, dist, gidset)
        vi.is_uvi_empty[] = false
    end
    return vi
end

function unset_flag!(vi::MixedVarInfo, vn::VarName, flag::String)
    hassymbol(vi.tvi, vn) ? unset_flag!(vi.tvi, vn, flag) : unset_flag!(vi.uvi, vn, flag)
    return vi
end
function is_flagged(vi::MixedVarInfo, vn::VarName, flag::String)
    if hassymbol(vi.tvi, vn)
        return is_flagged(vi.tvi, vn, flag)
    else
        return is_flagged(vi.uvi, vn, flag)
    end
end

function updategid!(vi::MixedVarInfo, spl::AbstractSampler; overwrite=false)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        updategid!(vi.tvi, spl; overwrite=overwrite)
    else
        updategid!(vi.uvi, spl; overwrite=overwrite)
    end
    return vi
end

function tonamedtuple(vi::MixedVarInfo)
    if vi.is_uvi_empty[]
        return tonamedtuple(vi.tvi)
    else
        return tonamedtuple(TypedVarInfo(vi))
    end
end
set_namedtuple!(vi::MixedVarInfo, nt::NamedTuple) = set_namedtuple!(vi.tvi, nt)
