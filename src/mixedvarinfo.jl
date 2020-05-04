struct MixedVarInfo{
    Ttvi <: Union{TypedVarInfo, Nothing},
    Tuvi <: UntypedVarInfo,
} <: AbstractVarInfo
    tvi::Ttvi
    uvi::Tuvi
    is_uvi_empty::Base.RefValue{Bool}
end
MixedVarInfo(vi::MixedVarInfo) = vi
MixedVarInfo(vi::TypedVarInfo) = MixedVarInfo(vi, VarInfo(), Ref(true))
function MixedVarInfo(vi::UntypedVarInfo)
    MixedVarInfo(TypedVarInfo(vi), empty!(deepcopy(vi)), Ref(true))
end
function VarInfo(old_vi::MixedVarInfo, spl, x::AbstractVector)
    return MixedVarInfo(VarInfo(old_vi.tvi, spl, x), old_vi.uvi, old_vi.is_uvi_empty)
end

function Base.show(io::IO, vi::MixedVarInfo)
    print(io, "Instance of MixedVarInfo")
end

function fullyinspace(spl::AbstractSampler, vi::TypedVarInfo)
    space = getspace(spl)
    return space !== () && all(haskey.(Ref(vi.metadata), space))
end

acclogp!(vi::MixedVarInfo, logp) = acclogp!(vi.tvi)
getlogp(vi::MixedVarInfo) = getlogp(vi.tvi)
resetlogp!(vi::MixedVarInfo) = resetlogp!(vi.tvi)
setlogp!(vi::MixedVarInfo, logp) = setlogp!(vi.tvi, logp)

get_num_produce(vi::MixedVarInfo) = get_num_produce(vi.tvi)
increment_num_produce!(vi::MixedVarInfo) = increment_num_produce!(vi.tvi)
reset_num_produce!(vi::MixedVarInfo) = reset_num_produce!(vi.tvi)
set_num_produce!(vi::MixedVarInfo, n::Int) = set_num_produce!(vi.tvi, n)

syms(vi::MixedVarInfo) = (syms(vi.tvi)..., syms(vi.uvi)...)

function setgid!(vi::MixedVarInfo, gid::Selector, vn::VarName)
    hassymbol(vi.tvi, vn) ? setgid!(vi.tvi, gid, vn) : setgid!(vi.uvi, gid, vn)
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

function link!(vi::MixedVarInfo, spl::AbstractSampler)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        link!(vi.tvi, spl)
    else
        link!(vi.tvi, spl)
        link!(vi.uvi, spl)
    end
    return vi
end
function invlink!(vi::MixedVarInfo, spl::AbstractSampler)
    if fullyinspace(spl, vi.tvi) || vi.is_uvi_empty[]
        invlink!(vi.tvi, spl)
    else
        invlink!(vi.tvi, spl)
        invlink!(vi.uvi, spl)
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
                return vcat(vi.tvi[spl], vi.uvi[spl])
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

function tonamedtuple(vi::MixedVarInfo)
    t1 = tonamedtuple(vi.tvi)
    return vi.is_uvi_empty[] ? t1 : merge(t1, tonamedtuple(vi.uvi))
end