module DynamicPPLComponentArraysExt
using DynamicPPL: DynamicPPL
using DynamicPPL.VarNamedTuples:
    PartialArray, AllowAll, SetPermissions, _setindex_optic!!, _getindex_optic
using ComponentArrays: ComponentArrays, ComponentArray, ComponentVector
using AbstractPPL

function DynamicPPL.VarNamedTuples._setindex_optic!!(
    pa::PartialArray{<:Any,<:Any,<:ComponentVector},
    value,
    optic::AbstractPPL.Property{S},
    template,
    permissions::SetPermissions=AllowAll(),
) where {S}
    ax = ComponentArrays.getaxes(pa.data)[1]
    idx = first(ax[S].idx)
    index_optic = AbstractPPL.Index((idx,), NamedTuple(), optic.child)
    return _setindex_optic!!(pa, value, index_optic, template, permissions)
end

function DynamicPPL.VarNamedTuples._getindex_optic(
    pa::PartialArray{<:Any,<:Any,<:ComponentVector},
    optic::AbstractPPL.Property{S},
) where {S}
    ax = ComponentArrays.getaxes(pa.data)[1]
    idx = first(ax[S].idx)
    index_optic = AbstractPPL.Index((idx,), NamedTuple(), optic.child)
    return _getindex_optic(pa, index_optic)
end

end