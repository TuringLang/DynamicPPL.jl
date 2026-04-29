module DynamicPPLComponentArraysExt
using DynamicPPL: DynamicPPL
using DynamicPPL.VarNamedTuples:
    PartialArray,
    AllowAll,
    SetPermissions,
    _setindex_optic!!,
    _getindex_optic,
    make_leaf,
    make_leaf_singleindex,
    _is_multiindex,
    make_leaf_multiindex
using ComponentArrays: ComponentArrays, ComponentArray, ComponentVector
using AbstractPPL

function DynamicPPL.VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Property{S}, template::ComponentVector
) where {S}
    if optic.child isa AbstractPPL.Iden
        ax = ComponentArrays.getaxes(template)[1]
        idx = first(ax[S].idx)
        index_optic = AbstractPPL.Index((idx,), NamedTuple(), optic.child)
        return make_leaf(value, index_optic, template)
    else
        return invoke(
            make_leaf,
            Tuple{Any, AbstractPPL.Property{S}, AbstractArray},
            value, optic, template,
        )
    end
end

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