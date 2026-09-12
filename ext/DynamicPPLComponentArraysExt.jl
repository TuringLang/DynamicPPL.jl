module DynamicPPLComponentArraysExt
using DynamicPPL: DynamicPPL
using DynamicPPL.VarNamedTuples:
    PartialArray,
    AllowAll,
    SetPermissions,
    _setindex_optic!!,
    _getindex_optic,
    _haskey_optic,
    make_leaf
using ComponentArrays: ComponentArrays, ComponentVector
using AbstractPPL

# Resolve properties through the component axes, including nested fields and slices.
function _property_to_index(
    template::ComponentVector, optic::AbstractPPL.Property{S}
) where {S}
    indices = ComponentVector(
        LinearIndices(ComponentArrays.getdata(template)), ComponentArrays.getaxes(template)
    )
    return AbstractPPL.Index(
        (_resolve_indices(optic, indices),), NamedTuple(), AbstractPPL.Iden()
    )
end

# Resolve child indices against unwrapped storage to avoid ComponentArrays'
# range-indexing ambiguity.
_resolve_indices(::AbstractPPL.Iden, indices) = indices
function _resolve_indices(optic::AbstractPPL.Property{S}, indices) where {S}
    return _resolve_indices(optic.child, getproperty(indices, S))
end
function _resolve_indices(optic::AbstractPPL.AbstractOptic, indices)
    return optic(ComponentArrays.getdata(indices))
end

function DynamicPPL.VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Property{S}, template::ComponentVector
) where {S}
    return make_leaf(value, _property_to_index(template, optic), template)
end

function DynamicPPL.VarNamedTuples._setindex_optic!!(
    pa::PartialArray{<:Any,<:Any,<:ComponentVector},
    value,
    optic::AbstractPPL.Property{S},
    template,
    permissions::SetPermissions=AllowAll(),
) where {S}
    index_optic = _property_to_index(pa.data, optic)
    return _setindex_optic!!(pa, value, index_optic, template, permissions)
end

function DynamicPPL.VarNamedTuples._getindex_optic(
    pa::PartialArray{<:Any,<:Any,<:ComponentVector}, optic::AbstractPPL.Property{S}, orig_vn
) where {S}
    index_optic = _property_to_index(pa.data, optic)
    return _getindex_optic(pa, index_optic, orig_vn)
end

function DynamicPPL.VarNamedTuples._haskey_optic(
    pa::PartialArray{<:Any,<:Any,<:ComponentVector}, optic::AbstractPPL.Property{S}
) where {S}
    indices = ComponentVector(
        LinearIndices(ComponentArrays.getdata(pa.data)), ComponentArrays.getaxes(pa.data)
    )
    AbstractPPL.canview(optic, indices) || return false
    return _haskey_optic(pa, _property_to_index(pa.data, optic))
end

end
