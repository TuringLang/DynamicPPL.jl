module DynamicPPLComponentArraysExt

using DynamicPPL: DynamicPPL
using DynamicPPL.VarNamedTuples: PartialArray, AllowAll, SetPermissions, _setindex_optic!!
using ComponentArrays: ComponentArrays, ComponentVector
using AbstractPPL

function DynamicPPL.VarNamedTuples._setindex_optic!!(
    pa::PartialArray{<:Any,<:Any,<:ComponentVector},
    value,
    optic::AbstractPPL.Property{S},
    template,
    permissions::SetPermissions=AllowAll(),
) where {S}
    idx = ComponentArrays.label2index(pa.data, S)
    index_optic = AbstractPPL.Index((idx,), NamedTuple(), optic.child)
    return _setindex_optic!!(pa, value, index_optic, template, permissions)
end

end
