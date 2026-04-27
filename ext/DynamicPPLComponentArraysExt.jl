module DynamicPPLComponentArraysExt

using DynamicPPL: DynamicPPL
using DynamicPPL.VarNamedTuples:
    PartialArray,
    AllowAll,
    SetPermissions,
    make_leaf_singleindex,
    make_leaf_multiindex,
    _is_multiindex,
    _setindex_optic!!
using ComponentArrays: ComponentArray, getaxes
using AbstractPPL

function DynamicPPL.VarNamedTuples.make_leaf(
    value, optic::AbstractPPL.Index, template::ComponentArray
)
    coptic = AbstractPPL.concretize_top_level(optic, template)
    return if _is_multiindex(template, coptic.ix...; coptic.kw...)
        make_leaf_multiindex(value, coptic, template)
    else
        make_leaf_singleindex(value, coptic, template)
    end
end

function DynamicPPL.VarNamedTuples._setindex_optic!!(
    pa::PartialArray{<:Any,<:Any,<:ComponentArray},
    value,
    optic::AbstractPPL.Property{S},
    template,
    permissions::SetPermissions=AllowAll(),
) where {S}
    ax = getaxes(pa.data)[1]
    idx = ax[S].idx
    index_optic = AbstractPPL.Index((idx,), NamedTuple(), optic.child)
    return _setindex_optic!!(pa, value, index_optic, template, permissions)
end

end
