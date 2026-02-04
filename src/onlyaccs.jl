"""
    OnlyAccsVarInfo

This is a wrapper around an `AccumulatorTuple` that implements the minimal `AbstractVarInfo`
interface to work with the `tilde_assume!!` and `tilde_observe!!` functions for
`InitContext`.

Note that this does not implement almost every other AbstractVarInfo interface function, and
so using this with a different leaf context such as `DefaultContext` will result in errors.

Conceptually, one can also think of this as a VarInfo that doesn't contain a metadata field.
This is also why it only works with `InitContext`: in this case, the parameters used for
evaluation are supplied by the context instead of the metadata.
"""
struct OnlyAccsVarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
OnlyAccsVarInfo() = OnlyAccsVarInfo(default_accumulators())
function OnlyAccsVarInfo(accs::NTuple{N,AbstractAccumulator}) where {N}
    return OnlyAccsVarInfo(AccumulatorTuple(accs))
end

# Minimal AbstractVarInfo interface
DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) = OnlyAccsVarInfo(accs)
function DynamicPPL.get_transform_strategy(::OnlyAccsVarInfo)
    # OAVI doesn't contain this info, we can't return a sensible value. Hopefully this
    # method doesn't ever get called though.
    return error(
        "get_transform_strategy cannot be implemented for OnlyAccsVarInfo; please specify a transform strategy manually in your call to `init!!`",
    )
end

# This allows us to make use of the main tilde_assume!!(::InitContext) method without
# having to duplicate the code here
@inline function DynamicPPL.setindex_with_dist!!(
    vi::OnlyAccsVarInfo, ::AbstractTransformedValue, ::Distribution, ::VarName, ::Any
)
    return vi
end
