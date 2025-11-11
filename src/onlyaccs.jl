"""
    OnlyAccsVarInfo

This is a wrapper around an `AccumulatorTuple` that implements the minimal `AbstractVarInfo`
interface to work with the `tilde_assume!!` and `tilde_observe!!` functions for
`InitContext`.

Note that this does not implement almost every other AbstractVarInfo interface function, and
so using attempting to use this with a different leaf context such as `DefaultContext` will
result in errors.

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
@inline DynamicPPL.maybe_invlink_before_eval!!(vi::OnlyAccsVarInfo, ::Model) = vi
@inline DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
@inline DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) =
    OnlyAccsVarInfo(accs)
@inline Base.haskey(::OnlyAccsVarInfo, ::VarName) = false
@inline DynamicPPL.is_transformed(::OnlyAccsVarInfo) = false
@inline BangBang.push!!(vi::OnlyAccsVarInfo, vn, y, dist) = vi
