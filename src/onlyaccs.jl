"""
    OnlyAccsVarInfo

This is a wrapper around an `AccumulatorTuple` that implements the minimal `AbstractVarInfo`
interface to work with the `accumulate_assume!!` and `accumulate_observe!!` functions.

Note that this does not implement almost every other AbstractVarInfo interface function, and
so using this outside of FastLDF will lead to errors.

Conceptually, one can also think of this as a VarInfo that doesn't contain a metadata field.
That is because values for random variables are obtained by reading from a separate entity
(such as a `FastLDFContext`), rather than from the VarInfo itself.
"""
struct OnlyAccsVarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
OnlyAccsVarInfo() = OnlyAccsVarInfo(default_accumulators())
function OnlyAccsVarInfo(accs::NTuple{N,AbstractAccumulator}) where {N}
    return OnlyAccsVarInfo(AccumulatorTuple(accs))
end

# AbstractVarInfo interface
@inline DynamicPPL.maybe_invlink_before_eval!!(vi::OnlyAccsVarInfo, ::Model) = vi
@inline DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
@inline DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) =
    OnlyAccsVarInfo(accs)
@inline Base.haskey(::OnlyAccsVarInfo, ::VarName) = false
@inline DynamicPPL.is_transformed(::OnlyAccsVarInfo) = false
@inline BangBang.push!!(vi::OnlyAccsVarInfo, vn, y, dist) = vi
