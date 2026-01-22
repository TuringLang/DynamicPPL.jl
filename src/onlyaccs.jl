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
DynamicPPL.maybe_invlink_before_eval!!(vi::OnlyAccsVarInfo, ::Model) = vi
DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) = OnlyAccsVarInfo(accs)

# Ideally, we'd define this together with InitContext, but alas that file comes way before
# this one, and sorting out the include order is a pain.
function tilde_assume!!(
    ctx::InitContext,
    dist::Distribution,
    vn::VarName,
    template::Any,
    vi::Union{OnlyAccsVarInfo,ThreadSafeVarInfo{<:OnlyAccsVarInfo}},
)
    # For OnlyAccsVarInfo, since we don't need to write into the VarInfo, we can 
    # cut out a lot of the code above.
    tval = init(ctx.rng, vn, dist, ctx.strategy)
    x, inv_logjac = with_logabsdet_jacobian(
        DynamicPPL.get_transform(tval), DynamicPPL.get_internal_value(tval)
    )
    vi = accumulate_assume!!(vi, x, tval, -inv_logjac, vn, dist, template)
    return x, vi
end
