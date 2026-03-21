const LINKEDVECTRANSFORM_ACCNAME = :LinkedVecTransformAccumulator
function _get_linked_vec_transform(val, tv, logjac, vn, dist)
    return FixedTransform(Bijectors.VectorBijectors.from_linked_vec(dist))
end

"""
    LinkedVecTransformAccumulator()

An accumulator that stores the transform required to convert a linked vector into the
original, untransformed value.
"""
LinkedVecTransformAccumulator() =
    VNTAccumulator{LINKEDVECTRANSFORM_ACCNAME}(_get_linked_vec_transform)

"""
    get_linked_vec_transforms(vi::DynamicPPL.AbstractVarInfo)

Extract the transforms stored in the `LinkedVecTransformAccumulator` of an AbstractVarInfo.
Errors if the AbstractVarInfo does not have a `LinkedVecTransformAccumulator`.
"""
function get_linked_vec_transforms(vi::DynamicPPL.AbstractVarInfo)
    return DynamicPPL.getacc(vi, Val(LINKEDVECTRANSFORM_ACCNAME)).values
end

function get_linked_vec_transforms(rng::Random.AbstractRNG, model::DynamicPPL.Model)
    accs = OnlyAccsVarInfo(LinkedVecTransformAccumulator())
    _, accs = init!!(rng, model, accs, InitFromPrior(), UnlinkAll())
    return get_linked_vec_transforms(accs)
end
function get_linked_vec_transforms(model::DynamicPPL.Model)
    return get_linked_vec_transforms(Random.default_rng(), model)
end
