const VECTORVAL_ACCNAME = :VectorValue
_get_vector_tval(val, tval::Union{VectorValue,LinkedVectorValue}, logjac, vn, dist) = tval
function _get_vector_tval(val, ::UntransformedValue, logjac, vn, dist)
    original_val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    f = to_vec_transform(dist)
    new_val, logjac = with_logabsdet_jacobian(f, val)
    @assert iszero(logjac) # otherwise we're in trouble...
    return VectorValue(new_val, inverse(f), original_val_size)
end

VectorValueAccumulator() = VNTAccumulator{VECTORVAL_ACCNAME}(_get_vector_tval)
