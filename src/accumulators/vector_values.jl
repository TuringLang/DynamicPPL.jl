const VECTORVAL_ACCNAME = :VectorValue

"""
    _get_vector_tval(val, tval, logjac, vn, dist)

Generate a `TransformedValue` that always has a vector as its stored value.
"""
function _get_vector_tval(
    val, tval::TransformedValue{V,T}, logjac, vn, dist
) where {V<:AbstractVector,T}
    # `tval.transform` could be DynamicLink(), Unlink(), or a fixed vectorising transform.
    return tval
end
function _get_vector_tval(val, tval::TransformedValue{V,T}, logjac, vn, dist) where {V,T}
    # If it's *not* an AbstractVector transformed value, then in principle, we need to
    # vectorise it before storing. We *could* do this by reversing the transformation, and
    # then applying a vectorisation transform; but the truth is that this is most likely to
    # be a user error where they tried to use a FixedTransform that does not vectorise. So
    # we just error here.
    return error(
        "Expected a vectorised or untransformed value for variable $vn, but got a TransformedValue with a value of $(tval.value).",
    )
end
function _get_vector_tval(
    val, ::TransformedValue{V,NoTransform}, logjac, vn, dist
) where {V}
    # This is an untransformed value, so we need to vectorise it. We can do this by applying
    # to_vec(dist).
    f = Bijectors.VectorBijectors.to_vec(dist)
    new_val, logjac = with_logabsdet_jacobian(f, val)
    @assert iszero(logjac) # otherwise we're in trouble...
    return TransformedValue(new_val, Unlink())
end
function _get_vector_tval(
    val, ::TransformedValue{V,NoTransform}, logjac, vn, dist
) where {V<:AbstractVector}
    # This is the same as above but just shortcircuited because `to_vec` should always
    # return TypedIdentity. Note that this method needs to be preserved to avoid method
    # ambiguities.
    return TransformedValue(val, Unlink())
end

# This is equivalent to `varinfo.values` where `varinfo isa VarInfo`
"""
    VectorValueAccumulator()

An accumulator that collects vectorised values, i.e. `TransformedValue{<:AbstractVector}`.

The exact type of the vectorised value (i.e., `tval.transform`) will depend on the transform
strategy that the model was evaluated with, and specifically, is equal to
`target_transform(transform_strategy, vn)`; *except* for the case where `target_transform`
is `Untransformed`, i.e., no transformation is to be applied. In this case, the
`VectorValueAccumulator` will apply a vectorisation transform to the untransformed value,
i.e., generate a `TransformedValue` with `Unlink()` as the transform.
"""
VectorValueAccumulator() = VNTAccumulator{VECTORVAL_ACCNAME}(_get_vector_tval)

"""
    internal_values_as_vector(vnt::VarNamedTuple)

Concatenate all the values in `vnt` into a single vector. This will error if any of the
values in `vnt` contain non-vector values.

```jldoctest
julia> using DynamicPPL

julia> vnt = @vnt begin
           x := TransformedValue([1.0, 2.0], Unlink())
           y := TransformedValue([3.0], DynamicLink())
       end
VarNamedTuple
├─ x => TransformedValue{Vector{Float64}, Unlink}([1.0, 2.0], Unlink())
└─ y => TransformedValue{Vector{Float64}, DynamicLink}([3.0], DynamicLink())

julia> internal_values_as_vector(vnt)
3-element Vector{Float64}:
 1.0
 2.0
 3.0
```

This is equivalent to `varinfo[:]` (for `varinfo::VarInfo`). However, instead of using a
`VarInfo` object, we strongly recommend that you use a `VectorValueAccumulator` and then
call `get_vector_values` on the accumulator.

```jldoctest
julia> using DynamicPPL, Distributions, LinearAlgebra

julia> vector_acc = VectorValueAccumulator();

julia> @model function f()
           x ~ MvNormal(zeros(2), I)
           y ~ Beta(2, 2)
       end;

julia> accs = OnlyAccsVarInfo(vector_acc);

julia> # note InitFromParams provides parameters in untransformed space
       _, accs = init!!(f(), accs, InitFromParams((x = [1.0, 2.0], y = 0.5)), LinkAll());

julia> # but because we specified LinkAll(), the vectorised values are transformed
       vector_vals = get_vector_values(accs)
VarNamedTuple
├─ x => TransformedValue{Vector{Float64}, DynamicLink}([1.0, 2.0], DynamicLink())
└─ y => TransformedValue{Vector{Float64}, DynamicLink}([0.0], DynamicLink())

julia> # we can extract the internal values as a single vector
       internal_values_as_vector(vector_vals)
3-element Vector{Float64}:
 1.0
 2.0
 0.0
```
"""
function internal_values_as_vector(vnt::VarNamedTuple)
    return mapfoldl(pair -> _as_vector(pair.second), vcat, vnt; init=Union{}[])
end
_as_vector(val::TransformedValue{T}) where {T<:AbstractVector} = val.value
function _as_vector(val::TransformedValue{T}) where {T}
    return error(
        "Expected a TransformedValue with a vector as its value, but got a TransformedValue with a value of $val.",
    )
end
