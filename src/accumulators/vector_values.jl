const VECTORVAL_ACCNAME = :VectorValue
_get_vector_tval(val, tval::Union{VectorValue,LinkedVectorValue}, logjac, vn, dist) = tval
function _get_vector_tval(val, ::UntransformedValue, logjac, vn, dist)
    original_val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    f = to_vec_transform(dist)
    new_val, logjac = with_logabsdet_jacobian(f, val)
    @assert iszero(logjac) # otherwise we're in trouble...
    return VectorValue(new_val, inverse(f), original_val_size)
end

# This is equivalent to `varinfo.values` where `varinfo isa VarInfo`
VectorValueAccumulator() = VNTAccumulator{VECTORVAL_ACCNAME}(_get_vector_tval)

# This is equivalent to `varinfo[:]`
_as_vector(val::VectorValue) = DynamicPPL.get_internal_value(val)
_as_vector(val::LinkedVectorValue) = DynamicPPL.get_internal_value(val)
_as_vector(val) = error("don't know how to convert $(typeof(val)) to a vector value")

"""
    get_vector_values(vnt::VarNamedTuple)

Concatenate all the `VectorValue`s and `LinkedVectorValue`s in `vnt` into a single vector.
This will error if any of the values in `vnt` are not `VectorValue`s or
`LinkedVectorValue`s.

```jldoctest
julia> using DynamicPPL

julia> # In a real setting the other fields would be filled in with meaningful values.
       vnt = @vnt begin
           x := VectorValue([1.0, 2.0], nothing, nothing)
           y := LinkedVectorValue([3.0], nothing, nothing)
       end
VarNamedTuple
├─ x => VectorValue{Vector{Float64}, Nothing, Nothing}([1.0, 2.0], nothing, nothing)
└─ y => LinkedVectorValue{Vector{Float64}, Nothing, Nothing}([3.0], nothing, nothing)

julia> get_vector_values(vnt)
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
       get_vector_values(getacc(accs, Val(accumulator_name(vector_acc))))
3-element Vector{Float64}:
 1.0
 2.0
 0.0
```
"""
function get_vector_values(vnt::VarNamedTuple)
    return mapfoldl(pair -> _as_vector(pair.second), vcat, vnt; init=Union{}[])
end
get_vector_values(acc::VNTAccumulator{VECTORVAL_ACCNAME}) = get_vector_values(acc.values)
