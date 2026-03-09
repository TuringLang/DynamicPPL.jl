@generated function skeleton(vnt::VarNamedTuple{names,types}) where {names,types}
    nms_to_include = Symbol[]
    for (n, t) in zip(names, types.parameters)
        if _element_needs_skeleton(t)
            push!(nms_to_include, n)
        end
    end
    return if isempty(nms_to_include)
        :(VarNamedTuple())
    else
        args = [:($n = skeleton(vnt.data.$n)) for n in nms_to_include]
        expr = Expr(:tuple, args...)
        :(VarNamedTuple($expr))
    end
end

function _element_needs_skeleton(::Type{VarNamedTuple{names,types}}) where {names,types}
    return any(_element_needs_skeleton, types.parameters)
end
_element_needs_skeleton(::Type{<:PartialArray}) = true
_element_needs_skeleton(::Type{T}) where {T} = VarNamedTuple <: T || PartialArray <: T

# For some reason this has to be @generated, because sometimes the compiler doesn't figure
# out _element_needs_skeleton
@generated function skeleton(pa::PartialArray{T}) where {T}
    if isconcretetype(T)
        # We can decide from the eltype whether we need to recurse
        if _element_needs_skeleton(T)
            # However, constructing the actual value requires recursing on the *values*
            # themselves rather than just the *type*, because the size of the array is not
            # determined in the type.
            # We need to be careful about this because some values in the PA may be unset.
            return quote
                idx = findfirst(pa.mask)
                idx === nothing && error("Unexpected PartialArray with no data")
                example_skeleton = skeleton(pa.data[idx])
                new_data = similar(pa.data, typeof(example_skeleton))
                fill!(new_data, example_skeleton)
                return new_data
            end
        else
            # Easy path, just fill with nothings.
            return :(similar(pa.data, Nothing))
        end
    else
        # Need to map over each element individually, which is quite ugly.
        return quote
            new_data = similar(pa.data)
            for i in eachindex(pa.data)
                # Need to use setindex!! here because the original eltype of pa.data might
                # not be broad enough to hold the skeletons themselves (which may have
                # different concrete types from pa.data[i]).
                if pa.mask[i]
                    di = pa.data[i]
                    if _element_needs_skeleton(typeof(di))
                        BangBang.setindex!!(new_data, skeleton(di), i)
                    else
                        BangBang.setindex!!(new_data, nothing, i)
                    end
                else
                    BangBang.setindex!!(new_data, nothing, i)
                end
            end
            return new_data
        end
    end
end
