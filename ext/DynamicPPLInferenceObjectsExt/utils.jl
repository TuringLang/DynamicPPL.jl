# adapted from MCMCChains
function isconcretetype_recursive(T)
    return isconcretetype(T) && (eltype(T) === T || isconcretetype_recursive(eltype(T)))
end

concretize(x) = x
function concretize(x::AbstractArray)
    if isconcretetype_recursive(typeof(x))
        return x
    else
        xnew = map(concretize, x)
        T = mapreduce(typeof, promote_type, xnew; init=Union{})
        if T <: eltype(xnew) && T !== Union{}
            return convert(AbstractArray{T}, xnew)
        else
            return xnew
        end
    end
end

dims2coords(dims) = NamedTuple{Dimensions.dim2key(dims)}(Dimensions.lookup(dims))
