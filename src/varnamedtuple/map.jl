Base.merge(x1::VarNamedTuple, x2::VarNamedTuple) = _merge_recursive(x1, x2)

# This needs to be a generated function for type stability.
@generated function _merge_recursive(
    vnt1::VarNamedTuple{names1}, vnt2::VarNamedTuple{names2}
) where {names1,names2}
    all_names = union(names1, names2)
    exs = Expr[]
    push!(exs, :(data = (;)))
    for name in all_names
        val_expr = if name in names1 && name in names2
            :(_merge_recursive(vnt1.data.$name, vnt2.data.$name))
        elseif name in names1
            :(vnt1.data.$name)
        else
            :(vnt2.data.$name)
        end
        push!(exs, :(data = merge(data, NamedTuple{($(QuoteNode(name)),)}(($val_expr,)))))
    end
    push!(exs, :(return VarNamedTuple(data)))
    return Expr(:block, exs...)
end

"""
    subset(vnt::VarNamedTuple, vns)

Create a new `VarNamedTuple` containing only the variables subsumed by ones in `vns`.
"""
function DynamicPPL.subset(parent_vnt::VarNamedTuple, vns)
    # TODO(mhauru) This could be done more efficiently by generating the code directly,
    # because we could short-circuit: For instance, if `vns` contains `a`, we could
    # directly include the whole subtree under `a`, without checking each individual
    # variable under it.
    return mapfoldl(
        identity,
        function (acc_vnt, pair)
            name, value = pair
            return if any(vn -> subsumes(vn, name), vns)
                templated_setindex!!(
                    acc_vnt, value, name, parent_vnt.data[AbstractPPL.getsym(name)]
                )
            else
                acc_vnt
            end
        end,
        parent_vnt;
        init=VarNamedTuple(),
    )
end

"""
    apply!!(func, vnt::VarNamedTuple, name::VarName)

Apply `func` to the subdata at `name` in `vnt`, and set the result back at `name`.

Like `map_values!!`, but only for a single `VarName`.

```jldoctest
julia> using DynamicPPL: VarNamedTuple, setindex!!

julia> using DynamicPPL.VarNamedTuples: apply!!

julia> vnt = VarNamedTuple()
VarNamedTuple()

julia> vnt = setindex!!(vnt, [1, 2, 3], @varname(a))
VarNamedTuple(a = [1, 2, 3],)

julia> apply!!(x -> x .+ 1, vnt, @varname(a))
VarNamedTuple(a = [2, 3, 4],)
```
"""
function apply!!(func, vnt::VarNamedTuple, name::VarName)
    if !haskey(vnt, name)
        throw(KeyError(repr(name)))
    end
    subdata = _getindex_optic(vnt, name)
    new_subdata = func(subdata)
    # The allow_new=Val(true) is a performance optimisation: Since we've already checked
    # that the key exists, we know that no new fields will be created.
    return _setindex_optic!!(
        vnt,
        new_subdata,
        AbstractPPL.varname_to_optic(name),
        NoTemplate(),
        false;
        allow_new=Val(false),
    )
end

"""
    _map_recursive!!(func, x, vn)

Call `func` on `vn => x`, except if `x` is a `VarNamedTuple` or `PartialArray`, in which
case call `_map_recursive!!` recursively on all their elements, updating `vn` with the right
prefix.

This is the internal implementation of `map_pairs!!`, but because it has a method defined
for literally every type in existence, we hide it behind the interface of the more
discriminating `map_pairs!!`. It makes the implementation a bit simpler, compared to
checking element types within `map_pairs!!` itself.
"""
_map_recursive!!(func, x, vn) = func(vn => x)

# TODO(mhauru) The below is type unstable for some complex VarNames. My example case
# for which type stability fails is @varname(e.f[3].g.h[2].i). I don't understand this
# well, but I think it's just because constant propagation gives up at some point, and fails
# to go through the lines that figure out `new_et`. I could be wrong. I tried fixing this by
# lifting the first three lines of the function into a generated function, but that seems
# to run into trouble when trying to call Core.Compiler.return_type recursively on the same
# function. An earlier implementation of this function that only operated on the values,
# not on pairs of key => value, was type stable (presumably because it was a bit easier on
# constant propagation).
function _map_recursive!!(func, pa::PartialArray, vn)
    # Ask the compiler to infer the return type of applying func recursively to eltype(pa).
    et = eltype(pa)
    index_type = AbstractPPL.Index{NTuple{ndims(pa),Int},@NamedTuple{},AbstractPPL.Iden}
    new_vn_type = Core.Compiler.return_type(
        AbstractPPL.append_optic, Tuple{typeof(vn),index_type}
    )
    new_et = Core.Compiler.return_type(
        Tuple{typeof(_map_recursive!!),typeof(func),et,new_vn_type}
    )
    new_data = if new_et <: et
        # We can reuse the existing data array.
        pa.data
    else
        # We need to allocate a new data array.
        similar(pa.data, new_et)
    end
    # Keep a dictionary of already-seen ArrayLikeBlocks to avoid redundant computations.
    # This matters not only for performance, but also for correctness, because
    # _map_recursive!! may mutate the value, and we don't want to mutate it multiple times.
    albs_old_to_new = Dict{ArrayLikeBlock,ArrayLikeBlock}()
    @inbounds for i in CartesianIndices(pa.mask)
        if pa.mask[i]
            val = pa.data[i]
            is_alb = val isa ArrayLikeBlock
            if is_alb
                if haskey(albs_old_to_new, val)
                    new_data[i] = albs_old_to_new[val]
                    continue
                end
            end
            ind = is_alb ? val.ix : Tuple(i)
            new_vn = AbstractPPL.append_optic(vn, AbstractPPL.Index(ind, (;)))
            new_val = _map_recursive!!(func, pa.data[i], new_vn)
            new_data[i] = new_val
            if is_alb
                albs_old_to_new[val] = new_val
            end
        end
    end
    # The above type inference may be overly conservative, so we concretise the eltype.
    return _concretise_eltype!!(PartialArray(new_data, pa.mask))
end

function _map_recursive!!(func, alb::ArrayLikeBlock, vn)
    new_block = _map_recursive!!(func, alb.block, vn)
    sz_new = vnt_size(new_block)
    sz_old = vnt_size(alb.block)
    if !(sz_new isa SkipSizeCheck) && !(sz_old isa SkipSizeCheck) && sz_new != sz_old
        throw(
            DimensionMismatch(
                "map_pairs!! can't change the size of an ArrayLikeBlock. Tried to change " *
                "from $(sz_old) to $(sz_new).",
            ),
        )
    end
    return ArrayLikeBlock(new_block, alb.ix, alb.kw, alb.index_size)
end

# As above but with a prefix VarName `vn`.
@generated function _map_recursive!!(func, vnt::VarNamedTuple{Names}, vn::T) where {Names,T}
    exs = Expr[]
    for name in Names
        push!(
            exs,
            :(_map_recursive!!(
                func, vnt.data.$name, AbstractPPL.prefix(VarName{$(QuoteNode(name))}(), vn)
            )),
        )
    end
    return quote
        return VarNamedTuple(NamedTuple{Names}(($(exs...),)))
    end
end

"""
    map_pairs!!(func, vnt::VarNamedTuple)

Apply `func` to all key => value pairs of `vnt`, in place if possible.

`func` should accept a pair of `VarName` and value, and return the new value to be set.
"""
@generated function map_pairs!!(func, vnt::VarNamedTuple{Names}) where {Names}
    exs = Expr[]
    for name in Names
        push!(exs, :(_map_recursive!!(func, vnt.data.$name, VarName{$(QuoteNode(name))}())))
    end
    return quote
        return VarNamedTuple(NamedTuple{Names}(($(exs...),)))
    end
end

Base.foreach(func, vnt::VarNamedTuple) = map_pairs!!(p -> (func(p); p), vnt)

"""
    map_values!!(func, vnt::VarNamedTuple)

Apply `func` to elements of `vnt`, in place if possible.
"""
map_values!!(func, vnt::VarNamedTuple) = map_pairs!!(pair -> func(pair.second), vnt)

"""
    mapreduce(f, op, vnt::VarNamedTuple; init)

Apply `f` to all elements of `vnt`, and reduce the results using `op`, starting from `init`.

The order is the same as in `mapfoldl`, i.e. left-associative with `init` as the
left-most value.

`init` is a keyword argument to conform to the usual `mapreduce` interface in Base, but it
is not optional.

`f` op` should accept pairs of `varname => value`.
"""
@generated function Base.mapreduce(
    f, op, vnt::VarNamedTuple{Names}; init::InitType=nothing
) where {Names,InitType}
    if InitType === Nothing
        return quote
            throw(
                ArgumentError(
                    "mapreduce without init is not implemented for VarNamedTuple."
                ),
            )
        end
    end

    exs = Expr[:(result = init)]
    for name in Names
        push!(
            exs,
            quote
                result = _mapreduce_recursive(
                    f, op, vnt.data.$name, VarName{$(QuoteNode(name))}(), result
                )
            end,
        )
    end
    push!(exs, :(return result))
    return Expr(:block, exs...)
end

# Our mapreduce is always left-associative.
Base.mapfoldl(f, op, vnt::VarNamedTuple; init=nothing) = mapreduce(f, op, vnt; init=init)

_mapreduce_recursive(f, op, x, vn, init) = op(init, f(vn => x))
function _mapreduce_recursive(f, op, alb::ArrayLikeBlock, vn, init)
    return op(init, f(vn => alb.block))
end

# As above but with a prefix VarName `vn`.
@generated function _mapreduce_recursive(
    f, op, vnt::VarNamedTuple{Names}, vn, init
) where {Names}
    exs = Expr[:(result = init)]
    for name in Names
        push!(
            exs,
            quote
                result = _mapreduce_recursive(
                    f,
                    op,
                    vnt.data.$name,
                    AbstractPPL.prefix(VarName{$(QuoteNode(name))}(), vn),
                    result,
                )
            end,
        )
    end
    push!(exs, :(return result))
    return Expr(:block, exs...)
end

function _mapreduce_recursive(f, op, pa::PartialArray, vn, init)
    result = init
    et = eltype(pa)

    albs_seen = Set{ArrayLikeBlock}()
    @inbounds for i in CartesianIndices(pa.mask)
        if pa.mask[i]
            val = pa.data[i]
            is_alb = val isa ArrayLikeBlock
            if is_alb
                if val in albs_seen
                    continue
                end
                push!(albs_seen, val)
            end
            ind = is_alb ? val.ix : Tuple(i)
            new_vn = AbstractPPL.append_optic(vn, AbstractPPL.Index(ind, (;)))
            result = _mapreduce_recursive(f, op, pa.data[i], new_vn, result)
        end
    end
    return result
end

# TODO(mhauru) We could try to keep the return types of these more tight, rather than always
# return the same, abstract element type. Would that be better? It would be faster in some
# cases, but would be less consistent, and could result in a lot of allocations in the
# mapreduce, as the element type is gradually expanded.
Base.keys(vnt::VarNamedTuple) = mapreduce(first, push!, vnt; init=VarName[])
Base.values(vnt::VarNamedTuple) = mapreduce(pair -> pair.second, push!, vnt; init=Any[])

function Base.length(vnt::VarNamedTuple)
    len = 0
    for subdata in vnt.data
        len += subdata isa VarNamedTuple || subdata isa PartialArray ? length(subdata) : 1
    end
    return len
end
