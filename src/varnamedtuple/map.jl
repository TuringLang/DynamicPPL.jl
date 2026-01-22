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
        throw(KeyError(name))
    end
    subdata = _getindex_optic(vnt, name)
    new_subdata = func(subdata)
    # The allow_new=Val(false) is a performance optimisation: Since we've already checked
    # that the key exists, we know that no new fields will be created.
    return _setindex_optic!!(
        vnt,
        new_subdata,
        AbstractPPL.varname_to_optic(name),
        NoTemplate();
        allow_new=Val(false),
    )
end

"""
    _map_pairs_recursive!!(func, x, vn)

Call `func` on `vn => x`, except if `x` is a `VarNamedTuple` or `PartialArray`, in which
case call `_map_pairs_recursive!!` recursively on all their elements, updating `vn` with the
right prefix.

This is the internal implementation of `map_pairs!!`, but because it has a method defined
for literally every type in existence, we hide it behind the interface of the more
discriminating `map_pairs!!`. It makes the implementation a bit simpler, compared to
checking element types within `map_pairs!!` itself.
"""
_map_pairs_recursive!!(pairfunc, x, vn) = pairfunc(vn => x)
_map_values_recursive!!(func, x) = func(x)

# TODO(mhauru) The below is type unstable for some complex VarNames. My example case
# for which type stability fails is @varname(e.f[3].g.h[2].i). I don't understand this
# well, but I think it's just because constant propagation gives up at some point, and fails
# to go through the lines that figure out `new_et`. I could be wrong. I tried fixing this by
# lifting the first three lines of the function into a generated function, but that seems
# to run into trouble when trying to call Core.Compiler.return_type recursively on the same
# function. An earlier implementation of this function that only operated on the values,
# not on pairs of key => value, was type stable (presumably because it was a bit easier on
# constant propagation).
function _map_pairs_recursive!!(func, pa::PartialArray, vn)
    return if eltype(pa) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(pa)
        # There are (or might be) some ALBs
        _map_pairs_recursive_pa_alb!!(func, pa, vn)
    else
        # There are definitely no ALBs
        _map_pairs_recursive_pa_noalb!!(func, pa, vn)
    end
end
function _map_values_recursive!!(func, pa::PartialArray)
    return if eltype(pa) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(pa)
        # There are (or might be) some ALBs
        _map_values_recursive_pa_alb!!(func, pa)
    else
        # There are definitely no ALBs
        _map_values_recursive_pa_noalb!!(func, pa)
    end
end

function _map_pairs_recursive_pa_alb!!(pairfunc, pa::PartialArray, vn)
    # Ask the compiler to infer the return type of applying func recursively to eltype(pa).
    et = eltype(pa)
    index_type = AbstractPPL.Index{NTuple{ndims(pa),Int},@NamedTuple{},AbstractPPL.Iden}
    new_vn_type = Core.Compiler.return_type(
        AbstractPPL.append_optic, Tuple{typeof(vn),index_type}
    )
    new_et = Core.Compiler.return_type(
        Tuple{typeof(_map_pairs_recursive!!),typeof(pairfunc),et,new_vn_type}
    )
    new_data = if new_et <: et
        # We can reuse the existing data array.
        pa.data
    else
        # We need to allocate a new data array.
        similar(pa.data, new_et)
    end
    # If there are any ArrayLikeBlocks, we will need to modify the mask during the
    # iteration, so make a copy. The alternative, which avoids mutating, is to store a Dict
    # of already-visited ArrayLikeBlocks along with the result of applying `func` to them,
    # but that is more expensive.
    mask = copy(pa.mask)
    for i in CartesianIndices(mask)
        if mask[i]
            val = pa.data[i]
            if val isa ArrayLikeBlock
                # Create the new value, set it at all the places where it needs to be set,
                # and mark those places as visited.
                new_vn = AbstractPPL.append_optic(vn, AbstractPPL.Index(val.ix, val.kw))
                new_val = _map_pairs_recursive!!(pairfunc, val, new_vn)
                new_data[val.ix..., val.kw...] .= new_val
                mask[val.ix..., val.kw...] .= false
            else
                new_vn = AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(i), (;)))
                new_data[i] = _map_pairs_recursive!!(pairfunc, val, new_vn)
            end
        end
    end
    # The above type inference may be overly conservative, so we concretise the eltype.
    return _concretise_eltype!!(PartialArray(new_data, pa.mask))
end
function _map_values_recursive_pa_alb!!(func, pa::PartialArray)
    et = eltype(pa)
    new_et = Core.Compiler.return_type(
        Tuple{typeof(_map_values_recursive!!),typeof(func),et}
    )
    new_data = if new_et <: et
        pa.data
    else
        similar(pa.data, new_et)
    end
    mask = copy(pa.mask)
    for i in CartesianIndices(mask)
        if mask[i]
            val = pa.data[i]
            if val isa ArrayLikeBlock
                # Create the new value, set it at all the places where it needs to be set,
                # and mark those places as visited.
                new_val = _map_values_recursive!!(func, val)
                new_data[val.ix..., val.kw...] .= new_val
                mask[val.ix..., val.kw...] .= false
            else
                new_data[i] = _map_values_recursive!!(func, val)
            end
        end
    end
    return _concretise_eltype!!(PartialArray(new_data, pa.mask))
end

function _map_pairs_recursive_pa_noalb!!(pairfunc, pa::PartialArray, vn)
    # Don't have to faff with ALBs. Just map `func` over all elements that are set.
    mask = pa.mask
    ci = CartesianIndices(mask)
    ixs_subset = @view ci[mask]
    data_subset = @view pa.data[mask]
    index_type = AbstractPPL.Index{NTuple{ndims(pa),Int},@NamedTuple{},AbstractPPL.Iden}
    new_vn_type = Core.Compiler.return_type(
        AbstractPPL.append_optic, Tuple{typeof(vn),index_type}
    )
    et = eltype(pa)
    new_et = Core.Compiler.return_type(
        Tuple{typeof(_map_pairs_recursive!!),typeof(pairfunc),eltype(pa),new_vn_type}
    )
    function inner(data_i, ix_i)
        new_vn = AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(ix_i), (;)))
        return _map_pairs_recursive!!(pairfunc, data_i, new_vn)
    end
    return if new_et <: et
        # Can write into existing data array.
        map!(inner, view(pa.data, mask), data_subset, ixs_subset)
        pa
    else
        # Need to allocate a new data array.
        new_pa_data = similar(pa.data, new_et)
        map!(inner, view(new_pa_data, mask), data_subset, ixs_subset)
        _concretise_eltype!!(PartialArray(new_pa_data, pa.mask))
    end
end
function _map_values_recursive_pa_noalb!!(func, pa::PartialArray)
    mask = pa.mask
    et = eltype(pa)
    new_et = Core.Compiler.return_type(
        Tuple{typeof(_map_values_recursive!!),typeof(func),eltype(pa)}
    )
    return if new_et <: et
        # Can write into existing data array.
        new_pa_data = view(pa.data, mask)
        # According to the docstring of `map!` on 1.11, "unexpected behaviour" may happen if
        # the input and output alias each other. But the docstring on 1.12 says that you can
        # do this, so...
        # https://docs.julialang.org/en/v1/base/collections/#Julia-1.12-9b4e28751024c91d
        map!(Base.Fix1(_map_values_recursive!!, func), new_pa_data, new_pa_data)
        pa
    else
        # Need to allocate a new data array.
        new_pa_data = similar(pa.data, new_et)
        map!(
            Base.Fix1(_map_values_recursive!!, func),
            view(new_pa_data, mask),
            view(pa.data, mask),
        )
        _concretise_eltype!!(PartialArray(new_pa_data, pa.mask))
    end
end

function _check_size(new_block, old_block)
    sz_new = vnt_size(new_block)
    sz_old = vnt_size(old_block)
    if sz_new != sz_old
        throw(
            DimensionMismatch(
                "map_pairs!! can't change the size of a block. Tried to change " *
                "from $(sz_old) to $(sz_new).",
            ),
        )
    end
end
function _map_pairs_recursive!!(pairfunc, alb::ArrayLikeBlock, vn)
    new_block = _map_pairs_recursive!!(pairfunc, alb.block, vn)
    _check_size(new_block, alb.block)
    return ArrayLikeBlock(new_block, alb.ix, alb.kw, alb.index_size)
end
function _map_values_recursive!!(func, alb::ArrayLikeBlock)
    new_block = _map_values_recursive!!(func, alb.block)
    _check_size(new_block, alb.block)
    return ArrayLikeBlock(new_block, alb.ix, alb.kw, alb.index_size)
end

@generated function _map_pairs_recursive!!(
    pairfunc, vnt::VarNamedTuple{Names}, vn::T
) where {Names,T}
    exs = Expr[]
    for name in Names
        push!(
            exs,
            :(_map_pairs_recursive!!(
                pairfunc,
                vnt.data.$name,
                AbstractPPL.append_optic(vn, AbstractPPL.Property{$(QuoteNode(name))}()),
            )),
        )
    end
    return quote
        return VarNamedTuple(NamedTuple{Names}(($(exs...),)))
    end
end
@generated function _map_values_recursive!!(func, vnt::VarNamedTuple{Names}) where {Names}
    exs = Expr[]
    for name in Names
        push!(exs, :(_map_values_recursive!!(func, vnt.data.$name)))
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
        push!(
            exs,
            :(_map_pairs_recursive!!(func, vnt.data.$name, VarName{$(QuoteNode(name))}())),
        )
    end
    return quote
        return VarNamedTuple(NamedTuple{Names}(($(exs...),)))
    end
end

"""
    map_values!!(func, vnt::VarNamedTuple)

Apply `func` to elements of `vnt`, in place if possible.
"""
@generated function map_values!!(func, vnt::VarNamedTuple{Names}) where {Names}
    exs = Expr[]
    for name in Names
        push!(exs, :(_map_values_recursive!!(func, vnt.data.$name)))
    end
    return quote
        return VarNamedTuple(NamedTuple{Names}(($(exs...),)))
    end
end

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
    # If there are any ArrayLikeBlocks, we will need to modify the mask during the
    # iteration, so make a copy. The alternative, which avoids mutating, is to store a Set
    # of already-visited ArrayLikeBlocks, but that is more expensive.
    mask = if eltype(pa) <: ArrayLikeBlock || ArrayLikeBlock <: eltype(pa)
        copy(pa.mask)
    else
        pa.mask
    end
    for i in CartesianIndices(mask)
        if mask[i]
            val = pa.data[i]
            is_alb = val isa ArrayLikeBlock
            if is_alb
                # Don't visit the same ArrayLikeBlock multiple times.
                mask[val.ix..., val.kw...] .= false
            end
            optic = if is_alb
                AbstractPPL.Index(val.ix, val.kw)
            else
                AbstractPPL.Index(Tuple(i), (;))
            end
            new_vn = AbstractPPL.append_optic(vn, optic)
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
