has_fixed_support(vi::VarInfo) = vi.fixed_support[]
function set_fixed_support!(vi::VarInfo, b::Bool)
    return vi.fixed_support[] = vi.fixed_support[] && b
end

getmode(vi::VarInfo) = vi.mode
issynced(vi::VarInfo) = vi.synced[]
setsynced!(vi::VarInfo, b::Bool) = vi.synced[] = b
value(x) = x

Base.merge(t::AbstractVarInfo) = t
function Base.merge(
    t1::AbstractVarInfo,
    t2::AbstractVarInfo,
    ts::AbstractVarInfo...,
)
    return merge(merge(t1, t2), ts...)
end
function Base.merge(t1::TypedVarInfo, t2::TypedVarInfo)
    return VarInfo(
        merge(t1.metadata, t2.metadata),
        Ref(getlogp(t1) + getlogp(t2)),
        Ref(0),
        getmode(t1),
        Ref(has_fixed_support(t1) && has_fixed_support(t2)),
        Ref(issynced(t1) && issynced(t2)),
    )
end
Base.merge(t1::UntypedVarInfo, t2::TypedVarInfo) = merge(TypedVarInfo(t1), t2)
Base.merge(t1::TypedVarInfo, t2::UntypedVarInfo) = merge(t2, t1)

"""
    empty!(meta::Metadata)

Empty the fields of `meta`.

This is useful when using a sampling algorithm that assumes an empty `meta`, e.g. `SMC`.
"""
function empty!(meta::Metadata)
    empty!(meta.idcs)
    empty!(meta.vns)
    empty!(meta.ranges)
    empty!(meta.vals)
    empty!(meta.trans_vals)
    empty!(meta.dists)
    empty!(meta.gids)
    empty!(meta.orders)
    for k in keys(meta.flags)
        empty!(meta.flags[k])
    end

    return meta
end

"""
    empty!(vi::VarInfo)

Empty the fields of `vi.metadata` and reset `vi.logp[]` and `vi.num_produce[]` to
zeros.

This is useful when using a sampling algorithm that assumes an empty `vi`, e.g. `SMC`.
"""
function empty!(vi::VarInfo)
    _empty!(vi.metadata)
    resetlogp!(vi)
    reset_num_produce!(vi)
    setsynced!(vi, false)
    return vi
end
@inline _empty!(metadata::Metadata) = empty!(metadata)
@generated function _empty!(metadata::NamedTuple{names}) where {names}
    expr = Expr(:block)
    for f in names
        push!(expr.args, :(empty!(metadata.$f)))
    end
    return expr
end

# Removes the first element of a NamedTuple. The pairs in a NamedTuple are ordered, so this is well-defined.
if VERSION < v"1.1"
    _tail(nt::NamedTuple{names}) where names = NamedTuple{Base.tail(names)}(nt)
else
    _tail(nt::NamedTuple) = Base.tail(nt)
end

"""
    getmetadata(vi::VarInfo, vn::VarName)

Return the metadata in `vi` that belongs to `vn`.
"""
getmetadata(vi::VarInfo, vn::VarName) = vi.metadata
getmetadata(vi::TypedVarInfo, vn::VarName) = getfield(vi.metadata, getsym(vn))

"""
    getidx(vi::AbstractVarInfo, vn::VarName)

Return the index of `vn` in the metadata of `vi` corresponding to `vn`.
"""
getidx(vi::AbstractVarInfo, vn::VarName) = getmetadata(vi, vn).idcs[vn]

"""
    getrange(vi::AbstractVarInfo, vn::VarName)

Return the index range of `vn` in the metadata of `vi`.
"""
getrange(vi::AbstractVarInfo, vn::VarName) = getmetadata(vi, vn).ranges[getidx(vi, vn)]

"""
    getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})

Return the indices of `vns` in the metadata of `vi` corresponding to `vn`.
"""
function getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})
    return mapreduce(vn -> getrange(vi, vn), vcat, vns, init=Int[])
end

"""
    getinitdist(vi::AbstractVarInfo, vn::VarName)

Return the distribution from which `vn` was sampled in `vi`.
"""
getinitdist(vi::AbstractVarInfo, vn::VarName) = getmetadata(vi, vn).dists[getidx(vi, vn)]

"""
    getgid(vi::AbstractVarInfo, vn::VarName)

Return the set of sampler selectors associated with `vn` in `vi`.
"""
getgid(vi::AbstractVarInfo, vn::VarName) = getmetadata(vi, vn).gids[getidx(vi, vn)]

"""
    syms(vi::VarInfo)

Returns a tuple of the unique symbols of random variables sampled in `vi`.
"""
syms(vi::UntypedVarInfo) = Tuple(unique!(map(getsym, vi.metadata.vns)))  # get all symbols
syms(vi::TypedVarInfo) = keys(vi.metadata)

# Get all indices of variables belonging to SampleFromPrior:
#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to
#   the SampleFromPrior sampler
@inline function getidcs(vi::UntypedVarInfo, ::SampleFromPrior)
    return filter(i -> isempty(vi.metadata.gids[i]) , 1:length(vi.metadata.gids))
end
# Get a NamedTuple of all the indices belonging to SampleFromPrior, one for each symbol
@inline function getidcs(vi::TypedVarInfo, ::SampleFromPrior)
    return getidcs(vi.metadata)
end
@generated function getidcs(metadata::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findinds(metadata.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# Get all indices of variables belonging to a given sampler
@inline function getidcs(vi::AbstractVarInfo, spl::Sampler)
    return getidcs(vi, spl.selector, Val(getspace(spl)))
end
@inline getidcs(vi::UntypedVarInfo, s::Selector, space::Val) = findinds(vi.metadata, s, space)
@inline getidcs(vi::TypedVarInfo, s::Selector, space::Val) = getidcs(vi.metadata, s, space)
# Get a NamedTuple for all the indices belonging to a given selector for each symbol
@generated function getidcs(metadata::NamedTuple{names}, s::Selector, ::Val{space}) where {names, space}
    exprs = []
    # Iterate through each varname in metadata.
    for f in names
        # If the varname is in the sampler space
        # or the sample space is empty (all variables)
        # then return the indices for that variable.
        if inspace(f, space) || length(space) == 0
            push!(exprs, :($f = findinds(metadata.$f, s, Val($space))))
        end
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end
@inline function findinds(f_meta, s, ::Val{space}) where {space}
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    return filter((i) ->
        (s in f_meta.gids[i] || isempty(f_meta.gids[i])) &&
        (isempty(space) || inspace(f_meta.vns[i], space)), 1:length(f_meta.gids))
end
@inline function findinds(f_meta)
    # Get all the idcs of the vns
    return filter((i) -> isempty(f_meta.gids[i]), 1:length(f_meta.gids))
end

# Get all vns of variables belonging to spl
getvns(vi::AbstractVarInfo, spl::Sampler) = getvns(vi, spl.selector, Val(getspace(spl)))
getvns(vi::AbstractVarInfo, spl::Union{SampleFromPrior, SampleFromUniform}) = getvns(vi, Selector(), Val(()))
getvns(vi::UntypedVarInfo, s::Selector, space::Val) = view(vi.metadata.vns, getidcs(vi, s, space))
function getvns(vi::TypedVarInfo, s::Selector, space::Val)
    return getvns(vi.metadata, getidcs(vi, s, space))
end
# Get a NamedTuple for all the `vns` of indices `idcs`, one entry for each symbol
@generated function getvns(metadata, idcs::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = metadata.$f.vns[idcs.$f]))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# Get the index (in vals) ranges of all the vns of variables belonging to spl
@inline function getranges(vi::AbstractVarInfo, spl::Sampler)
    ## Uncomment the spl.info stuff when it is concretely typed, not Dict{Symbol, Any}
    #if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    #if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
    #    spl.info[:ranges]
    #else
        #spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        ranges = getranges(vi, spl.selector, Val(getspace(spl)))
        #spl.info[:ranges] = ranges
        return ranges
    #end
end
# Get the index (in vals) ranges of all the vns of variables belonging to selector `s` in `space`
@inline function getranges(vi::AbstractVarInfo, s::Selector, space)
    return getranges(vi, getidcs(vi, s, space))
end
@inline function getranges(vi::UntypedVarInfo, idcs::Vector{Int})
    mapreduce(i -> vi.metadata.ranges[i], vcat, idcs, init=Int[])
end
@inline getranges(vi::TypedVarInfo, idcs::NamedTuple) = getranges(vi.metadata, idcs)

@generated function getranges(metadata::NamedTuple, idcs::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findranges(metadata.$f.ranges, idcs.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end
@inline function findranges(f_ranges, f_idcs)
    return mapreduce(i -> f_ranges[i], vcat, f_idcs, init=Int[])
end

"""
    set_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `true` in `vi`.
"""
function set_flag!(vi::AbstractVarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)] = true
end

# Functions defined only for UntypedVarInfo
"""
    keys(vi::UntypedVarInfo)

Return an iterator over all `vns` in `vi`.
"""
keys(vi::UntypedVarInfo) = keys(vi.metadata.idcs)

"""
    setgid!(vi::VarInfo, gid::Selector, vn::VarName)

Add `gid` to the set of sampler selectors associated with `vn` in `vi`.
"""
function setgid!(vi::AbstractVarInfo, gid::Selector, vn::VarName; overwrite=false)
    gids = getmetadata(vi, vn).gids[getidx(vi, vn)]
    overwrite && empty!(gids)
    push!(gids, gid)
    return vi
end

"""
    istrans(vi::VarInfo, vn::VarName)

Return true if `vn`'s values in `vi` are transformed to Euclidean space, and false if
they are in the support of `vn`'s distribution.
"""
function istrans(vi::AbstractVarInfo, vn::VarName)
    return is_flagged(vi, vn, "trans")
end

"""
    getlogp(vi::VarInfo)

Return the log of the joint probability of the observed data and parameters sampled in
`vi`.
"""
getlogp(vi::AbstractVarInfo) = vi.logp[]

"""
    setlogp!(vi::VarInfo, logp)

Set the log of the joint probability of the observed data and parameters sampled in
`vi` to `logp`.
"""
function setlogp!(vi::VarInfo, logp)
    vi.logp[] = logp
    return vi
end

"""
    acclogp!(vi::VarInfo, logp)

Add `logp` to the value of the log of the joint probability of the observed data and
parameters sampled in `vi`.
"""
function acclogp!(vi::VarInfo, logp)
    vi.logp[] += logp
    return vi
end

"""
    resetlogp!(vi::AbstractVarInfo)

Reset the value of the log of the joint probability of the observed data and parameters
sampled in `vi` to 0.
"""
resetlogp!(vi::AbstractVarInfo) = setlogp!(vi, zero(getlogp(vi)))

"""
    get_num_produce(vi::VarInfo)

Return the `num_produce` of `vi`.
"""
get_num_produce(vi::VarInfo) = vi.num_produce[]

"""
    set_num_produce!(vi::VarInfo, n::Int)

Set the `num_produce` field of `vi` to `n`.
"""
set_num_produce!(vi::VarInfo, n::Int) = vi.num_produce[] = n

"""
    increment_num_produce!(vi::VarInfo)

Add 1 to `num_produce` in `vi`.
"""
increment_num_produce!(vi::VarInfo) = vi.num_produce[] += 1

"""
    reset_num_produce!(vi::AbstractVarInfo)

Reset the value of `num_produce` the log of the joint probability of the observed data
and parameters sampled in `vi` to 0.
"""
reset_num_produce!(vi::AbstractVarInfo) = set_num_produce!(vi, 0)

"""
    isempty(vi::VarInfo)

Return true if `vi` is empty and false otherwise.
"""
isempty(vi::UntypedVarInfo) = isempty(vi.metadata.idcs)
isempty(vi::TypedVarInfo) = _isempty(vi.metadata)
@generated function _isempty(metadata::NamedTuple{names}) where {names}
    expr = Expr(:&&, :true)
    for f in names
        push!(expr.args, :(isempty(metadata.$f.idcs)))
    end
    return expr
end

"""
    tonamedtuple(vi::VarInfo)

Convert a `vi` into a `NamedTuple` where each variable symbol maps to the values and 
indexing string of the variable.

For example, a model that had a vector of vector-valued
variables `x` would return

```julia
(x = ([1.5, 2.0], [3.0, 1.0], ["x[1]", "x[2]"]), )
```
"""
function tonamedtuple(vi::UntypedVarInfo)
    return tonamedtuple(TypedVarInfo(vi))
end
function tonamedtuple(vi::TypedVarInfo)
    return tonamedtuple(vi.metadata, vi)
end
@generated function tonamedtuple(metadata::NamedTuple{names}, vi::VarInfo) where {names}
    length(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    map(names) do f
        push!(expr.args, Expr(:(=), f, :(getindex.(Ref(vi), metadata.$f.vns), string.(metadata.$f.vns))))
    end
    return expr
end

function Base.eltype(vi::AbstractVarInfo, spl::Union{AbstractSampler, SampleFromPrior})
    T = eltype(Core.Compiler.return_type(getindex, Tuple{typeof(vi), typeof(spl)}))
    if T === Union{}
        # To throw a meaningful error
        return eltype(vi[spl])
    else
        return T
    end
end

"""
    haskey(vi::VarInfo, vn::VarName)

Check whether `vn` has been sampled in `vi`.
"""
haskey(vi::VarInfo, vn::VarName) = haskey(getmetadata(vi, vn).idcs, vn)
function haskey(vi::TypedVarInfo, vn::VarName)
    metadata = vi.metadata
    Tmeta = typeof(metadata)
    return getsym(vn) in fieldnames(Tmeta) && haskey(getmetadata(vi, vn).idcs, vn)
end

"""
    hassymbol(vi::VarInfo, vn::VarName)

Check whether the symbol of `vn` has been sampled in `vi`.
"""
hassymbol(vi::UntypedVarInfo, vn::VarName) = any(keys(vi)) do vn2
    getsym(vn) == getsym(vn2)
end
hassymbol(vi::TypedVarInfo, vn::VarName) = haskey(vi.metadata, getsym(vn))

"""
    setorder!(vi::VarInfo, vn::VarName, index::Int)

Set the `order` of `vn` in `vi` to `index`, where `order` is the number of `observe
statements run before sampling `vn`.
"""
function setorder!(vi::VarInfo, vn::VarName, index::Int)
    metadata = getmetadata(vi, vn)
    if metadata.orders[metadata.idcs[vn]] != index
        metadata.orders[metadata.idcs[vn]] = index
    end
    return vi
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

"""
    is_flagged(vi::VarInfo, vn::VarName, flag::String)

Check whether `vn` has a true value for `flag` in `vi`.
"""
function is_flagged(vi::VarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)]
end

"""
    unset_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `false` in `vi`.
"""
function unset_flag!(vi::VarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)] = false
end

"""
    set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)

Set the `"del"` flag of variables in `vi` with `order > vi.num_produce[]` to `true`.
"""
function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a vector
    gidcs = getidcs(vi, spl)
    if get_num_produce(vi) == 0
        for i = length(gidcs):-1:1
          vi.metadata.flags["del"][gidcs[i]] = true
        end
    else
        for i in 1:length(vi.orders)
            if i in gidcs && vi.orders[i] > get_num_produce(vi)
                vi.metadata.flags["del"][i] = true
            end
        end
    end
    return nothing
end
function set_retained_vns_del_by_spl!(vi::TypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
    gidcs = getidcs(vi, spl)
    return _set_retained_vns_del_by_spl!(vi.metadata, gidcs, get_num_produce(vi))
end
@generated function _set_retained_vns_del_by_spl!(metadata, gidcs::NamedTuple{names}, num_produce) where {names}
    expr = Expr(:block)
    for f in names
        f_gidcs = :(gidcs.$f)
        f_orders = :(metadata.$f.orders)
        f_flags = :(metadata.$f.flags)
        push!(expr.args, quote
            # Set the flag for variables with symbol `f`
            if num_produce == 0
                for i = length($f_gidcs):-1:1
                    $f_flags["del"][$f_gidcs[i]] = true
                end
            else
                for i in 1:length($f_orders)
                    if i in $f_gidcs && $f_orders[i] > num_produce
                        $f_flags["del"][i] = true
                    end
                end
            end
        end)
    end
    return expr
end

"""
    updategid!(vi::VarInfo, vn::VarName, spl::Sampler; overwrite=false)

Set `vn`'s `gid` to `Set([spl.selector])`, if `vn` does not have a sampler selector linked
and `vn`'s symbol is in the space of `spl`.
"""
function updategid!(vi::AbstractVarInfo, vn::VarName, spl::Sampler; overwrite=false)
    if inspace(vn, getspace(spl))
        setgid!(vi, spl.selector, vn; overwrite=overwrite)
    end
end

"""
    set_namedtuple!(vi::VarInfo, nt::NamedTuple)

Places the values of a `NamedTuple` into the relevant places of a `VarInfo`.
"""
function set_namedtuple!(vi::VarInfo, nt::NamedTuple)
    @assert !islinked(vi)
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns

        n_vns = length(vns)
        n_vals = length(vals)
        v_isarr = vals isa AbstractArray
        
        if v_isarr && n_vals == 1 && n_vns > 1
            for (vn, val) in zip(vns, vals[1])
                vi[vn] = val
            end
        elseif v_isarr && n_vals > 1 && n_vns == 1
            vi[vns[1]] = vals
        elseif v_isarr && n_vals == n_vns > 1
            for (vn, val) in zip(vns, vals)
                vi[vn] = val
            end
        elseif v_isarr && n_vals == 1 && n_vns == 1
            vi[vns[1]] = vals[1]
        elseif !(v_isarr)
            vi[vns[1]] = vals
        else
            error("Cannot assign `NamedTuple` to `VarInfo`")
        end
    end
end

function updategid!(vi::AbstractVarInfo, spls::Tuple{Vararg{AbstractSampler}}; overwrite=false)
    foreach(spls) do spl
        updategid!(vi, spl; overwrite=overwrite)
    end
    return vi
end
function updategid!(vi::UntypedVarInfo, spl::AbstractSampler; overwrite=false)
    vns = vi.metadata.vns
    if inspace(vns[1], getspace(spl))
        for vn in vns
            updategid!(vi, vn, spl; overwrite=overwrite)
        end
    end
    return vi
end
function updategid!(vi::TypedVarInfo, spl::AbstractSampler; overwrite=false)
    foreach(keys(vi.metadata)) do k
        vns = vi.metadata[k].vns
        if inspace(vns[1], getspace(spl))
            for vn in vns
                updategid!(vi, vn, spl; overwrite=overwrite)
            end
        end
    end    
    return vi
end

function removedel!(vi::VarInfo)
    removedel!(vi.metadata)
    return vi
end
removedel!(md::NamedTuple{<:Any, <:Tuple{Vararg{Metadata}}}) = map(removedel!, md)
function removedel!(md::Metadata)
    vns_to_remove = similar(md.vns, 0)
    inds_to_keep = Int[]
    new_idcs = empty(md.idcs)
    i = 1
    for vn in md.vns
        idx = md.idcs[vn]
        if !(md.flags["del"][idx])
            push!(inds_to_keep, idx)
            new_idcs[vn] = i
            i += 1
        end
    end
    new_vns = md.vns[inds_to_keep]
    new_dists = md.dists[inds_to_keep]
    new_gids = md.gids[inds_to_keep]
    new_orders = md.orders[inds_to_keep]
    new_flags = Dict(k => md.flags[k][inds_to_keep] for k in keys(md.flags))

    nvals = length(inds_to_keep) == 0 ? 0 : sum(length, md.ranges[inds_to_keep])
    new_vals = similar(md.vals, nvals)
    new_trans_vals = similar(new_vals)
    new_ranges = similar(md.ranges, length(inds_to_keep))
    last_ind = 0
    for (_i, i) in enumerate(inds_to_keep)
        first_ind = last_ind + 1
        last_ind = last_ind + length(md.ranges[i])
        new_vals[first_ind:last_ind] = md.vals[md.ranges[i]]
        new_trans_vals[first_ind:last_ind] = md.trans_vals[md.ranges[i]]
        new_ranges[_i] = first_ind:last_ind
    end

    md.idcs.age = new_idcs.age
    md.idcs.count = new_idcs.count
    md.idcs.idxfloor = new_idcs.idxfloor
    copyto!(md.idcs.keys, new_idcs.keys)
    resize!(md.idcs.keys, length(new_idcs.keys))
    md.idcs.maxprobe = new_idcs.maxprobe
    copyto!(md.idcs.slots, new_idcs.slots)
    resize!(md.idcs.slots, length(new_idcs.slots))
    copyto!(md.idcs.vals, new_idcs.vals)
    resize!(md.idcs.vals, length(new_idcs.vals))

    copyto!(md.vns, new_vns)
    resize!(md.vns, length(new_vns))
    copyto!(md.dists, new_dists)
    resize!(md.dists, length(new_dists))
    copyto!(md.gids, new_gids)
    resize!(md.gids, length(new_gids))
    copyto!(md.orders, new_orders)
    resize!(md.orders, length(new_orders))
    copyto!(md.ranges, new_ranges)
    resize!(md.ranges, length(new_ranges))
    for k in keys(md.flags)
        copyto!(md.flags[k], new_flags[k])
        resize!(md.flags[k], length(new_flags[k]))
    end
    copyto!(md.vals, new_vals)
    resize!(md.vals, length(new_vals))
    copyto!(md.trans_vals, new_trans_vals)
    resize!(md.trans_vals, length(new_trans_vals))

    return md
end
