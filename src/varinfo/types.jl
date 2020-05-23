####################
# VarInfo metadata #
####################

"""
The `Metadata` struct stores some metadata about the parameters of the model. This helps
query certain information about a variable, such as its distribution, which samplers
sample this variable, its value and whether this value is transformed to real space or
not.

Let `md` be an instance of `Metadata`:
- `md.vns` is the vector of all `VarName` instances.
- `md.idcs` is the dictionary that maps each `VarName` instance to its index in
 `md.vns`, `md.ranges` `md.dists`, `md.orders` and `md.flags`.
- `md.vns[md.idcs[vn]] == vn`.
- `md.dists[md.idcs[vn]]` is the distribution of `vn`.
- `md.gids[md.idcs[vn]]` is the set of algorithms used to sample `vn`. This is used in
 the Gibbs sampling process.
- `md.orders[md.idcs[vn]]` is the number of `observe` statements before `vn` is sampled.
- `md.ranges[md.idcs[vn]]` is the index range of `vn` in `md.vals`.
- `md.vals[md.ranges[md.idcs[vn]]]` is the vector of values of corresponding to `vn`.
- `md.flags` is a dictionary of true/false flags. `md.flags[flag][md.idcs[vn]]` is the
 value of `flag` corresponding to `vn`.

To make `md::Metadata` type stable, all the `md.vns` must have the same symbol
and distribution type. However, one can have a Julia variable, say `x`, that is a
matrix or a hierarchical array sampled in partitions, e.g.
`x[1][:] ~ MvNormal(zeros(2), 1.0); x[2][:] ~ MvNormal(ones(2), 1.0)`, and is managed by
a single `md::Metadata` so long as all the distributions on the RHS of `~` are of the
same type. Type unstable `Metadata` will still work but will have inferior performance.
When sampling, the first iteration uses a type unstable `Metadata` for all the
variables then a specialized `Metadata` is used for each symbol along with a function
barrier to make the rest of the sampling type stable.
"""
struct Metadata{TIdcs <: Dict{<:VarName,Int}, TDists <: AbstractVector{<:Distribution}, TVN <: AbstractVector{<:VarName}, TVal <: AbstractVector{<:Real}, TTransVal <: AbstractVector{<:Real}, TGIds <: AbstractVector{Set{Selector}}}
    # Mapping from the `VarName` to its integer index in `vns`, `ranges` and `dists`
    idcs        ::    TIdcs # Dict{<:VarName,Int}

    # Vector of identifiers for the random variables, where `vns[idcs[vn]] == vn`
    vns         ::    TVN # AbstractVector{<:VarName}

    # Vector of index ranges in `vals` corresponding to `vns`
    # Each `VarName` `vn` has a single index or a set of contiguous indices in `vals`
    ranges      ::    Vector{UnitRange{Int}}

    # Vector of values of all the univariate, multivariate and matrix variables
    # The value(s) of `vn` is/are `vals[ranges[idcs[vn]]]`
    vals        ::    TVal # AbstractVector{<:Real}

    # Vector of the transformed values of all the univariate, multivariate and matrix 
    # variablse. The value(s) of `vn` is/are `vals[ranges[idcs[vn]]]`
    trans_vals  ::    TTransVal # AbstractVector{<:Real}

    # Vector of distributions correpsonding to `vns`
    dists       ::    TDists # AbstractVector{<:Distribution}

    # Vector of sampler ids corresponding to `vns`
    # Each random variable can be sampled using multiple samplers, e.g. in Gibbs, hence the `Set`
    gids        ::    TGIds # AbstractVector{Set{Selector}}

    # Number of `observe` statements before each random variable is sampled
    orders      ::    Vector{Int}

    # Each `flag` has a `BitVector` `flags[flag]`, where `flags[flag][i]` is the true/false flag value corresonding to `vns[i]`
    flags       ::    Dict{String, BitVector}
end

"""
    Metadata()

Construct an empty type unstable instance of `Metadata`.
"""
function Metadata()
    vals  = Vector{Real}()
    trans_vals  = Vector{Real}()
    flags = Dict{String, BitVector}()
    flags["del"] = BitVector()
    flags["trans"] = BitVector()

    return Metadata(
        Dict{VarName, Int}(),
        Vector{VarName}(),
        Vector{UnitRange{Int}}(),
        vals,
        trans_vals,
        Vector{Distribution}(),
        Vector{Set{Selector}}(),
        Vector{Int}(),
        flags
    )
end

###########
# VarInfo #
###########

abstract type VarInfoMode end

"""
    LinkMode

For any random variable whose `"trans"` flag is set to `true`:
1. The transformed values are used in `getindex` and `setindex!`.
2. The untransformed values are computed and cached, and
3. The `logpdf_with_trans` is computed with `trans` set as `true`.

For random variables whose `"trans"` flag is set to `false`, this is equivalent to 
the `StandardMode`. This model can be used when running HMC or MAP in the 
unconstrained space.
"""
struct LinkMode <: VarInfoMode end

"""
    InitLinkMode

For any random variable whose `"trans"` flag is set to `true`:
1. The untransformed values are used in `getindex` and `setindex!`.
2. The transformed values are computed and cached, and
3. The `logpdf_with_trans` is computed with `trans` set as `true`.

For random variables whose `"trans"` flag is set to `false`, this is equivalent to 
the `StandardMode`. This mode can be used to initialize a `VarInfo` for HMC or MAP.
"""
struct InitLinkMode <: VarInfoMode end

"""
    StandardMode

For all random variables:
1. The untransformed values are used in `getindex` and `setindex!`.
2. The `logpdf` is computed, ie. `logpdf_with_trans` with `trans` as `false`.

This mode can be used when running non-HMC samplers or when doing MAP on the 
constrained support directly.
"""
struct StandardMode <: VarInfoMode end

"""
```
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
```

A light wrapper over one or more instances of `Metadata`. Let `vi` be an instance of
`VarInfo`. If `vi isa VarInfo{<:Metadata}`, then only one `Metadata` instance is used
for all the sybmols. `VarInfo{<:Metadata}` is aliased `UntypedVarInfo`. If
`vi isa VarInfo{<:NamedTuple}`, then `vi.metadata` is a `NamedTuple` that maps each
symbol used on the LHS of `~` in the model to its `Metadata` instance. The latter allows
for the type specialization of `vi` after the first sampling iteration when all the
symbols have been observed. `VarInfo{<:NamedTuple}` is aliased `TypedVarInfo`.

Note: It is the user's responsibility to ensure that each "symbol" is visited at least
once whenever the model is called, regardless of any stochastic branching. Each symbol
refers to a Julia variable and can be a hierarchical array of many random variables, e.g. `x[1] ~ ...` and `x[2] ~ ...` both have the same symbol `x`.
"""
struct VarInfo{Tmeta <:  Union{Metadata, NamedTuple}, Tlogp, Tmode <: VarInfoMode} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
    mode::Tmode
    fixed_support::Base.RefValue{Bool}
    synced::Base.RefValue{Bool}
end
const UntypedVarInfo = VarInfo{<:Metadata}
const TypedVarInfo = VarInfo{<:NamedTuple}

function TypedVarInfo(model::Model, ctx = DefaultContext())
    vi = VarInfo()
    model(vi, SampleFromPrior(), ctx)
    return TypedVarInfo(vi)
end
function TypedVarInfo(model::Model, n::Integer, ctx = DefaultContext())
    return mapreduce(merge, 1:n) do _
        vi = VarInfo()
        model(vi, SampleFromPrior(), ctx)
        TypedVarInfo(vi)
    end
end
function VarInfo(old_vi::UntypedVarInfo, spl, x::AbstractVector)
    new_vi = deepcopy(old_vi)
    new_vi[spl] = x
    return new_vi
end
function VarInfo(old_vi::TypedVarInfo, spl, x::AbstractVector)
    md = newmetadata(old_vi.metadata, Val(getspace(spl)), x, Val(getmode(old_vi) isa LinkMode))
    return VarInfo(
        md,
        Base.RefValue{eltype(x)}(getlogp(old_vi)),
        Ref(get_num_produce(old_vi)),
        old_vi.mode,
        old_vi.fixed_support,
        Ref(false),
    )
end
@generated function newmetadata(metadata::NamedTuple{names}, ::Val{space}, x, ::Val{islinked}) where {names, space, islinked}
    exprs = []
    offset = :(0)
    for f in names
        mdf = :(metadata.$f)
        if inspace(f, space) || length(space) == 0
            len = :(length($mdf.vals))
            if islinked
                push!(exprs, :($f = Metadata($mdf.idcs,
                                            $mdf.vns,
                                            $mdf.ranges,
                                            $mdf.vals,
                                            x[($offset + 1):($offset + $len)],
                                            $mdf.dists,
                                            $mdf.gids,
                                            $mdf.orders,
                                            $mdf.flags
                                        )
                                )
                )
            else
                push!(exprs, :($f = Metadata($mdf.idcs,
                                            $mdf.vns,
                                            $mdf.ranges,
                                            x[($offset + 1):($offset + $len)],
                                            $mdf.trans_vals,
                                            $mdf.dists,
                                            $mdf.gids,
                                            $mdf.orders,
                                            $mdf.flags
                                        )
                                )
                )
            end
            offset = :($offset + $len)
        else
            push!(exprs, :($f = $mdf))
        end
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

VarInfo(meta=Metadata()) = VarInfo(meta, Ref{Float64}(0.0), Ref(0), StandardMode(), Ref(true), Ref(false))

"""
    TypedVarInfo(vi::UntypedVarInfo)

This function finds all the unique `sym`s from the instances of `VarName{sym}` found in
`vi.metadata.vns`. It then extracts the metadata associated with each symbol from the
global `vi.metadata` field. Finally, a new `VarInfo` is created with a new `metadata` as
a `NamedTuple` mapping from symbols to type-stable `Metadata` instances, one for each
symbol.
"""
function TypedVarInfo(vi::UntypedVarInfo)
    meta = vi.metadata
    new_metas = Metadata[]
    # Symbols of all instances of `VarName{sym}` in `vi.vns`
    syms_tuple = Tuple(syms(vi))
    for s in syms_tuple
        # Find all indices in `vns` with symbol `s`
        inds = findall(vn -> getsym(vn) === s, meta.vns)
        n = length(inds)
        # New `vns`
        sym_vns = getindex.((meta.vns,), inds)
        # New idcs
        sym_idcs = Dict(a => i for (i, a) in enumerate(sym_vns))
        # New dists
        sym_dists = getindex.((meta.dists,), inds)
        # New gids, can make a resizeable FillArray
        sym_gids = getindex.((meta.gids,), inds)
        @assert length(sym_gids) <= 1 ||
            all(x -> x == sym_gids[1], @view sym_gids[2:end])
        # New orders
        sym_orders = getindex.((meta.orders,), inds)
        # New flags
        sym_flags = Dict(a => meta.flags[a][inds] for a in keys(meta.flags))

        # Extract new ranges and vals
        _ranges = getindex.((meta.ranges,), inds)
        # `copy.()` is a workaround to reduce the eltype from Real to Int or Float64
        _vals = [copy.(meta.vals[_ranges[i]]) for i in 1:n]
        _trans_vals = [copy.(meta.trans_vals[_ranges[i]]) for i in 1:n]
        sym_ranges = Vector{eltype(_ranges)}(undef, n)
        start = 0
        for i in 1:n
            sym_ranges[i] = start + 1 : start + length(_vals[i])
            start += length(_vals[i])
        end
        sym_vals = foldl(vcat, _vals)
        sym_trans_vals = foldl(vcat, _trans_vals)

        push!(
            new_metas,
            Metadata(
                sym_idcs, sym_vns, sym_ranges, sym_vals, sym_trans_vals, 
                sym_dists, sym_gids, sym_orders, sym_flags
            )
        )
    end
    logp = getlogp(vi)
    num_produce = get_num_produce(vi)
    nt = NamedTuple{syms_tuple}(Tuple(new_metas))
    return VarInfo(nt, Ref(logp), Ref(num_produce), vi.mode, vi.fixed_support, vi.synced)
end
TypedVarInfo(vi::TypedVarInfo) = vi


####
#### Printing
####

function Base.show(io::IO, ::MIME"text/plain", vi::UntypedVarInfo)
    vi_str = """
    /=======================================================================
    | VarInfo
    |-----------------------------------------------------------------------
    | Varnames  :   $(string(vi.metadata.vns))
    | Range     :   $(vi.metadata.ranges)
    | Vals      :   $(vi.metadata.vals)
    | GIDs      :   $(vi.metadata.gids)
    | Orders    :   $(vi.metadata.orders)
    | Logp      :   $(getlogp(vi))
    | #produce  :   $(get_num_produce(vi))
    | flags     :   $(vi.metadata.flags)
    \\=======================================================================
    """
    print(io, vi_str)
end

const _MAX_VARS_SHOWN = 4

function _show_varnames(io::IO, vi)
    md = vi.metadata
    vns = md.vns

    groups = Dict{Symbol, Vector{VarName}}()
    for vn in vns
        group = get!(() -> Vector{VarName}(), groups, getsym(vn))
        push!(group, vn)
    end

    print(io, length(groups), length(groups) == 1 ? " variable " : " variables ", "(")
    join(io, Iterators.take(keys(groups), _MAX_VARS_SHOWN), ", ")
    length(groups) > _MAX_VARS_SHOWN && print(io, ", ...")
    print(io, "), dimension ", sum(prod(size(md.vals[md.ranges[md.idcs[vn]]])) for vn in vns))
end

function Base.show(io::IO, vi::UntypedVarInfo)
    print(io, "VarInfo (")
    _show_varnames(io, vi)
    print(io, "; logp: ", round(getlogp(vi), digits=3))
    print(io, ")")
end
