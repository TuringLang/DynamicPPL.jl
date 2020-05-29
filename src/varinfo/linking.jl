"""
    islinked(vi::VarInfo, spl::Sampler)

Check whether `vi` is in the transformed space for a particular sampler `spl`.

Turing's Hamiltonian samplers use the `link` and `invlink` functions from 
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) to map a constrained variable
(for example, one bounded to the space `[0, 1]`) from its constrained space to the set of 
real numbers. `islinked` checks if the number is in the constrained space or the real space.
"""
function islinked(vi::UntypedVarInfo, spl::Sampler)
    vns = getvns(vi, spl)
    return islinked(vi) && istrans(vi, vns[1])
end
function islinked(vi::TypedVarInfo, spl::Sampler)
    vns = getvns(vi, spl)
    return islinked(vi) && _islinked(vi, vns)
end
@generated function _islinked(vi, vns::NamedTuple{names}) where {names}
    out = []
    for f in names
        push!(out, :(length(vns.$f) == 0 ? false : istrans(vi, vns.$f[1])))
    end
    return Expr(:||, false, out...)
end
function islinked_and_trans(vi::AbstractVarInfo, vn::VarName)
    return islinked(vi) && istrans(vi, vn)
end

function Bijectors.link(vi::VarInfo)
    return VarInfo(
        vi.metadata,
        vi.logp,
        vi.num_produce,
        LinkMode(),
        vi.fixed_support,
        vi.synced,
    )
end
function initlink(vi::VarInfo)
    return VarInfo(
        vi.metadata,
        vi.logp,
        vi.num_produce,
        InitLinkMode(),
        vi.fixed_support,
        vi.synced,
    )
end
function Bijectors.invlink(vi::VarInfo)
    return VarInfo(
        vi.metadata,
        vi.logp,
        vi.num_produce,
        StandardMode(),
        vi.fixed_support,
        vi.synced,
    )
end
islinked(vi::AbstractVarInfo) = getmode(vi) isa LinkMode || getmode(vi) isa InitLinkMode

# X -> R for all variables associated with given sampler
"""
    init_dist_link!(vi::VarInfo, spl::Sampler)
Transform the values of the random variables sampled by `spl` in `vi` from the support
of their distributions to the Euclidean space and set their corresponding `"trans"`
flag values to `true`.
"""
function init_dist_link!(vi::UntypedVarInfo, spl::Sampler)
    # TODO: Change to a lazy iterator over `vns`
    vns = getvns(vi, spl)
    for vn in vns
        dist = getinitdist(vi, vn)
        initlink(vi)[vn, dist]
    end
    return vi
end
function init_dist_link!(vi::TypedVarInfo, spl::AbstractSampler)
    vns = getvns(vi, spl)
    _init_dist_link!(vi.metadata, vi, vns, Val(getspace(spl)))
    return vi
end
@generated function _init_dist_link!(metadata::NamedTuple{names}, vi, vns, ::Val{space}) where {names, space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(expr.args, quote
                f_vns = vi.metadata.$f.vns
                # Iterate over all `f_vns` and transform
                for vn in f_vns
                    dist = getinitdist(vi, vn)
                    initlink(vi)[vn, dist]
                end
            end)
        end
    end
    return expr
end

function invlink!(vi::AbstractVarInfo, spl::AbstractSampler, model)
    settrans!(vi, spl)
    if !issynced(vi)
        if has_fixed_support(vi)
            init_dist_invlink!(vi, spl)
        else
            model(link(vi), spl)
        end
        setsynced!(vi, true)
    end
    return vi
end
function link!(vi::AbstractVarInfo, spl::AbstractSampler, model)
    settrans!(vi, spl)
    if !issynced(vi)
        if has_fixed_support(vi)
            init_dist_link!(vi, spl)
        else
            model(initlink(vi), spl)
        end
        setsynced!(vi, true)
    end
    return vi
end

# R -> X for all variables associated with given sampler
"""
    init_dist_invlink!(vi::VarInfo, spl::AbstractSampler)
Transform the values of the random variables sampled by `spl` in `vi` from the
Euclidean space back to the support of their distributions and sets their corresponding
`"trans"` flag values to `false`.
"""
function init_dist_invlink!(vi::UntypedVarInfo, spl::AbstractSampler)
    vns = getvns(vi, spl)
    for vn in vns
        dist = getinitdist(vi, vn)
        link(vi)[vn, dist]
    end
    return vi
end
function init_dist_invlink!(vi::TypedVarInfo, spl::AbstractSampler)
    vns = getvns(vi, spl)
    _init_dist_invlink!(vi.metadata, vi, vns, Val(getspace(spl)))
    return vi
end
@generated function _init_dist_invlink!(metadata::NamedTuple{names}, vi, vns, ::Val{space}) where {names, space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(expr.args, quote
                f_vns = vi.metadata.$f.vns
                # Iterate over all `f_vns` and transform
                for vn in f_vns
                    dist = getinitdist(vi, vn)
                    link(vi)[vn, dist]
                end
            end)
        end
    end
    return expr
end

# X -> R for all variables associated with given sampler
"""
    settrans!(vi::VarInfo, spl::Sampler)

Set the `"trans"` flag to `true` for all the vaiables in the space of `spl`.
"""
function settrans!(vi::UntypedVarInfo, spl::Sampler)
    # TODO: Change to a lazy iterator over `vns`
    vns = getvns(vi, spl)
    if ~istrans(vi, vns[1])
        for vn in vns
            settrans!(vi, true, vn)
        end
    end
    return vi
end
function settrans!(vi::TypedVarInfo, spl::AbstractSampler)
    vns = getvns(vi, spl)
    _settrans!(vi, vns, Val(getspace(spl)))
    return vi
end
@generated function _settrans!(vi, ::NamedTuple{names}, ::Val{space}) where {names, space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(expr.args, quote
                f_vns = vi.metadata.$f.vns
                if length(f_vns) > 0 && ~istrans(vi, f_vns[1])
                    # Iterate over all `f_vns` and transform
                    for vn in f_vns
                        settrans!(vi, true, vn)
                    end
                end
            end)
        end
    end
    return expr
end

"""
    settrans!(vi::VarInfo, trans::Bool, vn::VarName)
Set the `trans` flag value of `vn` in `vi`.
"""
function settrans!(vi::AbstractVarInfo, trans::Bool, vn::VarName)
    trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")
end
