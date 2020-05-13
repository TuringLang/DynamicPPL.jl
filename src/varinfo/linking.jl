# X -> R for all variables associated with given sampler
"""
    link!(vi::VarInfo, spl::Sampler)

Transform the values of the random variables sampled by `spl` in `vi` from the support
of their distributions to the Euclidean space and set their corresponding `"trans"`
flag values to `true`.
"""
function link!(vi::UntypedVarInfo, spl::Sampler)
    # TODO: Change to a lazy iterator over `vns`
    vns = _getvns(vi, spl)
    if ~istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            # TODO: Use inplace versions to avoid allocations
            setval!(vi, vectorize(dist, Bijectors.link(dist, reconstruct(dist, getval(vi, vn)))), vn)
            settrans!(vi, true, vn)
        end
    else
        @warn("[DynamicPPL] attempt to link a linked vi")
    end
end
function link!(vi::TypedVarInfo, spl::AbstractSampler)
    vns = _getvns(vi, spl)
    return _link!(vi.metadata, vi, vns, Val(getspace(spl)))
end
@generated function _link!(metadata::NamedTuple{names}, vi, vns, ::Val{space}) where {names, space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(expr.args, quote
                f_vns = vi.metadata.$f.vns
                if ~istrans(vi, f_vns[1])
                    # Iterate over all `f_vns` and transform
                    for vn in f_vns
                        dist = getdist(vi, vn)
                        setval!(vi, vectorize(dist, Bijectors.link(dist, reconstruct(dist, getval(vi, vn)))), vn)
                        settrans!(vi, true, vn)
                    end
                else
                    @warn("[DynamicPPL] attempt to link a linked vi")
                end
            end)
        end
    end
    return expr
end

# R -> X for all variables associated with given sampler
"""
    invlink!(vi::VarInfo, spl::AbstractSampler)

Transform the values of the random variables sampled by `spl` in `vi` from the
Euclidean space back to the support of their distributions and sets their corresponding
`"trans"` flag values to `false`.
"""
function invlink!(vi::UntypedVarInfo, spl::AbstractSampler)
    vns = _getvns(vi, spl)
    if istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            setval!(vi, vectorize(dist, Bijectors.invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
            settrans!(vi, false, vn)
        end
    else
        @warn("[DynamicPPL] attempt to invlink an invlinked vi")
    end
end
function invlink!(vi::TypedVarInfo, spl::AbstractSampler)
    vns = _getvns(vi, spl)
    return _invlink!(vi.metadata, vi, vns, Val(getspace(spl)))
end
@generated function _invlink!(metadata::NamedTuple{names}, vi, vns, ::Val{space}) where {names, space}
    expr = Expr(:block)
    for f in names
        if inspace(f, space) || length(space) == 0
            push!(expr.args, quote
                f_vns = vi.metadata.$f.vns
                if istrans(vi, f_vns[1])
                    # Iterate over all `f_vns` and transform
                    for vn in f_vns
                        dist = getdist(vi, vn)
                        setval!(vi, vectorize(dist, Bijectors.invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
                        settrans!(vi, false, vn)
                    end
                else
                    @warn("[DynamicPPL] attempt to invlink an invlinked vi")
                end
            end)
        end
    end
    return expr
end


"""
    islinked(vi::VarInfo, spl::Sampler)

Check whether `vi` is in the transformed space for a particular sampler `spl`.

Turing's Hamiltonian samplers use the `link` and `invlink` functions from 
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) to map a constrained variable
(for example, one bounded to the space `[0, 1]`) from its constrained space to the set of 
real numbers. `islinked` checks if the number is in the constrained space or the real space.
"""
function islinked(vi::UntypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    return istrans(vi, vns[1])
end
function islinked(vi::TypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    return _islinked(vi, vns)
end
@generated function _islinked(vi, vns::NamedTuple{names}) where {names}
    out = []
    for f in names
        push!(out, :(length(vns.$f) == 0 ? false : istrans(vi, vns.$f[1])))
    end
    return Expr(:||, false, out...)
end

"""
    istrans(vi::VarInfo, vn::VarName)

Return true if `vn`'s values in `vi` are transformed to Euclidean space, and false if
they are in the support of `vn`'s distribution.
"""
istrans(vi::AbstractVarInfo, vn::VarName) = is_flagged(vi, vn, "trans")

"""
    settrans!(vi::VarInfo, trans::Bool, vn::VarName)

Set the `trans` flag value of `vn` in `vi`.
"""
function settrans!(vi::AbstractVarInfo, trans::Bool, vn::VarName)
    trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")
end
