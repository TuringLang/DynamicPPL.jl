function Bijectors.Stacked(
    model::Model, ::Val{sym2ranges}=Val(false); varinfo::VarInfo=VarInfo(model)
) where {sym2ranges}
    dists = vcat([varinfo.metadata[sym].dists for sym in keys(varinfo.metadata)]...)

    num_ranges = sum([
        length(varinfo.metadata[sym].ranges) for sym in keys(varinfo.metadata)
    ])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1

    # ranges might be discontinuous => values are vectors of ranges rather than just ranges
    sym_lookup = Dict{Symbol,Vector{UnitRange{Int}}}()
    for sym in keys(varinfo.metadata)
        sym_lookup[sym] = Vector{UnitRange{Int}}()
        for r in varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            push!(sym_lookup[sym], ranges[range_idx])
            range_idx += 1
        end

        idx += varinfo.metadata[sym].ranges[end][end]
    end

    b = Bijectors.Stacked(map(Bijectors.bijector, dists), ranges)
    return sym2ranges ? (b, Dict(zip(keys(sym_lookup), values(sym_lookup)))) : b
end

link!!(vi::AbstractVarInfo, model::Model) = link!!(vi, SampleFromPrior(), model)
function link!!(t::AbstractTransformation, vi::AbstractVarInfo, model::Model)
    return link!!(t, vi, SampleFromPrior(), model)
end
function link!!(vi::AbstractVarInfo, spl::AbstractSampler, model::Model)
    # Use `default_transformation` to decide which transformation to use if none is specified.
    return link!!(default_transformation(model, vi), vi, spl, model)
end
function link!!(
    t::DefaultTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    # TODO: Implement this properly, e.g. using a context or something.
    # Fall back to `Bijectors.Stacked` but then we act like we're using
    # the `DefaultTransformation` by setting the transformation accordingly.
    return settrans!!(
        link!!(BijectorTransformation(Bijectors.Stacked(model)), vi, spl, model), t
    )
end
function link!!(t::DefaultTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    # TODO: Implement this properly, e.g. using a context or something.
    DynamicPPL.link!(vi, spl)
    # TODO: Add `logabsdet_jacobian` correction to `logp`!
    return vi
end
function link!!(
    t::BijectorTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    b = t.bijector
    x = vi[spl]
    y, logjac = with_logabsdet_jacobian(b, x)

    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, y), lp_new)
    return settrans!!(vi_new, t)
end

invlink!!(vi::AbstractVarInfo, model::Model) = invlink!!(vi, SampleFromPrior(), model)
function invlink!!(t::AbstractTransformation, vi::AbstractVarInfo, model::Model)
    return invlink!!(t, vi, SampleFromPrior(), model)
end
function invlink!!(vi::AbstractVarInfo, spl::AbstractSampler, model::Model)
    # Here we extract the `transformation` from `vi` rather than using the default one.
    return invlink!!(transformation(vi), vi, spl, model)
end
function invlink!!(
    ::DefaultTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    # TODO: Implement this properly, e.g. using a context or something.
    return invlink!!(BijectorTransformation(Bijectors.Stacked(model)), vi, spl, model)
end
function invlink!!(::DefaultTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    # TODO: Implement this properly, e.g. using a context or something.
    DynamicPPL.invlink!(vi, spl)
    return vi
end
function invlink!!(
    t::BijectorTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    b = t.bijector
    ib = inverse(b)
    y = vi[spl]
    x, logjac = with_logabsdet_jacobian(ib, y)
    # TODO: Do we need this?
    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, x), lp_new)
    return settrans!!(vi_new, NoTransformation())
end
