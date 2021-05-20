@generated function _bijector(md::NamedTuple{names}; tuplify=false) where {names}
    expr = Expr(:tuple)
    for n in names
        e = quote
            if length(md.$n.dists) == 1 &&
               md.$n.dists[1] isa $(Distributions.UnivariateDistribution)
                $(Bijectors).bijector(md.$n.dists[1])
            elseif tuplify
                $(Bijectors.Stacked)(
                    map($(Bijectors).bijector, tuple(md.$n.dists...)), md.$n.ranges
                )
            else
                $(Bijectors.Stacked)(map($(Bijectors).bijector, md.$n.dists), md.$n.ranges)
            end
        end
        push!(expr.args, e)
    end

    return quote
        bs = NamedTuple{$names}($expr)
        return $(Bijectors).NamedBijector(bs)
    end
end

"""
    bijector(varinfo::VarInfo; tuplify=false)

Returns a `NamedBijector` which can transform different variants of `varinfo`.

If `tuplify` is true, then a type-stable bijector will be returned.
"""
Bijectors.bijector(vi::TypedVarInfo; kwargs...) = _bijector(vi.metadata; kwargs...)
