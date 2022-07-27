struct LazyTransformationContext{isinverse} <: AbstractContext end
NodeTrait(::LazyTransformationContext) = IsLeaf()

function tilde_assume(
    ::LazyTransformationContext{isinverse}, right, vn, vi
) where {isinverse}
    r = vi[vn, right]
    lp = Bijectors.logpdf_with_trans(right, r, !isinverse)

    if istrans(vi, vn)
        @assert isinverse "Trying to link already transformed variables"
    else
        @assert !isinverse "Trying to invlink non-transformed variables"
    end

    # Only transform if `!isinverse` since `vi[vn, right]`
    # already performs the inverse transformation if it's transformed.
    r_transformed = isinverse ? r : bijector(right)(r)
    return r, lp, setindex!!(vi, r_transformed, vn)
end

function dot_tilde_assume(
    ::LazyTransformationContext{isinverse},
    dist::Distribution,
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    vi,
) where {isinverse}
    r = getindex.((vi,), vns, (dist,))
    b = bijector(dist)

    is_trans_uniques = unique(istrans.((vi,), vns))
    @assert length(is_trans_uniques) == 1 "LazyTransformationContext only supports transforming all variables"
    is_trans = first(is_trans_uniques)
    if is_trans
        @assert isinverse "Trying to link already transformed variables"
    else
        @assert !isinverse "Trying to invlink non-transformed variables"
    end

    # Only transform if `!isinverse` since `vi[vn, right]`
    # already performs the inverse transformation if it's transformed.
    r_transformed = isinverse ? r : b.(r)
    lp = sum(Bijectors.logpdf_with_trans.((dist,), r, (!isinverse,)))
    return r, lp, setindex!!(vi, r_transformed, vns)
end

function dot_tilde_assume(
    ::LazyTransformationContext{isinverse},
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    vi::AbstractVarInfo,
) where {isinverse}
    @assert length(dist) == size(var, 1) "dimensionality of `var` ($(size(var, 1))) is incompatible with dimensionality of `dist` $(length(dist))"
    r = vi[vns, dist]

    # Compute `logpdf` with logabsdet-jacobian correction.
    lp = sum(zip(vns, eachcol(r))) do (vn, ri)
        return Bijectors.logpdf_with_trans(dist, ri, !isinverse)
    end

    # Transform _all_ values.
    is_trans_uniques = unique(istrans.((vi,), vns))
    @assert length(is_trans_uniques) == 1 "LazyTransformationContext only supports transforming all variables"
    is_trans = first(is_trans_uniques)
    if is_trans
        @assert isinverse "Trying to link already transformed variables"
    else
        @assert !isinverse "Trying to invlink non-transformed variables"
    end

    b = bijector(dist)
    for (vn, ri) in zip(vns, eachcol(r))
        # Only transform if `!isinverse` since `vi[vn, right]`
        # already performs the inverse transformation if it's transformed.
        vi = DynamicPPL.setindex!!(vi, isinverse ? ri : b(ri), vn)
    end

    return r, lp, vi
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
    return settrans!!(last(evaluate!!(model, vi, LazyTransformationContext{false}())), t)
end
function link!!(t::DefaultTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    link!(vi, spl)
    return vi
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
    return settrans!!(
        last(evaluate!!(model, vi, LazyTransformationContext{true}())), NoTransformation()
    )
end
function invlink!!(::DefaultTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    invlink!(vi, spl)
    return vi
end

# BijectorTransformation
function link!!(
    t::BijectorTransformation{<:Bijectors.Bijector{1}},
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    model::Model,
)
    b = t.bijector
    x = vi[spl]
    y, logjac = with_logabsdet_jacobian(b, x)

    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, y), lp_new)
    return settrans!!(vi_new, t)
end

function invlink!!(
    t::BijectorTransformation{<:Bijectors.Bijector{1}},
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    model::Model,
)
    b = t.bijector
    ib = inverse(b)
    y = vi[spl]
    x, logjac = with_logabsdet_jacobian(ib, y)

    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, x), lp_new)
    return settrans!!(vi_new, NoTransformation())
end
