struct DynamicTransformationContext{isinverse} <: AbstractContext end
NodeTrait(::DynamicTransformationContext) = IsLeaf()

function tilde_assume(
    ::DynamicTransformationContext{isinverse}, right, vn, vi
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
    ::DynamicTransformationContext{isinverse},
    dist::Distribution,
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    vi,
) where {isinverse}
    r = getindex.((vi,), vns, (dist,))
    b = bijector(dist)

    is_trans_uniques = unique(istrans.((vi,), vns))
    @assert length(is_trans_uniques) == 1 "DynamicTransformationContext only supports transforming all variables"
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
    ::DynamicTransformationContext{isinverse},
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
    @assert length(is_trans_uniques) == 1 "DynamicTransformationContext only supports transforming all variables"
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

function link!!(
    t::DynamicTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    return settrans!!(last(evaluate!!(model, vi, DynamicTransformationContext{false}())), t)
end

function invlink!!(
    ::DynamicTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    return settrans!!(
        last(evaluate!!(model, vi, DynamicTransformationContext{true}())),
        NoTransformation(),
    )
end
