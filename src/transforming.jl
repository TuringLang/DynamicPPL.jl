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
    t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model
)
    return link!!(t, vi.varinfo, spl, model)
end
function link!!(
    t::LazyTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    return settrans!!(last(evaluate!!(model, vi, LazyTransformationContext{false}())), t)
end
function link!!(t::LazyTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    # Call `_link!` instead of `link!` to avoid deprecation warning.
    _link!(vi, spl)
    return vi
end

"""
    maybe_invlink_before_eval!!([t::Transformation,] vi, context, model)

Return a possibly invlinked version of `vi`.

This will be called prior to `model` evaluation, allowing one to perform a single
`invlink!!` _before_ evaluation rather lazyily evaluate the transforms on as-we-need
basis as is done with [`LazyTransformation` ](@ref).

# Examples
```julia-repl
julia> using DynamicPPL, Distributions, Bijectors

julia> @model demo() = x ~ Normal()
demo (generic function with 2 methods)

julia> # By subtyping `Bijector{1}`, we inherit the `(inv)link!!` defined for
       # bijectors which acts on 1-dimensional arrays, i.e. vectors.
       struct MyBijector <: Bijectors.Bijector{1} end

julia> # Define some dummy `inverse` which will be used in the `link!!` call.
       Bijectors.inverse(f::MyBijector) = identity

julia> # We need to define `with_logabsdet_jacobian` for `MyBijector`
       # (`identity` already has `with_logabsdet_jacobian` defined)
       function Bijectors.with_logabsdet_jacobian(::MyBijector, x)
           # Just using a large number of the logabsdet-jacobian term
           # for demonstration purposes.
           return (x, 1000)
       end

julia> # Change the `default_transformation` for our model to be a
       # `StaticTransformation` using `MyBijector`.
       function DynamicPPL.default_transformation(::Model{typeof(demo)})
           return DynamicPPL.StaticTransformation(MyBijector())
       end

julia> model = demo();

julia> vi = SimpleVarInfo(x=1.0)
SimpleVarInfo((x = 1.0,), 0.0)

julia> # Uses the `inverse` of `MyBijector`, which we have defined as `identity`
       vi_linked = link!!(vi, model)
Transformed SimpleVarInfo((x = 1.0,), 0.0)

julia> # Now performs a single `invlink!!` before model evaluation.
       logjoint(model, vi_linked)
-1001.4189385332047
```
"""
function maybe_invlink_before_eval!!(
    vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    return maybe_invlink_before_eval!!(transformation(vi), vi, context, model)
end
function maybe_invlink_before_eval!!(
    t::AbstractTransformation, vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    # Default behavior is to _not_ transform.
    return vi
end
function maybe_invlink_before_eval!!(
    t::StaticTransformation, vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    return invlink!!(t, vi, _default_sampler(context), model)
end

function _default_sampler(context::AbstractContext)
    return _default_sampler(NodeTrait(_default_sampler, context), context)
end
_default_sampler(::IsLeaf, context::AbstractContext) = SampleFromPrior()
function _default_sampler(::IsParent, context::AbstractContext)
    return _default_sampler(childcontext(context))
end

invlink!!(vi::AbstractVarInfo, model::Model) = invlink!!(vi, SampleFromPrior(), model)
function invlink!!(t::AbstractTransformation, vi::AbstractVarInfo, model::Model)
    return invlink!!(t, vi, SampleFromPrior(), model)
end
function invlink!!(vi::AbstractVarInfo, spl::AbstractSampler, model::Model)
    # Here we extract the `transformation` from `vi` rather than using the default one.
    return invlink!!(transformation(vi), vi, spl, model)
end
function invlink!!(t::AbstractTransformation, vi::ThreadSafeVarInfo, spl::AbstractSampler, model::Model)
    return invlink!!(t, vi.varinfo, spl, model)
end
function invlink!!(
    ::LazyTransformation, vi::AbstractVarInfo, spl::AbstractSampler, model::Model
)
    return settrans!!(
        last(evaluate!!(model, vi, LazyTransformationContext{true}())), NoTransformation()
    )
end
function invlink!!(::LazyTransformation, vi::VarInfo, spl::AbstractSampler, model::Model)
    # Call `_invlink!` instead of `invlink!` to avoid deprecation warning.
    _invlink!(vi, spl)
    return vi
end

# Vector-based ones.
function link!!(
    t::StaticTransformation{<:Bijectors.Bijector{1}},
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    model::Model,
)
    b = inverse(t.bijector)
    x = vi[spl]
    y, logjac = with_logabsdet_jacobian(b, x)

    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, y), lp_new)
    return settrans!!(vi_new, t)
end

function invlink!!(
    t::StaticTransformation{<:Bijectors.Bijector{1}},
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    model::Model,
)
    b = t.bijector
    y = vi[spl]
    x, logjac = with_logabsdet_jacobian(b, y)

    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, x), lp_new)
    return settrans!!(vi_new, NoTransformation())
end
