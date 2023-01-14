struct LogDensityFunction{V,M,C}
    varinfo::V
    model::M
    context::C
end

function LogDensityFunction(
    varinfo::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler,
    context::AbstractContext,
)
    return LogDensityFunction(varinfo, model, SamplingContext(sampler, context))
end

# Convenient for end-user.
function LogDensityFunction(
    model::Model,
    varinfo::AbstractVarInfo=VarInfo(model),
    context::AbstractContext=DefaultContext(),
)
    return LogDensityFunction(varinfo, model, context)
end

# HACK: heavy usage of `AbstractSampler` for, well, _everything_, is being phased out. In the mean time
# we need to define these annoying methods to ensure that we stay compatible with everything.
getsampler(f::LogDensityFunction) = getsampler(f.context)
hassampler(f::LogDensityFunction) = hassampler(f.context)

# Evaluator.
function (f::LogDensityFunction)(θ::AbstractVector)
    vi_new = unflatten(f.varinfo, f.context, θ)
    return getlogp(last(evaluate!!(f.model, vi_new, f.context)))
end

# LogDensityProblems interface
LogDensityProblems.logdensity(f::LogDensityFunction, θ) = f(θ)
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end

_get_indexer(ctx::AbstractContext) = _get_indexer(NodeTrait(ctx), ctx)
_get_indexer(ctx::SamplingContext) = ctx.sampler
_get_indexer(::IsParent, ctx::AbstractContext) = _get_indexer(childcontext(ctx))
_get_indexer(::IsLeaf, ctx::AbstractContext) = Colon()

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[_get_indexer(f.context)]

LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))
