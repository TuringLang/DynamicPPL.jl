"""
    LogDensityFunction

A callable representing a log density function of a `model`.

# Fields
$(FIELDS)

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: LogDensityFunction, contextualize

julia> @model function demo(x)
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> model = demo(1.0);

julia> f = LogDensityFunction(model);

julia> # It implements the interface of LogDensityProblems.jl.
       using LogDensityProblems

julia> LogDensityProblems.logdensity(f, [0.0])
-2.3378770664093453

julia> LogDensityProblems.dimension(f)
1

julia> # By default it uses `VarInfo` under the hood, but this is not necessary.
       f = LogDensityFunction(model, SimpleVarInfo(model));

julia> LogDensityProblems.logdensity(f, [0.0])
-2.3378770664093453

julia> # This also respects the context in `model`.
       f_prior = LogDensityFunction(contextualize(model, DynamicPPL.PriorContext()), VarInfo(model));

julia> LogDensityProblems.logdensity(f_prior, [0.0]) == logpdf(Normal(), 0.0)
true
```
"""
struct LogDensityFunction{V,M,C}
    "varinfo used for evaluation"
    varinfo::V
    "model used for evaluation"
    model::M
    "context used for evaluation; if `nothing`, `model.context` will be used when applicable"
    context::C
end

# TODO: Deprecate.
function LogDensityFunction(
    varinfo::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler,
    context::AbstractContext,
)
    return LogDensityFunction(varinfo, model, SamplingContext(sampler, context))
end

function LogDensityFunction(
    model::Model,
    varinfo::AbstractVarInfo=VarInfo(model),
    context::Union{Nothing,AbstractContext}=nothing,
)
    return LogDensityFunction(varinfo, model, context)
end

# If a `context` has been specified, we use that. Otherwise we just use the leaf context of `model`.
getcontext(f::LogDensityFunction) = f.context === nothing ? leafcontext(f.model) : f.context

# HACK: heavy usage of `AbstractSampler` for, well, _everything_, is being phased out. In the mean time
# we need to define these annoying methods to ensure that we stay compatible with everything.
getsampler(f::LogDensityFunction) = getsampler(f.context)
hassampler(f::LogDensityFunction) = hassampler(f.context)

_get_indexer(ctx::AbstractContext) = _get_indexer(NodeTrait(ctx), ctx)
_get_indexer(ctx::SamplingContext) = ctx.sampler
_get_indexer(::IsParent, ctx::AbstractContext) = _get_indexer(childcontext(ctx))
_get_indexer(::IsLeaf, ctx::AbstractContext) = Colon()

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[_get_indexer(f.context)]

# LogDensityProblems interface
function LogDensityProblems.logdensity(f::LogDensityFunction, θ::AbstractVector)
    context = getcontext(f)
    vi_new = unflatten(f.varinfo, context, θ)
    return getlogp(last(evaluate!!(f.model, vi_new, context)))
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end
# TODO: should we instead implement and call on `length(f.varinfo)` (at least in the cases where no sampler is involved)?
LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))
