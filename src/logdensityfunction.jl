import DifferentiationInterface as DI

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
    "context used for evaluation; if `nothing`, `leafcontext(model.context)` will be used when applicable"
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
function getcontext(f::LogDensityFunction)
    return f.context === nothing ? leafcontext(f.model.context) : f.context
end

"""
    getmodel(f)

Return the `DynamicPPL.Model` wrapped in the given log-density function `f`.
"""
getmodel(f::DynamicPPL.LogDensityFunction) = f.model

"""
    setmodel(f, model[, adtype])

Set the `DynamicPPL.Model` in the given log-density function `f` to `model`.
"""
function setmodel(f::DynamicPPL.LogDensityFunction, model::DynamicPPL.Model)
    return Accessors.@set f.model = model
end

# HACK: heavy usage of `AbstractSampler` for, well, _everything_, is being phased out. In the mean time
# we need to define these annoying methods to ensure that we stay compatible with everything.
getsampler(f::LogDensityFunction) = getsampler(getcontext(f))
hassampler(f::LogDensityFunction) = hassampler(getcontext(f))

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[:]

# LogDensityProblems interface
function LogDensityProblems.logdensity(f::LogDensityFunction, θ::AbstractVector)
    context = getcontext(f)
    vi_new = unflatten(f.varinfo, θ)
    return getlogp(last(evaluate!!(f.model, vi_new, context)))
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end
# TODO: should we instead implement and call on `length(f.varinfo)` (at least in the cases where no sampler is involved)?
LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))

_flipped_logdensity(θ, f) = LogDensityProblems.logdensity(f, θ)

# By default, the AD backend to use is inferred from the context, which would
# typically be a SamplingContext which contains a sampler.
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction, θ::AbstractVector
)
    adtype = getadtype(getsampler(getcontext(f)))
    return LogDensityProblems.logdensity_and_gradient(f, θ, adtype)
end

# Extra method allowing one to manually specify the AD backend to use, thus
# overriding the default AD backend inferred from the sampler.
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction, θ::AbstractVector, adtype::ADTypes.AbstractADType
)
    # Ensure we concretise the elements of the params.
    θ = map(identity, θ) # TODO: Is this needed?
    prep = DI.prepare_gradient(_flipped_logdensity, adtype, θ, DI.Constant(f))
    return DI.value_and_gradient(_flipped_logdensity, prep, adtype, θ, DI.Constant(f))
end
