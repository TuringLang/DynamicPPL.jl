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
getmodel(f::LogDensityProblemsAD.ADGradientWrapper) =
    getmodel(LogDensityProblemsAD.parent(f))
getmodel(f::DynamicPPL.LogDensityFunction) = f.model

"""
    setmodel(f, model[, adtype])

Set the `DynamicPPL.Model` in the given log-density function `f` to `model`.

!!! warning
    Note that if `f` is a `LogDensityProblemsAD.ADGradientWrapper` wrapping a
    `DynamicPPL.LogDensityFunction`, performing an update of the `model` in `f`
    might require recompilation of the gradient tape, depending on the AD backend.
"""
function setmodel(
    f::LogDensityProblemsAD.ADGradientWrapper,
    model::DynamicPPL.Model,
    adtype::ADTypes.AbstractADType,
)
    # TODO: Should we handle `SciMLBase.NoAD`?
    # For an `ADGradientWrapper` we do the following:
    # 1. Update the `Model` in the underlying `LogDensityFunction`.
    # 2. Re-construct the `ADGradientWrapper` using `ADgradient` using the provided `adtype`
    #    to ensure that the recompilation of gradient tapes, etc. also occur. For example,
    #    ReverseDiff.jl in compiled mode will cache the compiled tape, which means that just
    #    replacing the corresponding field with the new model won't be sufficient to obtain
    #    the correct gradients.
    return LogDensityProblemsAD.ADgradient(
        adtype, setmodel(LogDensityProblemsAD.parent(f), model)
    )
end
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

# This is important for performance -- one needs to provide `ADGradient` with a vector of
# parameters, or DifferentiationInterface will not have sufficient information to e.g.
# compile a rule for Mooncake (because it won't know the type of the input), or pre-allocate
# a tape when using ReverseDiff.jl.
function _make_ad_gradient(ad::ADTypes.AbstractADType, ℓ::LogDensityFunction)
    x = map(identity, getparams(ℓ)) # ensure we concretise the elements of the params
    return LogDensityProblemsAD.ADgradient(ad, ℓ; x)
end

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoMooncake, f::LogDensityFunction)
    return _make_ad_gradient(ad, f)
end
function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoReverseDiff, f::LogDensityFunction)
    return _make_ad_gradient(ad, f)
end
