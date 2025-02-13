import DifferentiationInterface as DI

"""
    LogDensityFunction

A callable representing a log density function of a `model`.
`DynamicPPL.LogDensityFunction` implements the LogDensityProblems.jl interface,
but only to 0th-order, i.e. it is only possible to calculate the log density,
and not its gradient. If you need to calculate the gradient as well, you have
to construct a [`DynamicPPL.LogDensityFunctionWithGrad`](@ref) object.

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

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[:]

# LogDensityProblems interface: logp (0th order)
function LogDensityProblems.logdensity(f::LogDensityFunction, x::AbstractVector)
    context = getcontext(f)
    vi_new = unflatten(f.varinfo, x)
    return getlogp(last(evaluate!!(f.model, vi_new, context)))
end
function _flipped_logdensity(x::AbstractVector, f::LogDensityFunction)
    return LogDensityProblems.logdensity(f, x)
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end
# TODO: should we instead implement and call on `length(f.varinfo)` (at least in the cases where no sampler is involved)?
LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))

# LogDensityProblems interface: gradient (1st order)
"""
    LogDensityFunctionWithGrad(ldf::DynamicPPL.LogDensityFunction, adtype::ADTypes.AbstractADType)

A callable representing a log density function of a `model`.
`DynamicPPL.LogDensityFunctionWithGrad` implements the LogDensityProblems.jl
interface to 1st-order, meaning that you can both calculate the log density
using

    LogDensityProblems.logdensity(f, x)

and its gradient using

    LogDensityProblems.logdensity_and_gradient(f, x)

where `f` is a `LogDensityFunctionWithGrad` object and `x` is a vector of parameters.

# Fields
$(FIELDS)
"""
struct LogDensityFunctionWithGrad{V,M,C,TAD<:ADTypes.AbstractADType}
    ldf::LogDensityFunction{V,M,C}
    adtype::TAD
    prep::DI.GradientPrep

    function LogDensityFunctionWithGrad(
        ldf::LogDensityFunction{V,M,C}, adtype::TAD
    ) where {V,M,C,TAD}
        # Get a set of dummy params to use for prep and concretise type
        x = map(identity, getparams(ldf))
        prep = DI.prepare_gradient(_flipped_logdensity, adtype, x, DI.Constant(ldf))
        # Store the prep with the struct
        return new{V,M,C,TAD}(ldf, adtype, prep)
    end
end
function LogDensityProblems.logdensity(f::LogDensityFunctionWithGrad)
    return LogDensityProblems.logdensity(f.ldf)
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunctionWithGrad})
    return LogDensityProblems.LogDensityOrder{1}()
end
# By default, the AD backend to use is inferred from the context, which would
# typically be a SamplingContext which contains a sampler.
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunctionWithGrad, x::AbstractVector
)
    x = map(identity, x)  # Concretise type
    return DI.value_and_gradient(
        _flipped_logdensity, f.prep, f.adtype, x, DI.Constant(f.ldf)
    )
end
