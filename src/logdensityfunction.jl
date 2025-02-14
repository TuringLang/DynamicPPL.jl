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
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end
# TODO: should we instead implement and call on `length(f.varinfo)` (at least in the cases where no sampler is involved)?
LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))

# LogDensityProblems interface: gradient (1st order)
"""
    use_closure(adtype::ADTypes.AbstractADType)

In LogDensityProblems, we want to calculate the derivative of logdensity(f, x)
with respect to x, where f is the model (in our case LogDensityFunction) and is
a constant. However, DifferentiationInterface generally expects a
single-argument function g(x) to differentiate.

There are two ways of dealing with this:

1. Construct a closure over the model, i.e. let g = Base.Fix1(logdensity, f)

2. Use a constant context. This lets us pass a two-argument function to
   DifferentiationInterface, as long as we also give it the 'inactive argument'
   (i.e. the model) wrapped in `DI.Constant`.

The relative performance of the two approaches, however, depends on the AD
backend used. Some benchmarks are provided here:
https://github.com/TuringLang/DynamicPPL.jl/pull/806#issuecomment-2658061480

This function is used to determine whether a given AD backend should use a
closure or a constant. If `use_closure(adtype)` returns `true`, then the
closure approach will be used. By default, this function returns `false`, i.e.
the constant approach will be used.
"""
use_closure(::ADTypes.AbstractADType) = false
use_closure(::ADTypes.AutoForwardDiff) = false
use_closure(::ADTypes.AutoMooncake) = false
use_closure(::ADTypes.AutoReverseDiff) = true

"""
    _flipped_logdensity(f::LogDensityFunction, x::AbstractVector)

This function is the same as `LogDensityProblems.logdensity(f, x)` but with the
arguments flipped. It is used in the 'constant' approach to DifferentiationInterface
(see `use_closure` for more information).
"""
function _flipped_logdensity(x::AbstractVector, f::LogDensityFunction)
    return LogDensityProblems.logdensity(f, x)
end

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
    with_closure::Bool

    function LogDensityFunctionWithGrad(
        ldf::LogDensityFunction{V,M,C}, adtype::TAD
    ) where {V,M,C,TAD}
        # Get a set of dummy params to use for prep
        x = map(identity, getparams(ldf))
        with_closure = use_closure(adtype)
        if with_closure
            prep = DI.prepare_gradient(
                Base.Fix1(LogDensityProblems.logdensity, ldf), adtype, x
            )
        else
            prep = DI.prepare_gradient(_flipped_logdensity, adtype, x, DI.Constant(ldf))
        end
        # Store the prep with the struct. We also store whether a closure was used because
        # we need to know this when calling `DI.value_and_gradient`. In practice we could
        # recalculate it, but this runs the risk of introducing inconsistencies.
        return new{V,M,C,TAD}(ldf, adtype, prep, with_closure)
    end
end
function LogDensityProblems.logdensity(f::LogDensityFunctionWithGrad)
    return LogDensityProblems.logdensity(f.ldf)
end
function LogDensityProblems.capabilities(::Type{<:LogDensityFunctionWithGrad})
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunctionWithGrad, x::AbstractVector
)
    x = map(identity, x)  # Concretise type
    return if f.with_closure
        DI.value_and_gradient(
            Base.Fix1(LogDensityProblems.logdensity, f.ldf), f.prep, f.adtype, x
        )
    else
        DI.value_and_gradient(_flipped_logdensity, f.prep, f.adtype, x, DI.Constant(f.ldf))
    end
end
