using AbstractMCMC: AbstractModel
import DifferentiationInterface as DI

"""
    is_supported(adtype::AbstractADType)

Check if the given AD type is formally supported by DynamicPPL.

AD backends that are not formally supported can still be used for gradient
calculation; it is just that the DynamicPPL developers do not commit to
maintaining compatibility with them.
"""
is_supported(::ADTypes.AbstractADType) = false
is_supported(::ADTypes.AutoForwardDiff) = true
is_supported(::ADTypes.AutoMooncake) = true
is_supported(::ADTypes.AutoReverseDiff) = true

"""
    LogDensityFunction(
        model::Model,
        getlogdensity::Function=getlogjoint,
        varinfo::AbstractVarInfo=ldf_default_varinfo(model, getlogdensity),
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing
    )

A struct which contains a model, along with all the information necessary to:

 - calculate its log density at a given point;
 - and if `adtype` is provided, calculate the gradient of the log density at
 that point.

At its most basic level, a LogDensityFunction wraps the model together with a
function that specifies how to extract the log density, and the type of 
VarInfo to be used. These must be known in order to calculate the log density
(using [`DynamicPPL.evaluate!!`](@ref)).

If the `adtype` keyword argument is provided, then this struct will also store
the adtype along with other information for efficient calculation of the
gradient of the log density. Note that preparing a `LogDensityFunction` with an
AD type `AutoBackend()` requires the AD backend itself to have been loaded
(e.g. with `import Backend`).

`DynamicPPL.LogDensityFunction` implements the LogDensityProblems.jl interface.
If `adtype` is nothing, then only `logdensity` is implemented. If `adtype` is a
concrete AD backend type, then `logdensity_and_gradient` is also implemented.

# Fields
$(FIELDS)

# Examples

```jldoctest
julia> using Distributions

julia> using DynamicPPL: LogDensityFunction, setaccs!!

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
       f = LogDensityFunction(model, getlogjoint, SimpleVarInfo(model));

julia> LogDensityProblems.logdensity(f, [0.0])
-2.3378770664093453

julia> # One can also specify evaluating e.g. the log prior only:
       f_prior = LogDensityFunction(model, getlogprior);

julia> LogDensityProblems.logdensity(f_prior, [0.0]) == logpdf(Normal(), 0.0)
true

julia> # If we also need to calculate the gradient, we can specify an AD backend.
       import ForwardDiff, ADTypes

julia> f = LogDensityFunction(model, adtype=ADTypes.AutoForwardDiff());

julia> LogDensityProblems.logdensity_and_gradient(f, [0.0])
(-2.3378770664093453, [1.0])
```
"""
struct LogDensityFunction{
    M<:Model,F<:Function,V<:AbstractVarInfo,AD<:Union{Nothing,ADTypes.AbstractADType}
} <: AbstractModel
    "model used for evaluation"
    model::M
    "function to be called on `varinfo` to extract the log density. By default `getlogjoint`."
    getlogdensity::F
    "varinfo used for evaluation. If not specified, generated with `ldf_default_varinfo`."
    varinfo::V
    "AD type used for evaluation of log density gradient. If `nothing`, no gradient can be calculated"
    adtype::AD
    "(internal use only) gradient preparation object for the model"
    prep::Union{Nothing,DI.GradientPrep}

    function LogDensityFunction(
        model::Model,
        getlogdensity::Function=getlogjoint,
        varinfo::AbstractVarInfo=ldf_default_varinfo(model, getlogdensity),
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        if adtype === nothing
            prep = nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = tweak_adtype(adtype, model, varinfo)
            # Check whether it is supported
            is_supported(adtype) ||
                @warn "The AD backend $adtype is not officially supported by DynamicPPL. Gradient calculations may still work, but compatibility is not guaranteed."
            # Get a set of dummy params to use for prep
            x = map(identity, varinfo[:])
            if use_closure(adtype)
                prep = DI.prepare_gradient(
                    LogDensityAt(model, getlogdensity, varinfo), adtype, x
                )
            else
                prep = DI.prepare_gradient(
                    logdensity_at,
                    adtype,
                    x,
                    DI.Constant(model),
                    DI.Constant(getlogdensity),
                    DI.Constant(varinfo),
                )
            end
        end
        return new{typeof(model),typeof(getlogdensity),typeof(varinfo),typeof(adtype)}(
            model, getlogdensity, varinfo, adtype, prep
        )
    end
end

"""
    LogDensityFunction(
        ldf::LogDensityFunction,
        adtype::Union{Nothing,ADTypes.AbstractADType}
    )

Create a new LogDensityFunction using the model and varinfo from the given
`ldf` argument, but with the AD type set to `adtype`. To remove the AD type,
pass `nothing` as the second argument.
"""
function LogDensityFunction(
    f::LogDensityFunction, adtype::Union{Nothing,ADTypes.AbstractADType}
)
    return if adtype === f.adtype
        f  # Avoid recomputing prep if not needed
    else
        LogDensityFunction(f.model, f.getlogdensity, f.varinfo; adtype=adtype)
    end
end

"""
    ldf_default_varinfo(model::Model, getlogdensity::Function)

Create the default AbstractVarInfo that should be used for evaluating the log density.

Only the accumulators necesessary for `getlogdensity` will be used.
"""
function ldf_default_varinfo(::Model, getlogdensity::Function)
    msg = """
    LogDensityFunction does not know what sort of VarInfo should be used when \
    `getlogdensity` is $getlogdensity. Please specify a VarInfo explicitly.
    """
    return error(msg)
end

ldf_default_varinfo(model::Model, ::typeof(getlogjoint)) = VarInfo(model)

function ldf_default_varinfo(model::Model, ::typeof(getlogprior))
    return setaccs!!(VarInfo(model), (LogPriorAccumulator(),))
end

function ldf_default_varinfo(model::Model, ::typeof(getloglikelihood))
    return setaccs!!(VarInfo(model), (LogLikelihoodAccumulator(),))
end

"""
    logdensity_at(
        x::AbstractVector,
        model::Model,
        getlogdensity::Function,
        varinfo::AbstractVarInfo,
    )

Evaluate the log density of the given `model` at the given parameter values
`x`, using the given `varinfo`. Note that the `varinfo` argument is provided
only for its structure, in the sense that the parameters from the vector `x`
are inserted into it, and its own parameters are discarded. `getlogdensity` is
the function that extracts the log density from the evaluated varinfo.
"""
function logdensity_at(
    x::AbstractVector, model::Model, getlogdensity::Function, varinfo::AbstractVarInfo
)
    varinfo_new = unflatten(varinfo, x)
    varinfo_eval = last(evaluate!!(model, varinfo_new))
    return getlogdensity(varinfo_eval)
end

"""
    LogDensityAt{M<:Model,F<:Function,V<:AbstractVarInfo}(
        model::M
        getlogdensity::F,
        varinfo::V
    )

A callable struct that serves the same purpose as `x -> logdensity_at(x, model,
getlogdensity, varinfo)`.
"""
struct LogDensityAt{M<:Model,F<:Function,V<:AbstractVarInfo}
    model::M
    getlogdensity::F
    varinfo::V
end
function (ld::LogDensityAt)(x::AbstractVector)
    return logdensity_at(x, ld.model, ld.getlogdensity, ld.varinfo)
end

### LogDensityProblems interface

function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{M,F,V,Nothing}}
) where {M,F,V}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{M,F,V,AD}}
) where {M,F,V,AD<:ADTypes.AbstractADType}
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.logdensity(f::LogDensityFunction, x::AbstractVector)
    return logdensity_at(x, f.model, f.getlogdensity, f.varinfo)
end
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction{M,F,V,AD}, x::AbstractVector
) where {M,F,V,AD<:ADTypes.AbstractADType}
    f.prep === nothing &&
        error("Gradient preparation not available; this should not happen")
    x = map(identity, x)  # Concretise type
    # Make branching statically inferrable, i.e. type-stable (even if the two
    # branches happen to return different types)
    return if use_closure(f.adtype)
        DI.value_and_gradient(
            LogDensityAt(x, f.model, f.getlogdensity, f.varinfo), f.prep, f.adtype, x
        )
    else
        DI.value_and_gradient(
            logdensity_at,
            f.prep,
            f.adtype,
            x,
            DI.Constant(f.model),
            DI.Constant(f.getlogdensity),
            DI.Constant(f.varinfo),
        )
    end
end

# TODO: should we instead implement and call on `length(f.varinfo)` (at least in the cases where no sampler is involved)?
LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))

### Utils

"""
    tweak_adtype(
        adtype::ADTypes.AbstractADType,
        model::Model,
        varinfo::AbstractVarInfo,
    )

Return an 'optimised' form of the adtype. This is useful for doing
backend-specific optimisation of the adtype (e.g., for ForwardDiff, calculating
the chunk size: see the method override in `ext/DynamicPPLForwardDiffExt.jl`).
The model is passed as a parameter in case the optimisation depends on the
model.

By default, this just returns the input unchanged.
"""
tweak_adtype(adtype::ADTypes.AbstractADType, ::Model, ::AbstractVarInfo) = adtype

"""
    use_closure(adtype::ADTypes.AbstractADType)

In LogDensityProblems, we want to calculate the derivative of logdensity(f, x)
with respect to x, where f is the model (in our case LogDensityFunction) and is
a constant. However, DifferentiationInterface generally expects a
single-argument function g(x) to differentiate.

There are two ways of dealing with this:

1. Construct a closure over the model, i.e. let g = Base.Fix1(logdensity, f)

2. Use a constant DI.Context. This lets us pass a two-argument function to DI,
   as long as we also give it the 'inactive argument' (i.e. the model) wrapped
   in `DI.Constant`.

The relative performance of the two approaches, however, depends on the AD
backend used. Some benchmarks are provided here:
https://github.com/TuringLang/DynamicPPL.jl/issues/946#issuecomment-2931604829

This function is used to determine whether a given AD backend should use a
closure or a constant. If `use_closure(adtype)` returns `true`, then the
closure approach will be used. By default, this function returns `false`, i.e.
the constant approach will be used.
"""
use_closure(::ADTypes.AbstractADType) = true

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
    return LogDensityFunction(model, f.getlogdensity, f.varinfo; adtype=f.adtype)
end

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[:]
