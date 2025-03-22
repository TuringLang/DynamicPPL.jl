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
        varinfo::AbstractVarInfo=VarInfo(model),
        context::AbstractContext=DefaultContext()
    )

A struct which contains a model, along with all the information necessary to
calculate its log density at a given point.

At its most basic level, a LogDensityFunction wraps the model together with its
the type of varinfo to be used, as well as the evaluation context. These must
be known in order to calculate the log density (using
[`DynamicPPL.evaluate!!`](@ref)).

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

julia> # If we also need to calculate the gradient, an AD backend must be specified as part of the model.
       import ForwardDiff, ADTypes

julia> model_with_ad = Model(model, ADTypes.AutoForwardDiff());

julia> f = LogDensityFunction(model_with_ad);

julia> LogDensityProblems.logdensity_and_gradient(f, [0.0])
(-2.3378770664093453, [1.0])
```
"""
struct LogDensityFunction{M<:Model,V<:AbstractVarInfo,C<:AbstractContext}
    "model used for evaluation"
    model::M
    "varinfo used for evaluation"
    varinfo::V
    "context used for evaluation; if `nothing`, `leafcontext(model.context)` will be used when applicable"
    context::C
    "(internal use only) gradient preparation object for the model"
    prep::Union{Nothing,DI.GradientPrep}

    function LogDensityFunction(
        model::Model,
        varinfo::AbstractVarInfo=VarInfo(model),
        context::AbstractContext=leafcontext(model.context),
    )
        adtype = model.adtype
        if adtype === nothing
            prep = nothing
        else
            # Make backend-specific tweaks to the adtype
            # This should arguably be done in the model constructor, but it needs the
            # varinfo and context to do so, and it seems excessive to construct a
            # varinfo at the point of calling Model().
            adtype = tweak_adtype(adtype, model, varinfo, context)
            model = Model(model, adtype)
            # Check whether it is supported
            is_supported(adtype) ||
                @warn "The AD backend $adtype is not officially supported by DynamicPPL. Gradient calculations may still work, but compatibility is not guaranteed."
            # Get a set of dummy params to use for prep
            x = map(identity, varinfo[:])
            if use_closure(adtype)
                prep = DI.prepare_gradient(
                    x -> logdensity_at(x, model, varinfo, context), adtype, x
                )
            else
                prep = DI.prepare_gradient(
                    logdensity_at,
                    adtype,
                    x,
                    DI.Constant(model),
                    DI.Constant(varinfo),
                    DI.Constant(context),
                )
            end
        end
        return new{typeof(model),typeof(varinfo),typeof(context)}(
            model, varinfo, context, prep
        )
    end
end

"""
    LogDensityFunction(
        ldf::LogDensityFunction,
        adtype::Union{Nothing,ADTypes.AbstractADType}
    )

Create a new LogDensityFunction using the model, varinfo, and context from the given
`ldf` argument, but with the AD type set to `adtype`. To remove the AD type, pass
`nothing` as the second argument.
"""
function LogDensityFunction(
    f::LogDensityFunction, adtype::Union{Nothing,ADTypes.AbstractADType}
)
    return if adtype === f.model.adtype
        f  # Avoid recomputing prep if not needed
    else
        LogDensityFunction(Model(f.model, adtype), f.varinfo, f.context)
    end
end

"""
    logdensity_at(
        x::AbstractVector,
        model::Model,
        varinfo::AbstractVarInfo,
        context::AbstractContext
    )

Evaluate the log density of the given `model` at the given parameter values `x`,
using the given `varinfo` and `context`. Note that the `varinfo` argument is provided
only for its structure, in the sense that the parameters from the vector `x` are inserted into
it, and its own parameters are discarded. 
"""
function logdensity_at(
    x::AbstractVector, model::Model, varinfo::AbstractVarInfo, context::AbstractContext
)
    varinfo_new = unflatten(varinfo, x)
    return getlogp(last(evaluate!!(model, varinfo_new, context)))
end

### LogDensityProblems interface

function LogDensityProblems.capabilities(
    ::Type{
        <:LogDensityFunction{
            Model{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,Nothing},V,C
        },
    },
) where {F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,V,C}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(
    ::Type{
        <:LogDensityFunction{
            Model{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,TAD},V,C
        },
    },
) where {
    F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,V,C,TAD<:ADTypes.AbstractADType
}
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.logdensity(f::LogDensityFunction, x::AbstractVector)
    return logdensity_at(x, f.model, f.varinfo, f.context)
end
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction{M,V,C}, x::AbstractVector
) where {M,V,C}
    f.prep === nothing && error(
        "Attempted to call logdensity_and_gradient on a LogDensityFunction without an AD backend. You need to set an AD backend in the model before calculating the gradient of logp.",
    )
    x = map(identity, x)  # Concretise type
    # Make branching statically inferrable, i.e. type-stable (even if the two
    # branches happen to return different types)
    return if use_closure(f.model.adtype)
        DI.value_and_gradient(
            x -> logdensity_at(x, f.model, f.varinfo, f.context), f.prep, f.model.adtype, x
        )
    else
        DI.value_and_gradient(
            logdensity_at,
            f.prep,
            f.model.adtype,
            x,
            DI.Constant(f.model),
            DI.Constant(f.varinfo),
            DI.Constant(f.context),
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
        context::AbstractContext
    )

Return an 'optimised' form of the adtype. This is useful for doing
backend-specific optimisation of the adtype (e.g., for ForwardDiff, calculating
the chunk size: see the method override in `ext/DynamicPPLForwardDiffExt.jl`).
The model is passed as a parameter in case the optimisation depends on the
model.

By default, this just returns the input unchanged.
"""
tweak_adtype(
    adtype::ADTypes.AbstractADType, ::Model, ::AbstractVarInfo, ::AbstractContext
) = adtype

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
    getmodel(f)

Return the `DynamicPPL.Model` wrapped in the given log-density function `f`.
"""
getmodel(f::DynamicPPL.LogDensityFunction) = f.model

"""
    setmodel(f, model[, adtype])

Set the `DynamicPPL.Model` in the given log-density function `f` to `model`.
"""
function setmodel(f::DynamicPPL.LogDensityFunction, model::DynamicPPL.Model)
    return LogDensityFunction(model, f.varinfo, f.context)
end

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[:]
