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
        context::AbstractContext=DefaultContext();
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing
    )

A struct which contains a model, along with all the information necessary to:

 - calculate its log density at a given point;
 - and if `adtype` is provided, calculate the gradient of the log density at
 that point.

At its most basic level, a LogDensityFunction wraps the model together with its
the type of varinfo to be used, as well as the evaluation context. These must
be known in order to calculate the log density (using
[`DynamicPPL.evaluate!!`](@ref)).

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

julia> # If we also need to calculate the gradient, we can specify an AD backend.
       import ForwardDiff, ADTypes

julia> f = LogDensityFunction(model, adtype=ADTypes.AutoForwardDiff());

julia> LogDensityProblems.logdensity_and_gradient(f, [0.0])
(-2.3378770664093453, [1.0])
```
"""
struct LogDensityFunction{
    M<:Model,V<:AbstractVarInfo,C<:AbstractContext,AD<:Union{Nothing,ADTypes.AbstractADType}
}
    "model used for evaluation"
    model::M
    "varinfo used for evaluation"
    varinfo::V
    "context used for evaluation; if `nothing`, `leafcontext(model.context)` will be used when applicable"
    context::C
    "AD type used for evaluation of log density gradient. If `nothing`, no gradient can be calculated"
    adtype::AD
    "(internal use only) gradient preparation object for the model"
    prep::Union{Nothing,DI.GradientPrep}

    function LogDensityFunction(
        model::Model,
        varinfo::AbstractVarInfo=VarInfo(model),
        context::AbstractContext=leafcontext(model.context);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        if adtype === nothing
            prep = nothing
        else
            # Check support
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
        return new{typeof(model),typeof(varinfo),typeof(context),typeof(adtype)}(
            model, varinfo, context, adtype, prep
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
    return if adtype === f.adtype
        f  # Avoid recomputing prep if not needed
    else
        LogDensityFunction(f.model, f.varinfo, f.context; adtype=adtype)
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
    ::Type{<:LogDensityFunction{M,V,C,Nothing}}
) where {M,V,C}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{M,V,C,AD}}
) where {M,V,C,AD<:ADTypes.AbstractADType}
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.logdensity(f::LogDensityFunction, x::AbstractVector)
    return logdensity_at(x, f.model, f.varinfo, f.context)
end
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction{M,V,C,AD}, x::AbstractVector
) where {M,V,C,AD<:ADTypes.AbstractADType}
    f.prep === nothing &&
        error("Gradient preparation not available; this should not happen")
    x = map(identity, x)  # Concretise type
    # Make branching statically inferrable, i.e. type-stable (even if the two
    # branches happen to return different types)
    return if use_closure(f.adtype)
        DI.value_and_gradient(
            x -> logdensity_at(x, f.model, f.varinfo, f.context), f.prep, f.adtype, x
        )
    else
        DI.value_and_gradient(
            logdensity_at,
            f.prep,
            f.adtype,
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
    return LogDensityFunction(model, f.varinfo, f.context; adtype=f.adtype)
end

"""
    getparams(f::LogDensityFunction)

Return the parameters of the wrapped varinfo as a vector.
"""
getparams(f::LogDensityFunction) = f.varinfo[:]
