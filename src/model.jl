"""
    struct Model{F, parameternames, observationnames, Tparams, Tobs} <: AbstractProbabilisticProgram
        name::Symbol
        evaluator::F
        parameters::NamedTuple{parameternames,Tparams}
        observations::NamedTuple{observationnames,Tobs}
    end

A `Model` struct with model evaluation function of type `F`, parameters of names `parameternames` 
and types `Tparams`, and observations of names `observatonnames` and types `Tobs`.

`parameternames` and `observationnames` are tuples of symbols, e.g. `(:a, :b)`.

# Examples

```julia
TODO
```
"""
struct Model{F, parameternames, observationnames, Tparams, Tobs} <: AbstractProbabilisticProgram
    name::Symbol
    # code::Expr
    evaluator::F
    parameters::NamedTuple{parameternames,Tparams}
    observations::NamedTuple{observationnames,Tobs}
end



"""
    @ConditionedModel{; obs1::Type1, obs2::Type2, ...}
    @ConditionedModel{f, parameternames, Tparams; obs1::Type1, obs2::Type2, ...}

Macro with more convenient syntax for declaring `Model` types with observations (similar to the
`Base.@NamedTuple` macro).  The observations to the parameters part of the braces:
`@ConditionedModel{; x::Int, y}`.  Type annotations can be omitted, in which case the type is
defaulted to `Any`.

The non-parameters part can be used to match the other type arguments of `Model`: the evaluator
function type `F`, and the `parameternames` and their type tuple `Tparams`.
"""
macro ConditionedModel(ex)
    # Code adapted from Base.@NamedTuple macro; parameter lists in `:braces` expressions do work:
    # julia> :(@bla{f; x, y}).args
    # 3-element Array{Any,1}:
    #  Symbol("@bla")
    #  :(#= REPL[55]:1 =#)
    #  :({$(Expr(:parameters, :x, :y)), f})

    Meta.isexpr(ex, :braces) || throw(ArgumentError("@ConditionedModel expects {;...}"))
    decls = filter(e -> !(e isa LineNumberNode), ex.args)
    Meta.isexpr(decls[1], :parameters) || throw(ArgumentError("@ConditionedModel expects {;...}"))
    cond_part = decls[1].args
    types_part = decls[2:end]
    all(e -> e isa Symbol || Meta.isexpr(e, :(::)), cond_part) ||
        throw(ArgumentError("@ConditionedModel must contain a sequence of name or name::type expressions"))
    obsvars = [QuoteNode(e isa Symbol ? e : e.args[1]) for e in cond_part]
    obstypes = [esc(e isa Symbol ? :Any : e.args[2]) for e in cond_part]
    _f = esc(get(types_part, 1, :Any))
    _parameternames = esc(get(types_part, 2, :Any))
    _tparams = esc(get(types_part, 3, :Any))
    
    return :($(DynamicPPL.Model){
        $_f,
        $_parameternames,
        ($(obsvars...),),
        $_tparams,
        Tuple{$(obstypes...)}
    })
end

"""
    GenerativeModel{F, parameters, TParams}

Type alias for models without observations.
"""
const GenerativeModel{F, parameternames, Tparams} = @ConditionedModel{F, parameternames, Tparams;}

function Base.show(io::IO, model::Model)
    println(io, "Model ", model.name, " given")
    print(io, "    parameters    ")
    join(io, getparameternames(model), ", ")
    println()
    print(io, "    observations ")
    join(io, getobservationnames(model), ", ")
    # println(_pretty(model.code))
end


"""
    (model::Model)([rng, varinfo, sampler, context])

Sample from the `model` using the `sampler` with random number generator `rng` and the
`context`, and store the sample and log joint probability in `varinfo`.

The method resets the log joint probability of `varinfo` and increases the evaluation
number of `sampler`.
"""
function (model::Model)(
    rng::Random.AbstractRNG,
    varinfo::AbstractVarInfo = VarInfo(),
    sampler::AbstractSampler = SampleFromPrior(),
    context::AbstractContext = DefaultContext(),
)
    if Threads.nthreads() == 1
        return evaluate_threadunsafe(rng, model, varinfo, sampler, context)
    else
        return evaluate_threadsafe(rng, model, varinfo, sampler, context)
    end
end
function (model::Model)(arguments...)
    return model(Random.GLOBAL_RNG, arguments...)
end

# without VarInfo
function (model::Model)(
    rng::Random.AbstractRNG,
    sampler::AbstractSampler,
    arguments...,
)
    return model(rng, VarInfo(), sampler, arguments...)
end

# without VarInfo and without AbstractSampler
function (model::Model)(rng::Random.AbstractRNG, context::AbstractContext)
    return model(rng, VarInfo(), SampleFromPrior(), context)
end

"""
    evaluate_threadunsafe(rng, model, varinfo, sampler, context)

Evaluate the `model` without wrapping `varinfo` inside a `ThreadSafeVarInfo`.

If the `model` makes use of Julia's multithreading this will lead to undefined behaviour.
This method is not exposed and supposed to be used only internally in DynamicPPL.

See also: [`evaluate_threadsafe`](@ref)
"""
function evaluate_threadunsafe(rng, model, varinfo, sampler, context)
    resetlogp!(varinfo)
    return _evaluate(rng, model, varinfo, sampler, context)
end

"""
    evaluate_threadsafe(rng, model, varinfo, sampler, context)

Evaluate the `model` with `varinfo` wrapped inside a `ThreadSafeVarInfo`.

With the wrapper, Julia's multithreading can be used for observe statements in the `model`
but parallel sampling will lead to undefined behaviour.
This method is not exposed and supposed to be used only internally in DynamicPPL.

See also: [`evaluate_threadunsafe`](@ref)
"""
function evaluate_threadsafe(rng, model, varinfo, sampler, context)
    resetlogp!(varinfo)
    wrapper = ThreadSafeVarInfo(varinfo)
    result = _evaluate(rng, model, wrapper, sampler, context)
    setlogp!(varinfo, getlogp(wrapper))
    return result
end

"""
    _evaluate(rng, model::Model, varinfo, sampler, context)

Evaluate the `model` with the arguments matching the given `sampler` and `varinfo` object.
"""
function _evaluate(rng, model::Model, varinfo, sampler, context)
    unwrapped_observations = map(obs -> matchingvalue(sampler, varinfo, obs), model.observations)
    return model.evaluator(
        rng,
        model,
        varinfo,
        sampler,
        context,
        model.parameters,
        unwrapped_observations
    )
end

"""
    getparameternames(model::Model)

Get a tuple of the argument names of the `model`.
"""
getparameternames(model::Model{_F,parameternames}) where {_F,parameternames} = parameternames

"""
    getparameternames(model::Model)

Get a tuple of the observation names of the `model`.
"""
function getobservationnames(
    model::Model{_F,_parameternames,observationnames}
) where {_F,_parameternames,observationnames}
    return observationnames
end

"""
    nameof(model::Model)

Get the name of the `model` as `Symbol`.
"""
Base.nameof(model::Model) = model.name

"""
    logdensity(model::Model, varinfo::AbstractVarInfo)

Return the log joint probability of variables in `varinfo` for the probabilistic `model`.

See [`logprior`](@ref) and [`loglikelihood`](@ref).
"""
function AbstractPPL.logdensity(model::Model, varinfo::AbstractVarInfo)
    model(varinfo, SampleFromPrior(), DefaultContext())
    return getlogp(varinfo)
end

function AbstractPPL.decondition(model::Model, name = Symbol(model.name, "_joint"))
    return Model(name, model.evaluator, model.parameters, (;))
end

function AbstractPPL.condition(model::Model, observations, name = Symbol(model.name, "_cond"))
    return Model(name, model.evaluator, model.parameters, merge(model.observations, observations))
end


# """
#     logprior(model::Model, varinfo::AbstractVarInfo)

# Return the log prior probability of variables `varinfo` for the probabilistic `model`.

# See also [`logjoint`](@ref) and [`loglikelihood`](@ref).
# """
# function logprior(model::Model, varinfo::AbstractVarInfo)
#     model(varinfo, SampleFromPrior(), PriorContext())
#     return getlogp(varinfo)
# end

# """
#     loglikelihood(model::Model, varinfo::AbstractVarInfo)

# Return the log likelihood of variables `varinfo` for the probabilistic `model`.

# See also [`logjoint`](@ref) and [`logprior`](@ref).
# """
# function Distributions.loglikelihood(model::Model, varinfo::AbstractVarInfo)
#     model(varinfo, SampleFromPrior(), LikelihoodContext())
#     return getlogp(varinfo)
# # end

"""
    generated_quantities(model::Model, chain::AbstractChains)

Execute `model` for each of the samples in `chain` and return an array of the values
returned by the `model` for each sample.

# Examples
## General
Often you might have additional quantities computed inside the model that you want to
inspect, e.g.
```julia
@model function demo(x)
    # sample and observe
    θ ~ Prior()
    x ~ Likelihood()
    return interesting_quantity(θ, x)
end
m = demo(data)
chain = sample(m, alg, n)
# To inspect the `interesting_quantity(θ, x)` where `θ` is replaced by samples
# from the posterior/`chain`:
generated_quantities(m, chain) # <= results in a `Vector` of returned values
                               #    from `interesting_quantity(θ, x)`
```
## Concrete (and simple)
```julia
julia> using DynamicPPL, Turing

julia> @model function demo(xs)
           s ~ InverseGamma(2, 3)
           m_shifted ~ Normal(10, √s)
           m = m_shifted - 10

           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end

           return (m, )
       end
demo (generic function with 1 method)

julia> model = demo(randn(10));

julia> chain = sample(model, MH(), 10);

julia> generated_quantities(model, chain)
10×1 Array{Tuple{Float64},2}:
 (2.1964758025119338,)
 (2.1964758025119338,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.09270081916291417,)
 (0.043088571494005024,)
 (-0.16489786710222099,)
 (-0.16489786710222099,)
```
"""
function generated_quantities(model::Model, chain::AbstractChains)
    varinfo = VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    return map(iters) do (sample_idx, chain_idx)
        setval_and_resample!(varinfo, chain, sample_idx, chain_idx)
        model(varinfo)
    end
end
