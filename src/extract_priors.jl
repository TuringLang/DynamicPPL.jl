struct PriorDistributionAccumulator{D<:OrderedDict{VarName,Any}} <: AbstractAccumulator
    priors::D
end

PriorDistributionAccumulator() = PriorDistributionAccumulator(OrderedDict{VarName,Any}())

function Base.copy(acc::PriorDistributionAccumulator)
    return PriorDistributionAccumulator(copy(acc.priors))
end

accumulator_name(::PriorDistributionAccumulator) = :PriorDistributionAccumulator

split(acc::PriorDistributionAccumulator) = PriorDistributionAccumulator(empty(acc.priors))
function combine(acc1::PriorDistributionAccumulator, acc2::PriorDistributionAccumulator)
    return PriorDistributionAccumulator(merge(acc1.priors, acc2.priors))
end

function setprior!(acc::PriorDistributionAccumulator, vn::VarName, dist::Distribution)
    acc.priors[vn] = dist
    return acc
end

function setprior!(
    acc::PriorDistributionAccumulator, vns::AbstractArray{<:VarName}, dist::Distribution
)
    for vn in vns
        acc.priors[vn] = dist
    end
    return acc
end

function setprior!(
    acc::PriorDistributionAccumulator,
    vns::AbstractArray{<:VarName},
    dists::AbstractArray{<:Distribution},
)
    for (vn, dist) in zip(vns, dists)
        acc.priors[vn] = dist
    end
    return acc
end

function accumulate_assume!!(acc::PriorDistributionAccumulator, val, logjac, vn, right)
    return setprior!(acc, vn, right)
end

accumulate_observe!!(acc::PriorDistributionAccumulator, right, left, vn) = acc

"""
    extract_priors([rng::Random.AbstractRNG, ]model::Model)

Extract the priors from a model.

This is done by sampling from the model and
recording the distributions that are used to generate the samples.

!!! warning
    Because the extraction is done by execution of the model, there
    are several caveats:

    1. If one variable, say, `y ~ Normal(0, x)`, where `x ~ Normal()`
       is also a random variable, then the extracted prior will have
       different parameters in every extraction!
    2. If the model does _not_ have static support, say,
       `n ~ Categorical(1:10); x ~ MvNormmal(zeros(n), I)`, then the
       extracted priors themselves will be different between extractions,
       not just their parameters.

    Both of these caveats are demonstrated below.

# Examples

## Changing parameters

```jldoctest
julia> using Distributions, StableRNGs

julia> rng = StableRNG(42);

julia> @model function model_dynamic_parameters()
           x ~ Normal(0, 1)
           y ~ Normal(x, 1)
       end;

julia> model = model_dynamic_parameters();

julia> extract_priors(rng, model)[@varname(y)]
Normal{Float64}(μ=-0.6702516921145671, σ=1.0)

julia> extract_priors(rng, model)[@varname(y)]
Normal{Float64}(μ=1.3736306979834252, σ=1.0)
```

## Changing support

```jldoctest
julia> using LinearAlgebra, Distributions, StableRNGs

julia> rng = StableRNG(42);

julia> @model function model_dynamic_support()
           n ~ Categorical(ones(10) ./ 10)
           x ~ MvNormal(zeros(n), I)
       end;

julia> model = model_dynamic_support();

julia> length(extract_priors(rng, model)[@varname(x)])
6

julia> length(extract_priors(rng, model)[@varname(x)])
9
```
"""
extract_priors(args::Union{Model,AbstractVarInfo}...) =
    extract_priors(Random.default_rng(), args...)
function extract_priors(rng::Random.AbstractRNG, model::Model)
    varinfo = VarInfo()
    # TODO(mhauru) This doesn't actually need the NumProduceAccumulator, it's only a
    # workaround for the fact that `order` is still hardcoded in VarInfo, and hence you
    # can't push new variables without knowing the num_produce. Remove this when possible.
    varinfo = setaccs!!(varinfo, (PriorDistributionAccumulator(), NumProduceAccumulator()))
    varinfo = last(evaluate!!(model, varinfo, SamplingContext(rng)))
    return getacc(varinfo, Val(:PriorDistributionAccumulator)).priors
end

"""
    extract_priors(model::Model, varinfo::AbstractVarInfo)

Extract the priors from a model.

This is done by evaluating the model at the values present in `varinfo`
and recording the distributions that are present at each tilde statement.
"""
function extract_priors(model::Model, varinfo::AbstractVarInfo)
    # TODO(mhauru) This doesn't actually need the NumProduceAccumulator, it's only a
    # workaround for the fact that `order` is still hardcoded in VarInfo, and hence you
    # can't push new variables without knowing the num_produce. Remove this when possible.
    varinfo = setaccs!!(
        deepcopy(varinfo), (PriorDistributionAccumulator(), NumProduceAccumulator())
    )
    varinfo = last(evaluate!!(model, varinfo, DefaultContext()))
    return getacc(varinfo, Val(:PriorDistributionAccumulator)).priors
end
