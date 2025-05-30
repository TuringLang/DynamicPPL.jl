"""
    PointwiseLogProbAccumulator{whichlogprob,KeyType,D<:AbstractDict{KeyType}} <: AbstractAccumulator

An accumulator that stores the log-probabilities of each variable in a model.

Internally this context stores the log-probabilities in a dictionary, where the keys are
the variable names and the values are vectors of log-probabilities. Each element in a vector
corresponds to one execution of the model.

`whichlogprob` is a symbol that can be `:both`, `:prior`, or `:likelihood`, and specifies
which log-probabilities to store in the accumulator. `KeyType` is the type by which variable
names are stored, and should be `String` or `VarName`. `D` is the type of the dictionary
used internally to store the log-probabilities, by default
`OrderedDict{KeyType, Vector{LogProbType}}`.
"""
struct PointwiseLogProbAccumulator{whichlogprob,KeyType,D<:AbstractDict{KeyType}} <:
       AbstractAccumulator
    logps::D
end

function PointwiseLogProbAccumulator{whichlogprob}(logps) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob,keytype(logps),typeof(logps)}(logps)
end

function PointwiseLogProbAccumulator{whichlogprob}() where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob,VarName}()
end

function PointwiseLogProbAccumulator{whichlogprob,KeyType}() where {whichlogprob,KeyType}
    logps = OrderedDict{KeyType,Vector{LogProbType}}()
    return PointwiseLogProbAccumulator{whichlogprob,KeyType,typeof(logps)}(logps)
end

function Base.copy(acc::PointwiseLogProbAccumulator{whichlogprob}) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob}(copy(acc.logps))
end

function Base.push!(acc::PointwiseLogProbAccumulator, vn, logp)
    logps = acc.logps
    # The last(fieldtypes(eltype(...))) gets the type of the values, rather than the keys.
    T = last(fieldtypes(eltype(logps)))
    logpvec = get!(logps, vn, T())
    return push!(logpvec, logp)
end

function Base.push!(
    acc::PointwiseLogProbAccumulator{whichlogprob,String}, vn::VarName, logp
) where {whichlogprob}
    return push!(acc, string(vn), logp)
end

function accumulator_name(
    ::Type{<:PointwiseLogProbAccumulator{whichlogprob}}
) where {whichlogprob}
    return Symbol("PointwiseLogProbAccumulator{$whichlogprob}")
end

function split(acc::PointwiseLogProbAccumulator{whichlogprob}) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob}(empty(acc.logps))
end

function combine(
    acc::PointwiseLogProbAccumulator{whichlogprob},
    acc2::PointwiseLogProbAccumulator{whichlogprob},
) where {whichlogprob}
    return PointwiseLogProbAccumulator{whichlogprob}(mergewith(vcat, acc.logps, acc2.logps))
end

function accumulate_assume!!(
    acc::PointwiseLogProbAccumulator{whichlogprob}, val, logjac, vn, right
) where {whichlogprob}
    if whichlogprob == :both || whichlogprob == :prior
        # T is the element type of the vectors that are the values of `acc.logps`. Usually
        # it's LogProbType.
        T = eltype(last(fieldtypes(eltype(acc.logps))))
        subacc = accumulate_assume!!(LogPriorAccumulator{T}(), val, logjac, vn, right)
        push!(acc, vn, subacc.logp)
    end
    return acc
end

function accumulate_observe!!(
    acc::PointwiseLogProbAccumulator{whichlogprob}, right, left, vn
) where {whichlogprob}
    # If `vn` is nothing the LHS of ~ is a literal and we don't have a name to attach this
    # acc to, and thus do nothing.
    if vn === nothing
        return acc
    end
    if whichlogprob == :both || whichlogprob == :likelihood
        # T is the element type of the vectors that are the values of `acc.logps`. Usually
        # it's LogProbType.
        T = eltype(last(fieldtypes(eltype(acc.logps))))
        subacc = accumulate_observe!!(LogLikelihoodAccumulator{T}(), right, left, vn)
        push!(acc, vn, subacc.logp)
    end
    return acc
end

"""
    pointwise_logdensities(
        model::Model,
        chain::Chains,
        keytype=String,
        context=DefaultContext(),
        ::Val{whichlogprob}=Val(:both),
    )

Runs `model` on each sample in `chain` returning a `OrderedDict{String, Matrix{Float64}}`
with keys corresponding to symbols of the variables, and values being matrices
of shape `(num_chains, num_samples)`.

`keytype` specifies what the type of the keys used in the returned `OrderedDict` are.
Currently, only `String` and `VarName` are supported. `context` is the evaluation context,
and `whichlogprob` specifies which log-probabilities to compute. It can be `:both`,
`:prior`, or `:likelihood`.

See also: [`pointwise_loglikelihoods`](@ref), [`pointwise_loglikelihoods`](@ref).

# Notes
Say `y` is a `Vector` of `n` i.i.d. `Normal(μ, σ)` variables, with `μ` and `σ`
both being `<:Real`. Then the *observe* (i.e. when the left-hand side is an
*observation*) statements can be implemented in three ways:
1. using a `for` loop:
```julia
for i in eachindex(y)
    y[i] ~ Normal(μ, σ)
end
```
2. using `.~`:
```julia
y .~ Normal(μ, σ)
```
3. using `MvNormal`:
```julia
y ~ MvNormal(fill(μ, n), σ^2 * I)
```

In (1) and (2), `y` will be treated as a collection of `n` i.i.d. 1-dimensional variables,
while in (3) `y` will be treated as a _single_ n-dimensional observation.

This is important to keep in mind, in particular if the computation is used
for downstream computations.

# Examples
## From chain
```jldoctest pointwise-logdensities-chains; setup=:(using Distributions)
julia> using MCMCChains

julia> @model function demo(xs, y)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in eachindex(xs)
               xs[i] ~ Normal(m, √s)
           end
           y ~ Normal(m, √s)
       end
demo (generic function with 2 methods)

julia> # Example observations.
       model = demo([1.0, 2.0, 3.0], [4.0]);

julia> # A chain with 3 iterations.
       chain = Chains(
           reshape(1.:6., 3, 2),
           [:s, :m]
       );

julia> pointwise_logdensities(model, chain)
OrderedDict{String, Matrix{Float64}} with 6 entries:
  "s"     => [-0.802775; -1.38222; -2.09861;;]
  "m"     => [-8.91894; -7.51551; -7.46824;;]
  "xs[1]" => [-5.41894; -5.26551; -5.63491;;]
  "xs[2]" => [-2.91894; -3.51551; -4.13491;;]
  "xs[3]" => [-1.41894; -2.26551; -2.96824;;]
  "y"     => [-0.918939; -1.51551; -2.13491;;]

julia> pointwise_logdensities(model, chain, String)
OrderedDict{String, Matrix{Float64}} with 6 entries:
  "s"     => [-0.802775; -1.38222; -2.09861;;]
  "m"     => [-8.91894; -7.51551; -7.46824;;]
  "xs[1]" => [-5.41894; -5.26551; -5.63491;;]
  "xs[2]" => [-2.91894; -3.51551; -4.13491;;]
  "xs[3]" => [-1.41894; -2.26551; -2.96824;;]
  "y"     => [-0.918939; -1.51551; -2.13491;;]

julia> pointwise_logdensities(model, chain, VarName)
OrderedDict{VarName, Matrix{Float64}} with 6 entries:
  s     => [-0.802775; -1.38222; -2.09861;;]
  m     => [-8.91894; -7.51551; -7.46824;;]
  xs[1] => [-5.41894; -5.26551; -5.63491;;]
  xs[2] => [-2.91894; -3.51551; -4.13491;;]
  xs[3] => [-1.41894; -2.26551; -2.96824;;]
  y     => [-0.918939; -1.51551; -2.13491;;]
```

## Broadcasting
Note that `x .~ Dist()` will treat `x` as a collection of
_independent_ observations rather than as a single observation.

```jldoctest; setup = :(using Distributions)
julia> @model function demo(x)
           x .~ Normal()
       end;

julia> m = demo([1.0, ]);

julia> ℓ = pointwise_logdensities(m, VarInfo(m)); first(ℓ[@varname(x[1])])
-1.4189385332046727

julia> m = demo([1.0; 1.0]);

julia> ℓ = pointwise_logdensities(m, VarInfo(m)); first.((ℓ[@varname(x[1])], ℓ[@varname(x[2])]))
(-1.4189385332046727, -1.4189385332046727)
```
"""
function pointwise_logdensities(
    model::Model,
    chain,
    ::Type{KeyType}=String,
    context::AbstractContext=DefaultContext(),
    ::Val{whichlogprob}=Val(:both),
) where {KeyType,whichlogprob}
    # Get the data by executing the model once
    vi = VarInfo(model)

    AccType = PointwiseLogProbAccumulator{whichlogprob,KeyType}
    vi = setaccs!!(vi, (AccType(),))

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    for (sample_idx, chain_idx) in iters
        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)

        # Execute model
        vi = last(evaluate!!(model, vi, context))
    end

    logps = getacc(vi, Val(accumulator_name(AccType))).logps
    niters = size(chain, 1)
    nchains = size(chain, 3)
    logdensities = OrderedDict(
        varname => reshape(vals, niters, nchains) for (varname, vals) in logps
    )
    return logdensities
end

function pointwise_logdensities(
    model::Model,
    varinfo::AbstractVarInfo,
    context::AbstractContext=DefaultContext(),
    ::Val{whichlogprob}=Val(:both),
) where {whichlogprob}
    AccType = PointwiseLogProbAccumulator{whichlogprob}
    varinfo = setaccs!!(varinfo, (AccType(),))
    varinfo = last(evaluate!!(model, varinfo, context))
    return getacc(varinfo, Val(accumulator_name(AccType))).logps
end

"""
    pointwise_loglikelihoods(model, chain[, keytype, context])

Compute the pointwise log-likelihoods of the model given the chain.
This is the same as `pointwise_logdensities(model, chain, context)`, but only
including the likelihood terms.

See also: [`pointwise_logdensities`](@ref), [`pointwise_prior_logdensities`](@ref).
"""
function pointwise_loglikelihoods(
    model::Model, chain, keytype::Type{T}=String, context::AbstractContext=DefaultContext()
) where {T}
    return pointwise_logdensities(model, chain, T, context, Val(:likelihood))
end

function pointwise_loglikelihoods(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext=DefaultContext()
)
    return pointwise_logdensities(model, varinfo, context, Val(:likelihood))
end

"""
    pointwise_prior_logdensities(model, chain[, keytype, context])

Compute the pointwise log-prior-densities of the model given the chain.
This is the same as `pointwise_logdensities(model, chain, context)`, but only
including the prior terms.

See also: [`pointwise_logdensities`](@ref), [`pointwise_loglikelihoods`](@ref).
"""
function pointwise_prior_logdensities(
    model::Model, chain, keytype::Type{T}=String, context::AbstractContext=DefaultContext()
) where {T}
    return pointwise_logdensities(model, chain, T, context, Val(:prior))
end

function pointwise_prior_logdensities(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext=DefaultContext()
)
    return pointwise_logdensities(model, varinfo, context, Val(:prior))
end
