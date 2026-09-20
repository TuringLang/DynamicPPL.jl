"""
    ParamsWithStats

A struct which contains parameter values extracted from a `VarInfo`, along with any
statistics associated with the VarInfo. The statistics are provided as a NamedTuple and are
optional.
"""
struct ParamsWithStats{P<:VarNamedTuple,S<:NamedTuple}
    params::P
    stats::S
end

"""
    ParamsWithStats(
        init_strategy::AbstractInitStrategy,
        model::Model,
        stats::NamedTuple=NamedTuple();
        include_colon_eq::Bool=true,
        include_log_probs::Bool=true,
    )

Generate a `ParamsWithStats` by re-evaluating the given `model` with the provided
`init_strategy`. Re-evaluation of the model is often necessary to obtain correct parameter
values as well as log probabilities. This is especially true when using linked VarInfos,
i.e., when variables have been transformed to unconstrained space, and if this is not done,
subtle correctness bugs may arise: see, e.g.,
https://github.com/TuringLang/Turing.jl/issues/2195.

`include_colon_eq` controls whether variables on the left-hand side of `:=` are included in
the resulting parameters.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    init_strategy::AbstractInitStrategy,
    model::DynamicPPL.Model,
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
)
    # Route through the same `@noinline _pws_eval` barrier as the vector-based method below,
    # so the accumulator tuple -- and hence the `(retval, varinfo)` result of `init!!` -- is
    # concretely typed at the call site rather than `Union`-typed (see `_pws_eval` for why
    # this matters on Julia 1.12 `-O2`).
    return if include_log_probs
        accs = (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.RawValueAccumulator(include_colon_eq),
        )
        _pws_eval(model, accs, init_strategy, stats, true)
    else
        accs = (DynamicPPL.RawValueAccumulator(include_colon_eq),)
        _pws_eval(model, accs, init_strategy, stats, false)
    end
end

"""
    ParamsWithStats(
        varinfo::AbstractVarInfo,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

There is one case where re-evaluation is not necessary, which is when the VarInfos all
already contain `DynamicPPL.RawValueAccumulator`. This accumulator stores values
as seen during the model evaluation, so the values can be simply read off. In this case,
the `model` argument can be omitted, and no re-evaluation will be performed. However, it is
the caller's responsibility to ensure that `RawValueAccumulator` is indeed present
inside `varinfo`.

`include_log_probs` controls whether log probabilities (log prior, log likelihood, and log
joint) are added to the resulting statistics NamedTuple.
"""
function ParamsWithStats(
    varinfo::AbstractVarInfo, stats::NamedTuple=NamedTuple(); include_log_probs::Bool=true
)
    params = densify!!(get_raw_values(varinfo))
    if include_log_probs
        has_prior_acc = DynamicPPL.hasacc(varinfo, Val(:LogPrior))
        has_likelihood_acc = DynamicPPL.hasacc(varinfo, Val(:LogLikelihood))
        if has_prior_acc
            stats = merge(stats, (logprior=DynamicPPL.getlogprior(varinfo),))
        end
        if has_likelihood_acc
            stats = merge(stats, (loglikelihood=DynamicPPL.getloglikelihood(varinfo),))
        end
        if has_prior_acc && has_likelihood_acc
            stats = merge(stats, (logjoint=DynamicPPL.getlogjoint(varinfo),))
        end
    end
    return ParamsWithStats(params, stats)
end

"""
    ParamsWithStats(
        param_vector::AbstractVector,
        ldf::DynamicPPL.LogDensityFunction,
        stats::NamedTuple=NamedTuple();
        include_colon_eq::Bool=true,
        include_log_probs::Bool=true,
    )

Generate a `ParamsWithStats` by re-evaluating the given `ldf` with the provided
`param_vector`.

This method is intended to replace the old method of obtaining parameters and statistics
via `unflatten!!` plus re-evaluation. It is faster for two reasons:

1. It does not rely on `deepcopy`-ing the VarInfo object (this used to be mandatory as
   otherwise re-evaluation would mutate the VarInfo, rendering it unusable for subsequent
   MCMC iterations).
2. The re-evaluation is faster as it uses `OnlyAccsVarInfo`.

Furthermore, if the `LogDensityFunction` has all fixed transforms (i.e., was constructed
with `fix_transforms=true`), and neither `include_log_probs` nor `include_colon_eq` is
set, then model re-evaluation is skipped entirely and the raw parameter values are
extracted directly from the parameter vector using the cached transforms.
"""
function ParamsWithStats(
    param_vector::AbstractVector,
    ldf::DynamicPPL.LogDensityFunction,
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
)
    return pws_with_eval(param_vector, ldf, stats; include_colon_eq, include_log_probs)
end
function pws_with_eval(
    param_vector::AbstractVector,
    ldf::DynamicPPL.LogDensityFunction,
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
)
    strategy = InitFromVector(param_vector, ldf)
    # `_pws_eval` is called separately from each branch so that `accs` -- and hence the
    # `(retval, varinfo)` tuple returned by `init!!` inside it -- has a concrete type at each
    # call site. If `accs` were `Union`-typed here, Julia 1.12's `-O2` optimizer would
    # heap-box that union-typed tuple and, because `retval` is unused, leave its pointer
    # fields uninitialized across a GC safepoint while the box is GC-rooted; a garbage
    # collection landing there scans those fields and segfaults in `gc_mark_obj8`.
    return if include_log_probs
        accs = (
            DynamicPPL.LogPriorAccumulator(),
            DynamicPPL.LogLikelihoodAccumulator(),
            DynamicPPL.RawValueAccumulator(include_colon_eq),
        )
        _pws_eval(ldf.model, accs, strategy, stats, true)
    else
        accs = (DynamicPPL.RawValueAccumulator(include_colon_eq),)
        _pws_eval(ldf.model, accs, strategy, stats, false)
    end
end
@noinline function _pws_eval(
    model::Model, accs::Tuple, strategy, stats::NamedTuple, include_log_probs::Bool
)
    # UnlinkAll() actually doesn't have any impact here, because there isn't even a
    # LogJacobianAccumulator; consequently, it doesn't matter whether we interpret the
    # parameters as being in linked space or not. However, we just include it for clarity.
    _, vi = DynamicPPL.init!!(
        model, OnlyAccsVarInfo(AccumulatorTuple(accs)), strategy, UnlinkAll()
    )
    params = densify!!(get_raw_values(vi))
    if include_log_probs
        stats = merge(
            stats,
            (
                logprior=DynamicPPL.getlogprior(vi),
                loglikelihood=DynamicPPL.getloglikelihood(vi),
                logjoint=DynamicPPL.getlogjoint(vi),
            ),
        )
    end
    return ParamsWithStats(params, stats)
end

# Specialisation for when the LDF is known to have all fixed transforms. In this case, we
# can avoid reevaluating the model because the transformed values + their transforms are
# all known (unless we need log probs, or `:=` results, in which case we will just have
# to reevaluate anyway).
function ParamsWithStats(
    param_vector::AbstractVector,
    ldf::LogDensityFunction{M,A,L,F,V,D,X,C,true},
    stats::NamedTuple=NamedTuple();
    include_colon_eq::Bool=true,
    include_log_probs::Bool=true,
) where {M,A,L,F,V,D,X,C}
    return if include_log_probs || include_colon_eq
        pws_with_eval(param_vector, ldf, stats; include_colon_eq, include_log_probs)
    else
        actual_length = length(param_vector)
        expected_length = LogDensityProblems.dimension(ldf)
        if actual_length != expected_length
            throw(
                ArgumentError(
                    "The length of the input vector is $(actual_length), but the LogDensityFunction expects a vector of length $(expected_length) based on the ranges that were extracted when the LogDensityFunction was constructed.",
                ),
            )
        end
        params = VarNamedTuple()
        for (vn, rat) in pairs(ldf._varname_ranges)
            top_sym = AbstractPPL.getsym(vn)
            template = get(ldf._varname_ranges.data, top_sym, DynamicPPL.NoTemplate())
            raw_val = rat.transform.transform(param_vector[rat.range])
            params = DynamicPPL.templated_setindex!!(params, raw_val, vn, template)
        end
        params = densify!!(params)
        ParamsWithStats(params, stats)
    end
end

function Base.show(io::IO, ::MIME"text/plain", pws::ParamsWithStats)
    printstyled(io, "ParamsWithStats"; bold=true)
    print(io, "\n ├─ ")
    if isempty(pws.params)
        printstyled(io, "params"; bold=true)
        println(io, " (empty)")
    else
        printstyled(io, "params"; bold=true)
        print(io, "\n │  ")
        DynamicPPL.VarNamedTuples.vnt_pretty_print(io, pws.params, " │  ", 0)
        println(io)
    end
    print(io, " └─ ")
    printstyled(io, "stats"; bold=true)
    if isempty(pws.stats)
        println(io, " (empty)")
    else
        n = length(pws.stats)
        for (i, (k, v)) in enumerate(pairs(pws.stats))
            if i == n
                print(io, "\n    └─ ")
            else
                print(io, "\n    ├─ ")
            end
            printstyled(io, k; color=:blue)
            print(io, " = ")
            show(io, v)
        end
    end
    return nothing
end

function Base.:(==)(pws1::ParamsWithStats, pws2::ParamsWithStats)
    return (pws1.params == pws2.params) & (pws1.stats == pws2.stats)
end
function Base.isequal(pws1::ParamsWithStats, pws2::ParamsWithStats)
    return isequal(pws1.params, pws2.params) && isequal(pws1.stats, pws2.stats)
end

"""
    InitFromParams(
        ps::ParamsWithStats,
        fallback::Union{Nothing,AbstractInitStrategy}=InitFromPrior()
    )

Initialise a model using the parameters stored in `ps`. The stats are ignored. `fallback` is
used if the model requires the value of a parameter which is not present in `ps.params`.
"""
function InitFromParams(
    ps::ParamsWithStats, fallback::Union{Nothing,AbstractInitStrategy}=InitFromPrior()
)
    return InitFromParams(ps.params, fallback)
end

_sampling_output_params(draw::ParamsWithStats) = draw.params
_sampling_output_params(draw::VarNamedTuple) = draw

"""
    convert(::Type{T}, output::AbstractMCMC.SamplingOutput)

Convert structured `SamplingOutput` draws to an `AbstractMCMC.AbstractChains` type using
`AbstractMCMC.from_samples`. This fallback converts only the draws and drops chain-level
metadata. Chain packages that preserve metadata should overload this method, or extend
`from_samples` to accept `iterations`, `sampling_stats`, and `sampler_states` and forward
those fields from their overload.

```julia
AbstractMCMC.from_samples(::Type{T}, draws; iterations, sampling_stats, sampler_states) where {T} =
    Chain(draws; iterations, sampling_stats, sampler_states) # package-specific constructor
Base.convert(::Type{T}, o::AbstractMCMC.SamplingOutput) where {T<:AbstractMCMC.AbstractChains} =
    AbstractMCMC.from_samples(T, o.samples; iterations=o.iterations,
                              sampling_stats=o.sampling_stats, sampler_states=o.sampler_states)
```
"""
function Base.convert(
    ::Type{T}, output::AbstractMCMC.SamplingOutput{<:Union{ParamsWithStats,VarNamedTuple}}
) where {T<:AbstractChains}
    return AbstractMCMC.from_samples(T, output.samples)
end

"""
    returned(model::Model, chain::AbstractMCMC.SamplingOutput)

Return a matrix of model return values evaluated at each draw's parameters.
"""
function returned(
    model::Model, chain::AbstractMCMC.SamplingOutput{<:Union{ParamsWithStats,VarNamedTuple}}
)
    return map(draw -> returned(model, _sampling_output_params(draw)), chain.samples)
end

for f in (:logjoint, :logprior, :(Distributions.loglikelihood))
    @eval function $f(
        model::Model,
        chain::AbstractMCMC.SamplingOutput{<:Union{ParamsWithStats,VarNamedTuple}},
    )
        return map(draw -> $f(model, _sampling_output_params(draw)), chain.samples)
    end
end

for f in (:pointwise_logdensities, :pointwise_loglikelihoods, :pointwise_prior_logdensities)
    @eval begin
        """
            $($f)(model::Model, chain::AbstractMCMC.SamplingOutput; factorize=false)

        Evaluate `$($f)` at each draw and return a `SamplingOutput` of `VarNamedTuple`s.
        Preserve iteration indices; omit sampling statistics and sampler states.
        All model parameters must be supplied in each draw.

        $(_FACTORIZE_KWARG_DOC)
        """
        function $f(
            model::Model,
            chain::AbstractMCMC.SamplingOutput{<:Union{ParamsWithStats,VarNamedTuple}};
            factorize=false,
        )
            densities = map(chain.samples) do draw
                $f(model, InitFromParams(_sampling_output_params(draw), nothing); factorize)
            end
            return AbstractMCMC.SamplingOutput(densities; iterations=chain.iterations)
        end
    end
end

"""
    predict([rng::AbstractRNG,] model::Model, chain::AbstractMCMC.SamplingOutput; include_all=true)

Sample predictions using each draw's parameters, drawing absent variables from their priors.

Return a `SamplingOutput` with the input's iteration indices and freshly evaluated log
probabilities. Set `include_all=false` to omit parameters supplied by each input draw.
Sampling times and sampler states are not carried over to the predictions.
"""
function predict(
    rng::Random.AbstractRNG,
    model::Model,
    chain::AbstractMCMC.SamplingOutput{<:Union{ParamsWithStats,VarNamedTuple}};
    include_all::Bool=true,
)
    predictions = map(chain.samples) do draw
        params = _sampling_output_params(draw)
        vi = OnlyAccsVarInfo(
            AccumulatorTuple(
                LogPriorAccumulator(),
                LogLikelihoodAccumulator(),
                RawValueAccumulator(true),
            ),
        )
        _, vi = init!!(rng, model, vi, InitFromParams(params), UnlinkAll())
        prediction = ParamsWithStats(vi)
        if include_all
            prediction
        else
            predicted_params = VarNamedTuple()
            # Raw values retain sampled-variable boundaries before arrays are densified.
            for (vn, value) in pairs(get_raw_values(vi))
                leaves = AbstractPPL.varname_and_value_leaves(vn, value)
                isempty(leaves) && haskey(params, vn) && !ismissing(params[vn]) && continue
                keep_all = all(
                    p -> !haskey(params, first(p)) || ismissing(params[first(p)]),
                    leaves,
                )
                retained = keep_all ? ((vn, value),) : leaves
                for (leaf, leaf_value) in retained
                    if keep_all || !haskey(params, leaf) || ismissing(params[leaf])
                        predicted_params = templated_setindex!!(
                            predicted_params,
                            leaf_value,
                            leaf,
                            prediction.params.data[AbstractPPL.getsym(leaf)],
                        )
                    end
                end
            end
            ParamsWithStats(densify!!(predicted_params), prediction.stats)
        end
    end
    return AbstractMCMC.SamplingOutput(predictions; iterations=chain.iterations)
end
function predict(
    model::Model,
    chain::AbstractMCMC.SamplingOutput{<:Union{ParamsWithStats,VarNamedTuple}};
    kwargs...,
)
    return predict(Random.default_rng(), model, chain; kwargs...)
end
