module DynamicPPLFlexiChainsExtTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using AbstractMCMC: AbstractMCMC
using DimensionalData: DimensionalData as DD
using Distributions
using DynamicPPL
using FlexiChains: FlexiChains, FlexiChain, VNChain, Parameter, Extra
using LinearAlgebra: I
using OffsetArrays: OffsetArray
using PosteriorStats: PosteriorStats
using Random: Random, Xoshiro
using StableRNGs: StableRNG
using Test

_LOGPRIOR_KEY = Extra(:logprior)
_LOGLIKELIHOOD_KEY = Extra(:loglikelihood)
_LOGJOINT_KEY = Extra(:logjoint)

function sample_from_prior(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    n_iters::Int,
    n_chains::Int=1;
    make_chain=true,
)
    vi = DynamicPPL.OnlyAccsVarInfo((
        DynamicPPL.default_accumulators()..., DynamicPPL.RawValueAccumulator(true)
    ))
    ps = [
        DynamicPPL.ParamsWithStats(
            last(DynamicPPL.init!!(rng, model, vi, InitFromPrior(), UnlinkAll()))
        ) for _ in 1:n_iters, _ in 1:n_chains
    ]
    return if make_chain
        AbstractMCMC.from_samples(VNChain, ps)
    else
        ps
    end
end
function sample_from_prior(
    model::DynamicPPL.Model, n_iters::Int, n_chains::Int=1; make_chain=true
)
    return sample_from_prior(Random.default_rng(), model, n_iters, n_chains; make_chain)
end

# For some of the `predict` tests, we need some way to draw from the posterior. We'll use
# importance sampling here since it's simple to implement.
function sample_from_posterior(rng::Random.AbstractRNG, model::DynamicPPL.Model)
    prior_samples = sample_from_prior(rng, model, 20000; make_chain=false)
    log_weights = vec([p.stats.loglikelihood for p in prior_samples])
    max_logw = maximum(log_weights)
    weights = exp.(log_weights .- max_logw)
    weights ./= sum(weights)
    dist = Categorical(weights)
    idxs = rand(rng, dist, 2000)
    return AbstractMCMC.from_samples(VNChain, hcat(prior_samples[idxs]))
end
function sample_from_posterior(model::DynamicPPL.Model)
    return sample_from_posterior(Random.default_rng(), model)
end

@testset "FlexiChainsExt" begin
    @testset "InitFromParams(chain, i, j)" begin
        @model function f()
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = f()
        chn = sample_from_prior(model, 50)

        for i in 1:50
            accs = OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
            _, accs = DynamicPPL.init!!(model, accs, InitFromParams(chn, i, 1), UnlinkAll())
            raw_values = get_raw_values(accs)
            for vn in (@varname(x), @varname(y))
                @test raw_values[vn] == chn[vn, iter=i, chain=1]
            end
        end
    end

    @testset "AbstractMCMC.from_samples" begin
        @model function f(z)
            x ~ Normal()
            y := x + 1
            return z ~ Normal(y)
        end

        z = 1.0
        model = f(z)

        ps = sample_from_prior(Xoshiro(468), model, 50, 3; make_chain=false)
        c = sample_from_prior(Xoshiro(468), model, 50, 3; make_chain=true)
        @test FlexiChains.parameters(c) == [@varname(x), @varname(y)]
        @test c[@varname(x)] == map(p -> p.params[@varname(x)], ps)
        @test c[@varname(y)] == c[@varname(x)] .+ 1
        @test logpdf.(Normal(), c[@varname(x)]) ≈ c[Extra(:logprior)]

        # test with VarNamedTuple
        vnts = [rand(model) for _ in 1:100, _ in 1:3]
        c2 = AbstractMCMC.from_samples(VNChain, vnts)
        @test c2 isa VNChain
        @test size(c2) == (100, 3)
        @test Set(FlexiChains.parameters(c2)) == Set(keys(rand(model)))
        @test c2[@varname(x)] == map(vnt -> vnt[@varname(x)], vnts)
    end

    @testset "parameters_at and values_at" begin
        @model function f()
            x ~ Normal()
            y = zeros(3)
            y[2] ~ Normal()
            z = (; a=nothing)
            return z.a ~ Normal()
        end
        Ni, Nc = 10, 2

        # These should give the same results, but chn is just the ParamsWithStats
        # bundled into a VNChain.
        chn = sample_from_prior(Xoshiro(468), f(), Ni, Nc; make_chain=true)
        pwss = sample_from_prior(Xoshiro(468), f(), Ni, Nc; make_chain=false)

        for i in 1:Ni, c in 1:Nc
            prms = FlexiChains.parameters_at(chn; iter=i, chain=c)
            @test prms isa VarNamedTuple
            @test prms == pwss[i, c].params
            vals = FlexiChains.values_at(chn; iter=i, chain=c)
            @test vals isa DynamicPPL.ParamsWithStats
            @test vals == pwss[i, c]
        end
    end

    @testset "return type of rand" begin
        @model function f()
            x ~ Normal()
            y ~ Normal()
            return nothing
        end
        chn = sample_from_prior(f(), 10; make_chain=true)
        @test rand(chn) isa DynamicPPL.ParamsWithStats
        @test rand(chn; parameters_only=true) isa DynamicPPL.VarNamedTuple
        @test rand(chn, 5) isa Vector{<:DynamicPPL.ParamsWithStats}
        @test rand(chn, 5; parameters_only=true) isa Vector{<:DynamicPPL.VarNamedTuple}
    end

    @testset "AbstractMCMC.to_samples" begin
        @model function f(z)
            x ~ Normal()
            y := x + 1
            return z ~ Normal(y)
        end

        # Make the chain first
        z = 1.0
        model = f(z)
        ps = sample_from_prior(Xoshiro(468), model, 50; make_chain=false)
        c = sample_from_prior(Xoshiro(468), model, 50; make_chain=true)

        # Then convert back to ParamsWithStats
        @model function newmodel()
            error(
                "This model should never be run, because there is structure info" *
                " in the chain.",
            )
            x ~ Normal()
            return nothing
        end

        @testset "with model" begin
            # Make sure that the model isn't actually ever used, by passing one that
            # errors when run.
            arr_pss = AbstractMCMC.to_samples(DynamicPPL.ParamsWithStats, c, newmodel())
            @test arr_pss == ps
            arr_pss = AbstractMCMC.to_samples(DynamicPPL.VarNamedTuple, c, newmodel())
            @test arr_pss == map(p -> p.params, ps)
        end
        @testset "without model" begin
            arr_pss = AbstractMCMC.to_samples(DynamicPPL.ParamsWithStats, c)
            @test arr_pss == ps
            arr_pss = AbstractMCMC.to_samples(DynamicPPL.VarNamedTuple, c)
            @test arr_pss == map(p -> p.params, ps)
        end
    end

    @testset "logp(model, chain)" begin
        @model function f()
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = f() | (; y=1.0)
        chn = sample_from_prior(model, 100; make_chain=true)
        xs = chn[@varname(x)]
        expected_logprior = logpdf.(Normal(), xs)
        expected_loglike = logpdf.(Normal.(xs), 1.0)

        @testset "logprior" begin
            lprior = logprior(model, chn)
            @test isapprox(lprior, expected_logprior)
            @test parent(parent(DD.dims(lprior, :iter))) == FlexiChains.iter_indices(chn)
            @test parent(parent(DD.dims(lprior, :chain))) == FlexiChains.chain_indices(chn)
        end
        @testset "loglikelihood" begin
            llike = loglikelihood(model, chn)
            @test isapprox(llike, expected_loglike)
            @test parent(parent(DD.dims(llike, :iter))) == FlexiChains.iter_indices(chn)
            @test parent(parent(DD.dims(llike, :chain))) == FlexiChains.chain_indices(chn)
        end
        @testset "logjoint" begin
            ljoint = logjoint(model, chn)
            @test isapprox(ljoint, expected_logprior .+ expected_loglike)
            @test parent(parent(DD.dims(ljoint, :iter))) == FlexiChains.iter_indices(chn)
            @test parent(parent(DD.dims(ljoint, :chain))) == FlexiChains.chain_indices(chn)
        end

        @testset "errors on missing variables" begin
            @model function xonly()
                return x ~ Normal()
            end
            @model function xy()
                x ~ Normal()
                return y ~ Normal()
            end
            chn = sample_from_prior(xonly(), 100; make_chain=true)
            @test_throws "not found in chain" logprior(xy(), chn)
            @test_throws "not found in chain" loglikelihood(xy(), chn)
            @test_throws "not found in chain" logjoint(xy(), chn)
        end

        @testset "with non-standard Array variables" begin
            @model function offset_lp(y)
                x = OffsetArray(zeros(2), -2:-1)
                x[-2] ~ Normal()
                y ~ Normal(x[-2])
                return nothing
            end
            model = offset_lp(2.0)
            chn = sample_from_prior(model, 50; make_chain=true)
            lprior = logprior(model, chn)
            @test logprior(model, chn) ≈ logpdf.(Normal(), chn[@varname(x[-2])])
            @test loglikelihood(model, chn) ≈ logpdf.(Normal.(chn[@varname(x[-2])]), 2.0)
        end
    end

    @testset "pointwise logprobs" begin
        @model function f(y)
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = f(1.0)

        chn = sample_from_prior(model, 100; make_chain=true)
        xs = chn[@varname(x)]

        @testset "logdensities" begin
            pld = DynamicPPL.pointwise_logdensities(model, chn)
            @test pld isa VNChain
            @test FlexiChains.iter_indices(pld) == FlexiChains.iter_indices(chn)
            @test FlexiChains.chain_indices(pld) == FlexiChains.chain_indices(chn)
            @test length(keys(pld)) == 2
            @test isapprox(pld[@varname(x)], logpdf.(Normal(), xs))
            @test isapprox(pld[@varname(y)], logpdf.(Normal.(xs), 1.0))
        end

        @testset "loglikelihoods" begin
            pld = DynamicPPL.pointwise_loglikelihoods(model, chn)
            @test pld isa VNChain
            @test FlexiChains.iter_indices(pld) == FlexiChains.iter_indices(chn)
            @test FlexiChains.chain_indices(pld) == FlexiChains.chain_indices(chn)
            @test length(keys(pld)) == 1
            @test isapprox(pld[@varname(y)], logpdf.(Normal.(xs), 1.0))
        end

        @testset "logpriors" begin
            pld = DynamicPPL.pointwise_prior_logdensities(model, chn)
            @test pld isa VNChain
            @test FlexiChains.iter_indices(pld) == FlexiChains.iter_indices(chn)
            @test FlexiChains.chain_indices(pld) == FlexiChains.chain_indices(chn)
            @test length(keys(pld)) == 1
            @test isapprox(pld[@varname(x)], logpdf.(Normal(), xs))
        end

        @testset "errors on missing variables" begin
            @model function xonly()
                return x ~ Normal()
            end
            @model function xy()
                x ~ Normal()
                return y ~ Normal()
            end
            chn = sample_from_prior(xonly(), 100; make_chain=true)
            @test_throws "not found in chain" DynamicPPL.pointwise_logdensities(xy(), chn)
            @test_throws "not found in chain" DynamicPPL.pointwise_loglikelihoods(xy(), chn)
            @test_throws "not found in chain" DynamicPPL.pointwise_prior_logdensities(
                xy(), chn
            )
        end

        @testset "with non-standard Array variables" begin
            @model function offset_pld(y)
                x = OffsetArray(zeros(2), -2:-1)
                x[-2] ~ Normal()
                y ~ Normal(x[-2])
                return nothing
            end
            model = offset_pld(2.0)
            chn = sample_from_prior(model, 50; make_chain=true)
            plds = DynamicPPL.pointwise_logdensities(model, chn)
            @test plds[@varname(x[-2])] == logpdf.(Normal(), chn[@varname(x[-2])])
            @test plds[@varname(y)] == logpdf.(Normal.(chn[@varname(x[-2])]), 2.0)
        end

        @testset "factorize=true" begin
            @model function array_pw(y, z)
                x ~ MvNormal(zeros(2), I)
                return y ~ MvNormal(x, I)
                # Doesn't work yet https://github.com/sethaxen/PartitionedDistributions.jl/issues/20
                # z ~ Normal()
            end
            y = randn(2)
            z = randn(2)
            model = array_pw(y, z)
            chn = sample_from_prior(model, 50; make_chain=true)

            plds = DynamicPPL.pointwise_logdensities(model, chn)
            @test plds[@varname(x)] == logpdf.(Ref(MvNormal(zeros(2), I)), chn[@varname(x)])
            @test plds[@varname(y)] == logpdf.(MvNormal.(chn[@varname(x)], Ref(I)), Ref(y))

            plls = DynamicPPL.pointwise_loglikelihoods(model, chn)
            @test plls[@varname(y)] == logpdf.(MvNormal.(chn[@varname(x)], Ref(I)), Ref(y))
            @test !haskey(plls, @varname(x))

            pplds = DynamicPPL.pointwise_prior_logdensities(model, chn)
            @test pplds[@varname(x)] ==
                logpdf.(Ref(MvNormal(zeros(2), I)), chn[@varname(x)])
            @test !haskey(pplds, @varname(y))

            plds = DynamicPPL.pointwise_logdensities(model, chn; factorize=true)
            for (x_pld, y_pld, x_val) in
                zip(plds[@varname(x)], plds[@varname(y)], chn[@varname(x)])
                @test x_pld isa Vector{<:Real}
                @test length(x_pld) == 2
                @test x_pld[1] == logpdf(Normal(), x_val[1])
                @test x_pld[2] == logpdf(Normal(), x_val[2])
                @test y_pld isa Vector{<:Real}
                @test length(y_pld) == 2
                @test y_pld[1] == logpdf(Normal(x_val[1], 1), y[1])
                @test y_pld[2] == logpdf(Normal(x_val[2], 1), y[2])
            end

            plls = DynamicPPL.pointwise_loglikelihoods(model, chn; factorize=true)
            for (y_pll, x_val) in zip(plls[@varname(y)], chn[@varname(x)])
                @test y_pll isa Vector{<:Real}
                @test length(y_pll) == 2
                @test y_pll[1] == logpdf(Normal(x_val[1], 1), y[1])
                @test y_pll[2] == logpdf(Normal(x_val[2], 1), y[2])
            end

            pplds = DynamicPPL.pointwise_prior_logdensities(model, chn; factorize=true)
            for (x_ppld, x_val) in zip(pplds[@varname(x)], chn[@varname(x)])
                @test x_ppld isa Vector{<:Real}
                @test length(x_ppld) == 2
                @test x_ppld[1] == logpdf(Normal(), x_val[1])
                @test x_ppld[2] == logpdf(Normal(), x_val[2])
            end
        end
    end

    @testset "returned" begin
        @model function f()
            x ~ Normal()
            y ~ MvNormal(zeros(2), I)
            return x + y[1] + y[2]
        end
        model = f()
        chn = sample_from_prior(model, 100; make_chain=true)
        expected_rtnd = chn[@varname(x)] .+ chn[@varname(y[1])] .+ chn[@varname(y[2])]

        rtnd = returned(model, chn)
        @test isapprox(rtnd, expected_rtnd)
        @test rtnd isa DD.DimMatrix
        @test parent(parent(DD.dims(rtnd, :iter))) == FlexiChains.iter_indices(chn)
        @test parent(parent(DD.dims(rtnd, :chain))) == FlexiChains.chain_indices(chn)

        @testset "works even for dists that hasvalue isn't implemented for" begin
            @model function f_product()
                return x ~ product_distribution((; a=Normal()))
            end
            model = f_product()
            chn = sample_from_prior(model, 100; make_chain=true)
            rets = returned(f_product(), chn)
            @test chn[@varname(x)] == rets
        end

        @testset "errors on missing variables" begin
            @model function xonly()
                return x ~ Normal()
            end
            @model function xy()
                x ~ Normal()
                return y ~ Normal()
            end
            chn = sample_from_prior(xonly(), 100; make_chain=true)
            @test_throws "not found in chain" returned(xy(), chn)
        end

        @testset "stacks DimArray return values" begin
            @model function return_dimarray()
                x ~ Normal()
                return DD.DimArray(randn(2, 3), (:a, :b))
            end
            chn = sample_from_prior(return_dimarray(), 50; make_chain=true)
            rets = returned(return_dimarray(), chn)
            @test rets isa DD.DimArray{T,4} where {T}
            @test size(rets) == (50, 1, 2, 3)
            @test DD.name.(DD.dims(rets)) == (:iter, :chain, :a, :b)
        end

        @testset "with non-standard Array variables" begin
            # This essentially tests that templates are correctly used when calling
            # returned()
            @model function offset()
                x = OffsetArray(zeros(2), -2:-1)
                # Don't sample all elements of `x` to prevent it from being densified,
                # thus bypassing the code that we want to check.
                x[-2] ~ Normal()
                return first(x)
            end
            model = offset()
            chn = sample_from_prior(model, 50; make_chain=true)
            rets = returned(model, chn)
            @test rets == chn[@varname(x[-2])]
        end
    end

    @testset "predict" begin
        @model function f()
            # By default, FlexiChains will store `m` as a single variable. However, this
            # also lets us check behaviour after splitting up VarNames (i.e., if the chain
            # has m[1] and m[2] but the model has m).
            m ~ MvNormal(zeros(2), I)
            # Same but with dot tilde; on DPPL v0.40 onwards, the model will have p[1] and
            # p[2] but since the VNT is densified before chain construction, the chain will
            # have p.
            p = zeros(2)
            p .~ Normal()
            # Then some normal parameters.
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = f() | (; y=4.0)

        # Sanity check
        chn = sample_from_posterior(StableRNG(468), model)
        @test isapprox(mean(chn[@varname(x)]), 2.0; atol=0.1)
        @test isapprox(mean(chn[@varname(m[1])]), 0.0; atol=0.1)
        @test isapprox(mean(chn[@varname(m[2])]), 0.0; atol=0.1)
        @test isapprox(mean(chn[@varname(p[1])]), 0.0; atol=0.1)
        @test isapprox(mean(chn[@varname(p[2])]), 0.0; atol=0.1)

        @testset "chain values are actually used" begin
            pdns = predict(StableRNG(468), f(), chn)
            # Sanity check.
            @test pdns[@varname(x)] == chn[@varname(x)]
            @test pdns[@varname(m)] == chn[@varname(m)]
            @test pdns[@varname(p)] == chn[@varname(p)]
            # Since the model was conditioned with y = 4.0, we should
            # expect that the chain's mean of x is approx 2.0.
            # So the posterior predictions for y should be centred on
            # 2.0 (ish).
            @test isapprox(mean(pdns[@varname(y)]), 2.0; atol=0.1)
        end

        @testset "logp" begin
            pdns = predict(f(), chn)
            # Since we deconditioned `y`, there are no likelihood terms.
            @test all(iszero, pdns[FlexiChains._LOGLIKELIHOOD_KEY])
            # The logprior should be the same as that of the original chain, but
            # with an extra term for y ~ Normal(x)
            chn_logprior = chn[FlexiChains._LOGPRIOR_KEY]
            pdns_logprior = pdns[FlexiChains._LOGPRIOR_KEY]
            expected_diff = logpdf.(Normal.(chn[@varname(x)]), pdns[@varname(y)])
            @test isapprox(pdns_logprior, chn_logprior .+ expected_diff)
            # Logjoint should be the same as logprior
            @test pdns[FlexiChains._LOGJOINT_KEY] == pdns[FlexiChains._LOGPRIOR_KEY]
        end

        @testset "non-parameter keys are preserved" begin
            pdns = predict(f(), chn)
            # Check that the only new thing added was the prediction for y.
            @test only(setdiff(Set(keys(pdns)), Set(keys(chn)))) == Parameter(@varname(y))
            # Check that no other keys originally in `chn` were removed.
            @test isempty(setdiff(Set(keys(chn)), Set(keys(pdns))))
        end

        @testset "include_all=false" begin
            pdns = predict(f(), chn; include_all=false)
            # Check that the only parameter in the chain is the prediction for y.
            @test only(Set(FlexiChains.parameters(pdns))) == @varname(y)
        end

        @testset "indices are preserved" begin
            pdns = predict(f(), chn)
            @test FlexiChains.iter_indices(pdns) == FlexiChains.iter_indices(chn)
            @test FlexiChains.chain_indices(pdns) == FlexiChains.chain_indices(chn)
        end

        @testset "no sampling time and sampler state" begin
            # it just doesn't really make sense for the predictions to carry those
            # information
            pdns = predict(f(), chn)
            @test all(ismissing, FlexiChains.sampling_time(pdns))
            @test all(ismissing, FlexiChains.last_sampler_state(pdns))
        end

        @testset "rng is respected" begin
            pdns1 = predict(Xoshiro(468), f(), chn)
            pdns2 = predict(Xoshiro(468), f(), chn)
            @test FlexiChains.has_same_data(pdns1, pdns2)
            pdns3 = predict(Xoshiro(469), f(), chn)
            @test !FlexiChains.has_same_data(pdns1, pdns3)
        end

        @testset "with non-standard Array variables" begin
            # This essentially tests that templates are correctly used when calling
            # predict().
            @model function offset2()
                x = OffsetArray(zeros(2), -2:-1)
                # Don't sample all elements of `x` to prevent it from being densified,
                # thus bypassing the code that we want to check.
                x[-2] ~ Normal()
                return y ~ Normal(x[-2])
            end
            cond_model = offset2() | (; y=2.0)
            chn = sample_from_posterior(StableRNG(468), cond_model)
            @test mean(chn[@varname(x[-2])]) ≈ 1.0 atol = 0.05
            pdns = predict(StableRNG(468), offset2(), chn)
            @test pdns[@varname(x[-2])] == chn[@varname(x[-2])]
            @test mean(pdns[@varname(y)]) ≈ 1.0 atol = 0.05
        end
    end

    @testset "Models with variable-length parameters" begin
        # These tests are mainly to check the interaction of VarNamedTuple with chains.
        @testset "single variable" begin
            @model function varlen_single()
                n ~ DiscreteUniform(2, 5)
                x ~ MvNormal(zeros(n), I)
                y ~ Normal(sum(x))
                return prod(x)
            end
            cond_model = varlen_single() | (; y=1.0)
            chn = sample_from_prior(cond_model, 100; make_chain=true)
            # Sanity check
            @test chn[@varname(n)] == length.(chn[@varname(x)])
            # Check that returned and predict both work. For returned we can also
            # check correctness, but for predict we just check that it runs.
            @test isapprox(returned(cond_model, chn), prod.(chn[@varname(x)]))
            pdns = predict(varlen_single(), chn)
            @test pdns isa VNChain
            for vn in FlexiChains.parameters(chn)
                @test pdns[vn] == chn[vn]
            end
            @test @varname(y) in FlexiChains.parameters(pdns)
        end

        @testset "dense vector" begin
            # For this model, `x` should still be represented in the chain as a single
            # variable, since the PartialArray will get densified.
            @model function varlen_dense()
                n ~ DiscreteUniform(2, 5)
                x = zeros(n)
                x .~ Normal()
                y ~ Normal(sum(x))
                return prod(x)
            end
            cond_model = varlen_dense() | (; y=1.0)
            chn = sample_from_prior(cond_model, 100; make_chain=true)
            # Sanity check
            @test chn[@varname(n)] == length.(chn[@varname(x)])
            # Check that returned and predict both work. For returned we can also
            # check correctness, but for predict we just check that it runs.
            @test isapprox(returned(cond_model, chn), prod.(chn[@varname(x)]))
            pdns = predict(varlen_dense(), chn)
            @test pdns isa VNChain
            for vn in FlexiChains.parameters(chn)
                @test pdns[vn] == chn[vn]
            end
            @test @varname(y) in FlexiChains.parameters(pdns)
        end

        @testset "nondense (sparse?) vector" begin
            # For this model, `x` will be broken up in the chain, because not
            # all entries in the PartialArray are filled
            @model function varlen_nondense()
                n ~ DiscreteUniform(2, 5)
                x = zeros(n + 2)
                for i in 1:n
                    x[i] ~ Normal()
                end
                y ~ Normal(sum(x[1:n]))
                return prod(x[1:n])
            end
            cond_model = varlen_nondense() | (; y=1.0)
            chn = sample_from_prior(cond_model, 100; make_chain=true)
            # Check that returned and predict both work.
            @test returned(cond_model, chn) isa DD.DimArray
            pdns = predict(varlen_nondense(), chn)
            @test pdns isa VNChain
            for vn in FlexiChains.parameters(chn)
                @test isequal(pdns[vn], chn[vn]) # might have missing so need isequal
            end
            @test @varname(y) in FlexiChains.parameters(pdns)
        end
    end

    @testset "PosteriorStats.loo" begin
        @testset "no factorisation" begin
            @model function f(y)
                x ~ Normal()
                return y .~ Normal(x)
            end
            model = f(randn(10))
            chain = sample_from_prior(model, 500, 3; make_chain=true)
            result = PosteriorStats.loo(model, chain)
            @test result.param_names == [@varname(y[i]) for i in 1:10]
            @test result.loo isa PosteriorStats.PSISLOOResult
        end

        @testset "factorize kwarg" begin
            @model function farray(y)
                x ~ MvNormal(zeros(2), I)
                return y ~ MvNormal(x, I)
            end
            model = farray(randn(2))
            chain = sample_from_prior(model, 500, 3; make_chain=true)
            result = PosteriorStats.loo(model, chain; factorize=true)
            @test result.param_names == [@varname(y[i]) for i in 1:2]
            @test result.loo isa PosteriorStats.PSISLOOResult

            result = PosteriorStats.loo(model, chain; factorize=false)
            @test result.param_names == [@varname(y)]
            @test result.loo isa PosteriorStats.PSISLOOResult
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
