# some functors (#367)
struct MyModel
    a::Int
end
@model function (f::MyModel)(x)
    m ~ Normal(f.a, 1)
    return x ~ Normal(m, 1)
end
struct MyZeroModel end
@model function (::MyZeroModel)(x)
    m ~ Normal(0, 1)
    return x ~ Normal(m, 1)
end

innermost_distribution_type(d::Distribution) = typeof(d)
function innermost_distribution_type(d::Distributions.ReshapedDistribution)
    return innermost_distribution_type(d.dist)
end
function innermost_distribution_type(d::Distributions.Product)
    dists = map(innermost_distribution_type, d.v)
    if any(!=(dists[1]), dists)
        error("Cannot extract innermost distribution type from $d")
    end

    return dists[1]
end

is_typed_varinfo(::DynamicPPL.AbstractVarInfo) = false
is_typed_varinfo(varinfo::DynamicPPL.TypedVarInfo) = true
is_typed_varinfo(varinfo::DynamicPPL.SimpleVarInfo{<:NamedTuple}) = true

const GDEMO_DEFAULT = DynamicPPL.TestUtils.demo_assume_observe_literal()

@testset "model.jl" begin
    @testset "convenience functions" begin
        model = GDEMO_DEFAULT

        # sample from model and extract variables
        vi = VarInfo(model)
        s = vi[@varname(s)]
        m = vi[@varname(m)]

        # extract log pdf of variable object
        lp = getlogp(vi)

        # log prior probability
        lprior = logprior(model, vi)
        @test lprior ≈ logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m)

        # log likelihood
        llikelihood = loglikelihood(model, vi)
        @test llikelihood ≈ loglikelihood(Normal(m, sqrt(s)), [1.5, 2.0])

        # log joint probability
        ljoint = logjoint(model, vi)
        @test ljoint ≈ lprior + llikelihood
        @test ljoint ≈ lp

        #### logprior, logjoint, loglikelihood for MCMC chains ####
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            N = 200
            chain = make_chain_from_prior(model, N)
            logpriors = logprior(model, chain)
            loglikelihoods = loglikelihood(model, chain)
            logjoints = logjoint(model, chain)

            # Construct mapping of varname symbols to varname-parent symbols.
            # Here, varname_leaves is used to ensure compatibility with the
            # variables stored in the chain
            var_info = VarInfo(model)
            chain_sym_map = Dict{Symbol,Symbol}()
            for vn_parent in keys(var_info)
                sym = DynamicPPL.getsym(vn_parent)
                vn_children = DynamicPPL.varname_leaves(vn_parent, var_info[vn_parent])
                for vn_child in vn_children
                    chain_sym_map[Symbol(vn_child)] = sym
                end
            end

            # compare them with true values
            for i in 1:N
                samples_dict = Dict()
                for chain_key in keys(chain)
                    value = chain[i, chain_key, 1]
                    key = chain_sym_map[chain_key]
                    existing_value = get(samples_dict, key, Float64[])
                    push!(existing_value, value)
                    samples_dict[key] = existing_value
                end
                samples = (; samples_dict...)
                samples = modify_value_representation(samples) # `modify_value_representation` defined in test/test_util.jl
                @test logpriors[i] ≈
                    DynamicPPL.TestUtils.logprior_true(model, samples[:s], samples[:m])
                @test loglikelihoods[i] ≈ DynamicPPL.TestUtils.loglikelihood_true(
                    model, samples[:s], samples[:m]
                )
                @test logjoints[i] ≈
                    DynamicPPL.TestUtils.logjoint_true(model, samples[:s], samples[:m])
            end
        end
    end

    @testset "model conditioning with various arguments" begin
        @model function demo_condition()
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = demo_condition()
        # Test that different syntaxes work and give the same underlying ConditionContext
        @testset "NamedTuple ConditionContext" begin
            expected_values = (y=2,)
            @test condition(model, (y=2,)).context.values == expected_values
            @test condition(model; y=2).context.values == expected_values
            @test condition(model; y=2).context.values == expected_values
            @test (model | (y=2,)).context.values == expected_values
        end
        @testset "AbstractDict ConditionContext" begin
            expected_values = Dict(@varname(y) => 2)
            @test condition(model, Dict(@varname(y) => 2)).context.values == expected_values
            @test condition(model, @varname(y) => 2).context.values == expected_values
            @test (model | (@varname(y) => 2,)).context.values == expected_values
        end
    end

    @testset "DynamicPPL#684: threadsafe evaluation with multiple types" begin
        @model function multiple_types(x)
            ns ~ filldist(Normal(0, 2.0), 3)
            m ~ Uniform(0, 1)
            return x ~ Normal(m, 1)
        end
        model = multiple_types(1)
        chain = make_chain_from_prior(model, 10)
        loglikelihood(model, chain)
        logprior(model, chain)
        logjoint(model, chain)
    end

    @testset "rng" begin
        model = GDEMO_DEFAULT

        for sampler in (SampleFromPrior(), SampleFromUniform())
            for i in 1:10
                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.default_rng(), vi, sampler)
                vals = DynamicPPL.getall(vi)

                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.default_rng(), vi, sampler)
                @test DynamicPPL.getall(vi) == vals
            end
        end
    end

    @testset "defaults without VarInfo, Sampler, and Context" begin
        model = GDEMO_DEFAULT

        Random.seed!(100)
        retval = model()

        Random.seed!(100)
        retval2 = model(Random.default_rng())
        @test retval2.s == retval.s
        @test retval2.m == retval.m
    end

    @testset "nameof" begin
        @model function test1(x)
            m ~ Normal(0, 1)
            return x ~ Normal(m, 1)
        end
        @model test2(x) = begin
            m ~ Normal(0, 1)
            x ~ Normal(m, 1)
        end
        function test3 end
        @model function (::typeof(test3))(x)
            m ~ Normal(0, 1)
            return x ~ Normal(m, 1)
        end
        function test4 end
        @model function (a::typeof(test4))(x)
            m ~ Normal(0, 1)
            return x ~ Normal(m, 1)
        end

        @test nameof(test1(rand())) == :test1
        @test nameof(test2(rand())) == :test2
        @test nameof(test3(rand())) == :test3
        @test nameof(test4(rand())) == :test4

        # callables
        @test nameof(MyModel(3)(rand())) == Symbol("MyModel(3)")
        @test nameof(MyZeroModel()(rand())) == Symbol("MyZeroModel()")
    end

    @testset "Internal methods" begin
        model = GDEMO_DEFAULT

        # sample from model and extract variables
        vi = VarInfo(model)

        # Second component of return-value of `evaluate!!` should
        # be a `DynamicPPL.AbstractVarInfo`.
        evaluate_retval = DynamicPPL.evaluate!!(model, vi, DefaultContext())
        @test evaluate_retval[2] isa DynamicPPL.AbstractVarInfo

        # Should not return `AbstractVarInfo` when we call the model.
        call_retval = model()
        @test !any(map(x -> x isa DynamicPPL.AbstractVarInfo, call_retval))
    end

    @testset "Dynamic constraints, Metadata" begin
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        spl = SampleFromPrior()
        vi = VarInfo(model, spl, DefaultContext(), DynamicPPL.Metadata())
        link!!(vi, spl, model)

        for i in 1:10
            # Sample with large variations.
            r_raw = randn(length(vi[spl])) * 10
            vi[spl] = r_raw
            @test vi[@varname(m)] == r_raw[1]
            @test vi[@varname(x)] != r_raw[2]
            model(vi)
        end
    end

    @testset "Dynamic constraints, VectorVarInfo" begin
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        for i in 1:10
            vi = VarInfo(model)
            @test vi[@varname(x)] >= vi[@varname(m)]
        end
    end

    @testset "rand" begin
        model = GDEMO_DEFAULT

        Random.seed!(1776)
        s, m = model()
        sample_namedtuple = (; s=s, m=m)
        sample_dict = OrderedDict(@varname(s) => s, @varname(m) => m)

        # With explicit RNG
        @test rand(Random.seed!(1776), model) == sample_namedtuple
        @test rand(Random.seed!(1776), NamedTuple, model) == sample_namedtuple
        @test rand(Random.seed!(1776), Dict, model) == sample_dict

        # Without explicit RNG
        Random.seed!(1776)
        @test rand(model) == sample_namedtuple
        Random.seed!(1776)
        @test rand(NamedTuple, model) == sample_namedtuple
        Random.seed!(1776)
        @test rand(OrderedDict, model) == sample_dict
    end

    @testset "default arguments" begin
        @model test_defaults(x, n=length(x)) = x ~ MvNormal(zeros(n), I)
        @test length(test_defaults(missing, 2)()) == 2
    end

    @testset "missing kwarg" begin
        @model test_missing_kwarg(; x=missing) = x ~ Normal(0, 1)
        @test :x in keys(rand(test_missing_kwarg()))
    end

    @testset "extract priors" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            priors = extract_priors(model)

            # We know that any variable starting with `s` should have `InverseGamma`
            # and any variable starting with `m` should have `Normal`.
            for (vn, prior) in priors
                if DynamicPPL.getsym(vn) == :s
                    @test innermost_distribution_type(prior) <: InverseGamma
                elseif DynamicPPL.getsym(vn) == :m
                    @test innermost_distribution_type(prior) <: Union{Normal,MvNormal}
                else
                    error("Unexpected variable name: $vn")
                end
            end
        end
    end

    @testset "TestUtils" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            x = DynamicPPL.TestUtils.rand_prior_true(model)
            # `rand_prior_true` should return a `NamedTuple`.
            @test x isa NamedTuple

            # `rand` with a `AbstractDict` should have `varnames` as keys.
            x_rand_dict = rand(OrderedDict, model)
            for vn in DynamicPPL.TestUtils.varnames(model)
                @test haskey(x_rand_dict, vn)
            end
            # `rand` with a `NamedTuple` should have `map(Symbol, varnames)` as keys.
            x_rand_nt = rand(NamedTuple, model)
            for vn in DynamicPPL.TestUtils.varnames(model)
                @test haskey(x_rand_nt, Symbol(vn))
            end

            # Ensure log-probability computations are implemented.
            @test logprior(model, x) ≈ DynamicPPL.TestUtils.logprior_true(model, x...)
            @test loglikelihood(model, x) ≈
                DynamicPPL.TestUtils.loglikelihood_true(model, x...)
            @test logjoint(model, x) ≈ DynamicPPL.TestUtils.logjoint_true(model, x...)
            @test logjoint(model, x) !=
                DynamicPPL.TestUtils.logjoint_true_with_logabsdet_jacobian(model, x...)
            # Ensure `varnames` is implemented.
            vi = last(
                DynamicPPL.evaluate!!(
                    model, SimpleVarInfo(OrderedDict()), SamplingContext()
                ),
            )
            @test all(collect(keys(vi)) .== DynamicPPL.TestUtils.varnames(model))
            # Ensure `posterior_mean` is implemented.
            @test DynamicPPL.TestUtils.posterior_mean(model) isa typeof(x)
        end
    end

    @testset "returned() on `LKJCholesky`" begin
        n = 10
        d = 2
        model = DynamicPPL.TestUtils.demo_lkjchol(d)
        xs = [model().x for _ in 1:n]

        # Extract varnames and values.
        vns_and_vals_xs = map(
            collect ∘ Base.Fix1(DynamicPPL.varname_and_value_leaves, @varname(x)), xs
        )
        vns = map(first, first(vns_and_vals_xs))
        vals = map(vns_and_vals_xs) do vns_and_vals
            map(last, vns_and_vals)
        end

        # Construct the chain.
        syms = map(Symbol, vns)
        vns_to_syms = OrderedDict{VarName,Any}(zip(vns, syms))

        chain = MCMCChains.Chains(
            permutedims(stack(vals)), syms; info=(varname_to_symbol=vns_to_syms,)
        )

        # Test!
        results = returned(model, chain)
        for (x_true, result) in zip(xs, results)
            @test x_true.UL == result.x.UL
        end

        # With variables that aren't in the `model`.
        vns_to_syms_with_extra = let d = deepcopy(vns_to_syms)
            d[@varname(y)] = :y
            d
        end
        vals_with_extra = map(enumerate(vals)) do (i, v)
            vcat(v, i)
        end
        chain_with_extra = MCMCChains.Chains(
            permutedims(stack(vals_with_extra)),
            vcat(syms, [:y]);
            info=(varname_to_symbol=vns_to_syms_with_extra,),
        )
        # Test!
        results = returned(model, chain_with_extra)
        for (x_true, result) in zip(xs, results)
            @test x_true.UL == result.x.UL
        end
    end

    if VERSION >= v"1.8"
        @testset "Type stability of models" begin
            models_to_test = [
                DynamicPPL.TestUtils.DEMO_MODELS..., DynamicPPL.TestUtils.demo_lkjchol(2)
            ]
            context = DefaultContext()
            @testset "$(model.f)" for model in models_to_test
                vns = DynamicPPL.TestUtils.varnames(model)
                example_values = DynamicPPL.TestUtils.rand_prior_true(model)
                varinfos = filter(
                    is_typed_varinfo,
                    DynamicPPL.TestUtils.setup_varinfos(model, example_values, vns),
                )
                @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
                    @test begin
                        @inferred(DynamicPPL.evaluate!!(model, varinfo, context))
                        true
                    end

                    varinfo_linked = DynamicPPL.link(varinfo, model)
                    @test begin
                        @inferred(DynamicPPL.evaluate!!(model, varinfo_linked, context))
                        true
                    end
                end
            end
        end
    end

    @testset "values_as_in_model" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            vns = DynamicPPL.TestUtils.varnames(model)
            example_values = DynamicPPL.TestUtils.rand_prior_true(model)
            varinfos = DynamicPPL.TestUtils.setup_varinfos(model, example_values, vns)
            @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
                # We can set the include_colon_eq arg to false because none of
                # the demo models contain :=. The behaviour when
                # include_colon_eq is true is tested in test/compiler.jl
                realizations = values_as_in_model(model, false, varinfo)
                # Ensure that all variables are found.
                vns_found = collect(keys(realizations))
                @test vns ∩ vns_found == vns ∪ vns_found
                # Ensure that the values are the same.
                for vn in vns
                    @test realizations[vn] == varinfo[vn]
                end
            end
        end

        @testset "check that sampling obeys rng if passed" begin
            @model function f()
                x ~ Normal(0)
                return y ~ Normal(x)
            end
            model = f()
            # Call values_as_in_model with the rng
            values = values_as_in_model(Random.Xoshiro(43), model, false)
            # Check that they match the values that would be used if vi was seeded
            # with that seed instead
            expected_vi = VarInfo(Random.Xoshiro(43), model)
            for vn in keys(values)
                @test values[vn] == expected_vi[vn]
            end
        end
    end

    @testset "Erroneous model call" begin
        # Calling a model with the wrong arguments used to lead to infinite recursion, see
        # https://github.com/TuringLang/Turing.jl/issues/2182. This guards against it.
        @model function a_model(x)
            m ~ Normal(0, 1)
            x ~ Normal(m, 1)
            return nothing
        end
        instance = a_model(1.0)
        # `instance` should be called with rng, context, etc., but one may easily get
        # confused and call it the way you are meant to call `a_model`.
        @test_throws MethodError instance(1.0)
    end

    @testset "Product distribution with changing support" begin
        @model function product_dirichlet()
            return x ~ product_distribution(fill(Dirichlet(ones(4)), 2, 3))
        end
        model = product_dirichlet()

        varinfos = [
            DynamicPPL.untyped_varinfo(model),
            DynamicPPL.typed_varinfo(model),
            DynamicPPL.typed_simple_varinfo(model),
            DynamicPPL.untyped_simple_varinfo(model),
        ]
        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            varinfo_linked = DynamicPPL.link(varinfo, model)
            varinfo_linked_result = last(
                DynamicPPL.evaluate!!(model, deepcopy(varinfo_linked), DefaultContext())
            )
            @test getlogp(varinfo_linked) ≈ getlogp(varinfo_linked_result)
        end
    end

    @testset "predict" begin
        @testset "with MCMCChains.Chains" begin
            @model function linear_reg(x, y, σ=0.1)
                β ~ Normal(0, 1)
                for i in eachindex(y)
                    y[i] ~ Normal(β * x[i], σ)
                end
                # Insert a := block to test that it is not included in predictions
                return σ2 := σ^2
            end

            # Construct a chain with 'sampled values' of β
            ground_truth_β = 2
            β_chain = MCMCChains.Chains(rand(Normal(ground_truth_β, 0.002), 1000), [:β])

            # Generate predictions from that chain
            xs_test = [10 + 0.1, 10 + 2 * 0.1]
            m_lin_reg_test = linear_reg(xs_test, fill(missing, length(xs_test)))
            predictions = DynamicPPL.predict(m_lin_reg_test, β_chain)

            # Also test a vectorized model
            @model function linear_reg_vec(x, y, σ=0.1)
                β ~ Normal(0, 1)
                return y ~ MvNormal(β .* x, σ^2 * I)
            end
            m_lin_reg_test_vec = linear_reg_vec(xs_test, missing)

            @testset "variables in chain" begin
                # Note that this also checks that variables on the lhs of :=,
                # such as σ2, are not included in the resulting chain
                @test Set(keys(predictions)) == Set([Symbol("y[1]"), Symbol("y[2]")])
            end

            @testset "accuracy" begin
                ys_pred = vec(mean(Array(group(predictions, :y)); dims=1))
                @test ys_pred[1] ≈ ground_truth_β * xs_test[1] atol = 0.01
                @test ys_pred[2] ≈ ground_truth_β * xs_test[2] atol = 0.01
            end

            @testset "ensure that rng is respected" begin
                rng = MersenneTwister(42)
                predictions1 = DynamicPPL.predict(rng, m_lin_reg_test, β_chain[1:2])
                predictions2 = DynamicPPL.predict(
                    MersenneTwister(42), m_lin_reg_test, β_chain[1:2]
                )
                @test all(Array(predictions1) .== Array(predictions2))
            end

            @testset "accuracy on vectorized model" begin
                predictions_vec = DynamicPPL.predict(m_lin_reg_test_vec, β_chain)
                ys_pred_vec = vec(mean(Array(group(predictions_vec, :y)); dims=1))

                @test ys_pred_vec[1] ≈ ground_truth_β * xs_test[1] atol = 0.01
                @test ys_pred_vec[2] ≈ ground_truth_β * xs_test[2] atol = 0.01
            end

            @testset "prediction from multiple chains" begin
                # Normal linreg model
                multiple_β_chain = MCMCChains.Chains(
                    reshape(rand(Normal(ground_truth_β, 0.002), 1000, 2), 1000, 1, 2), [:β]
                )
                predictions = DynamicPPL.predict(m_lin_reg_test, multiple_β_chain)
                @test size(multiple_β_chain, 3) == size(predictions, 3)

                for chain_idx in MCMCChains.chains(multiple_β_chain)
                    ys_pred = vec(
                        mean(Array(group(predictions[:, :, chain_idx], :y)); dims=1)
                    )
                    @test ys_pred[1] ≈ ground_truth_β * xs_test[1] atol = 0.01
                    @test ys_pred[2] ≈ ground_truth_β * xs_test[2] atol = 0.01
                end

                # Vectorized linreg model
                predictions_vec = DynamicPPL.predict(m_lin_reg_test_vec, multiple_β_chain)

                for chain_idx in MCMCChains.chains(multiple_β_chain)
                    ys_pred_vec = vec(
                        mean(Array(group(predictions_vec[:, :, chain_idx], :y)); dims=1)
                    )
                    @test ys_pred_vec[1] ≈ ground_truth_β * xs_test[1] atol = 0.01
                    @test ys_pred_vec[2] ≈ ground_truth_β * xs_test[2] atol = 0.01
                end
            end
        end

        @testset "with AbstractVector{<:AbstractVarInfo}" begin
            @model function linear_reg(x, y, σ=0.1)
                β ~ Normal(1, 1)
                for i in eachindex(y)
                    y[i] ~ Normal(β * x[i], σ)
                end
            end

            ground_truth_β = 2.0
            # the data will be ignored, as we are generating samples from the prior
            xs_train = 1:0.1:10
            ys_train = ground_truth_β .* xs_train + rand(Normal(0, 0.1), length(xs_train))
            m_lin_reg = linear_reg(xs_train, ys_train)
            chain = [evaluate!!(m_lin_reg)[2] for _ in 1:10000]

            # chain is generated from the prior
            @test mean([chain[i][@varname(β)] for i in eachindex(chain)]) ≈ 1.0 atol = 0.1

            xs_test = [10 + 0.1, 10 + 2 * 0.1]
            m_lin_reg_test = linear_reg(xs_test, fill(missing, length(xs_test)))
            predicted_vis = DynamicPPL.predict(m_lin_reg_test, chain)

            @test size(predicted_vis) == size(chain)
            @test Set(keys(predicted_vis[1])) ==
                Set([@varname(β), @varname(y[1]), @varname(y[2])])
            # because β samples are from the prior, the std will be larger
            @test mean([
                predicted_vis[i][@varname(y[1])] for i in eachindex(predicted_vis)
            ]) ≈ 1.0 * xs_test[1] rtol = 0.1
            @test mean([
                predicted_vis[i][@varname(y[2])] for i in eachindex(predicted_vis)
            ]) ≈ 1.0 * xs_test[2] rtol = 0.1
        end
    end
end
