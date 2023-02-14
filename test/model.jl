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

@testset "model.jl" begin
    @testset "convenience functions" begin
        model = gdemo_default

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

        # logprior, logjoint, loglikelihood for MCMC chains 
        for model in DynamicPPL.TestUtils.DEMO_MODELS
            var_info = VarInfo(model)
            vns = DynamicPPL.TestUtils.varnames(model)
            syms = unique(DynamicPPL.getsym.(vns))
            # generate a chain of sample parameter values.
            N = 200
            vals_OrderedDict = mapreduce(hcat, 1:N) do _
                rand(OrderedDict, model)
            end
            vals_mat = mapreduce(hcat, 1:N) do i
                [vals_OrderedDict[i][vn] for vn in vns]
            end
            vec_of_vec = [vcat(x...)' for x in eachcol(vals_mat)]
            chain_mat = vcat(vec_of_vec...)
            # devise parameter names for chain
            symbol_names = Symbol[]
            if size(chain_mat, 2) != length(keys(var_info)) # some parameter names need to be splatted
                # examine each vn in vns, and create splatted new variable symbol_names.
                for (key, val) in vals_OrderedDict[1]
                    if length(val) > 1
                        for kk in 1:length(val)
                            push!(symbol_names, Symbol(key, "[", kk, "]"))
                        end
                    else
                        push!(symbol_names, Symbol(key))
                    end
                end
            else
                symbol_names = keys(var_info)
            end
            if model isa
                Model{typeof(DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix)}
                symbol_names = [
                    Symbol(vns[k], "[", kk, "]") for k in 1:length(vns) for
                    kk in 1:size(vals_mat[k, 1], 1)
                ]
            end
            chain = Chains(chain_mat, symbol_names)
            # count repeatitions of parameter names in keys(chain), for laster use in constructing samples_dict in tests below.
            reps = Dict(
                sym => count(i -> contains(String(i), String(sym)), keys(chain)) for
                sym in syms
            )
            # calculate the pointwise loglikelihoods for the whole chain
            logpriors = logprior(model, chain)
            loglikelihoods = loglikelihood(model, chain)
            logjoints = logjoint(model, chain)
            # compare them with true values
            for i in 1:N
                # extract parameter values from chain: we need to aggregate the values belonging to the same parameter into a vector. 
                samples_dict = Dict()
                for sym in syms
                    if reps[sym] > 1 # collect all the values from chain which belong to the same parameter
                        chain_param_names = [
                            key for key in keys(chain) if contains(String(key), String(sym))
                        ]
                        samples_dict[sym] = [
                            chain[i, chain_param_name, 1] for
                            chain_param_name in chain_param_names
                        ]
                    else
                        samples_dict[sym] = chain[i, Symbol(sym), 1]
                    end
                end
                samples = (; samples_dict...)
                @test logpriors[i] ≈ DynamicPPL.TestUtils.logprior_true(
                    model, [samples[sym] for sym in syms]...
                )
                @test loglikelihoods[i] ≈ DynamicPPL.TestUtils.loglikelihood_true(
                    model, [samples[sym] for sym in syms]...
                )
                @test logjoints[i] ≈ DynamicPPL.TestUtils.logjoint_true(
                    model, [samples[sym] for sym in syms]...
                )
            end
        end
    end

    @testset "rng" begin
        model = gdemo_default

        for sampler in (SampleFromPrior(), SampleFromUniform())
            for i in 1:10
                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.GLOBAL_RNG, vi, sampler)
                vals = DynamicPPL.getall(vi)

                Random.seed!(100 + i)
                vi = VarInfo()
                model(Random.GLOBAL_RNG, vi, sampler)
                @test DynamicPPL.getall(vi) == vals
            end
        end
    end

    @testset "defaults without VarInfo, Sampler, and Context" begin
        model = gdemo_default

        Random.seed!(100)
        s, m = model()

        Random.seed!(100)
        @test model(Random.GLOBAL_RNG) == (s, m)
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
        model = gdemo_default

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

    @testset "Dynamic constraints" begin
        model = DynamicPPL.TestUtils.demo_dynamic_constraint()
        vi = VarInfo(model)
        spl = SampleFromPrior()
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

    @testset "rand" begin
        model = gdemo_default

        Random.seed!(1776)
        s, m = model()
        sample_namedtuple = (; s=s, m=m)
        sample_dict = Dict(@varname(s) => s, @varname(m) => m)

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
        @test rand(Dict, model) == sample_dict
    end
end
