# helper functions for testing (both varname_leaves and values_from_chain are in utils.jl, for some reason they are not loaded)
varname_leaves(vn::VarName, ::Real) = [vn]
function varname_leaves(vn::VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (
        VarName(vn, getlens(vn) ∘ Setfield.IndexLens(Tuple(I))) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_leaves(VarName(vn, getlens(vn) ∘ Setfield.IndexLens(Tuple(I))), val[I]) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::DynamicPPL.VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        lens = Setfield.PropertyLens{sym}()
        varname_leaves(vn ∘ lens, get(val, lens))
    end
    return Iterators.flatten(iter)
end

#481
function values_from_chain(x, vn_parent, chain, chain_idx, iteration_idx)
    # HACK: If it's not an array, we fall back to just returning the first value.
    return only(chain[iteration_idx, Symbol(vn_parent), chain_idx])
end
function values_from_chain(
    x::AbstractArray, vn_parent::VarName{sym}, chain, chain_idx, iteration_idx
) where {sym}
    # We use `VarName{sym}()` so that the resulting leaf `vn` only contains the tail of the lens.
    # This way we can use `getlens(vn)` to extract the value from `x` and use `vn_parent ∘ getlens(vn)`
    # to extract the value from the `chain`.
    return reduce(varname_leaves(VarName{sym}(), x); init=similar(x)) do x, vn
        # Update `x`, possibly in place, and return.
        l = AbstractPPL.getlens(vn)
        Setfield.set(
            x,
            BangBang.prefermutation(l),
            chain[iteration_idx, Symbol(vn_parent ∘ l), chain_idx],
        )
    end
end
function values_from_chain(vi::AbstractVarInfo, vn_parent, chain, chain_idx, iteration_idx)
    # Use the value `vi[vn_parent]` to obtain a buffer.
    return values_from_chain(vi[vn_parent], vn_parent, chain, chain_idx, iteration_idx)
end

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
@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
gdemo_default = gdemo_d()
# function to modify the representation of values based on their length
function modify_value_representation(nt::NamedTuple)
    modified_nt = NamedTuple()
    for (key, value) in zip(keys(nt), values(nt))
        if length(value) == 1  # Scalar value
            modified_value = value[1]
        else  # Non-scalar value
            modified_value = value
        end
        modified_nt = merge(modified_nt, (key => modified_value,))
    end
    return modified_nt
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

        #### logprior, logjoint, loglikelihood for MCMC chains ####
        model_no = 0
        for model in DynamicPPL.TestUtils.DEMO_MODELS
            model_no += 1; println("\n model $model_no \n")
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
            println("\n vals_mat: ", size(vals_mat))
            vec_of_vec = [vcat(x...)' for x in eachcol(vals_mat)]
            println("\n vec_of_vec: ", size(vec_of_vec))
            chain_mat = vcat(vec_of_vec...)
            println("\n chain_mat: ", size(chain_mat))

            # devise parameter names for chain
            sample_values_vec = collect(values(vals_OrderedDict[1]))
            symbol_names = []
            chain_sym_map = Dict()
            for k in 1:length(keys(var_info))
                vn_parent = keys(var_info)[k]
                sym = DynamicPPL.getsym(vn_parent)
                vn_children = varname_leaves(vn_parent, sample_values_vec[k])
                for vn_child in vn_children
                    chain_sym_map[Symbol(vn_child)] = sym
                    symbol_names = [symbol_names; Symbol(vn_child)]
                end
            end
            chain = Chains(chain_mat, symbol_names)

            # calculate the pointwise loglikelihoods for the whole chain using the newly written functions
            logpriors = logprior(model, chain)
            loglikelihoods = loglikelihood(model, chain)
            logjoints = logjoint(model, chain)
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
                samples = modify_value_representation(samples)
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

    @testset "rng" begin
        model = gdemo_default

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
        model = gdemo_default

        Random.seed!(100)
        s, m = model()

        Random.seed!(100)
        @test model(Random.default_rng()) == (s, m)
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

    @testset "default arguments" begin
        @model test_defaults(x, n=length(x)) = x ~ MvNormal(zeros(n), I)
        @test length(test_defaults(missing, 2)()) == 2
    end
end
