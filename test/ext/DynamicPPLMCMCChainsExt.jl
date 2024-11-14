@testset "DynamicPPLMCMCChainsExt" begin
    @model demo() = x ~ Normal()
    model = demo()

    chain = MCMCChains.Chains(randn(1000, 2, 1), [:x, :y], Dict(:internals => [:y]))
    chain_generated = @test_nowarn generated_quantities(model, chain)
    @test size(chain_generated) == (1000, 1)
    @test mean(chain_generated) ≈ 0 atol = 0.1
end

@testset "predict" begin
    DynamicPPL.Random.seed!(100)

    @model function linear_reg(x, y, σ=0.1)
        β ~ Normal(0, 1)

        for i in eachindex(y)
            y[i] ~ Normal(β * x[i], σ)
        end
    end

    @model function linear_reg_vec(x, y, σ=0.1)
        β ~ Normal(0, 1)
        return y ~ MvNormal(β .* x, σ^2 * I)
    end

    f(x) = 2 * x + 0.1 * randn()

    Δ = 0.1
    xs_train = 0:Δ:10
    ys_train = f.(xs_train)
    xs_test = [10 + Δ, 10 + 2 * Δ]
    ys_test = f.(xs_test)

    # Infer
    m_lin_reg = linear_reg(xs_train, ys_train)
    chain_lin_reg = sample(
        DynamicPPL.LogDensityFunction(m_lin_reg, DynamicPPL.VarInfo(m_lin_reg)),
        AdvancedHMC.NUTS(0.65),
        200;
        chain_type=MCMCChains.Chains,
        param_names=[:β],
    )

    # Predict on two last indices
    m_lin_reg_test = linear_reg(xs_test, fill(missing, length(ys_test)))
    predictions = DynamicPPL.predict(m_lin_reg_test, chain_lin_reg)

    ys_pred = vec(mean(Array(group(predictions, :y)); dims=1))

    @test sum(abs2, ys_test - ys_pred) ≤ 0.1

    # Ensure that `rng` is respected
    predictions1 = let rng = MersenneTwister(42)
        DynamicPPL.predict(rng, m_lin_reg_test, chain_lin_reg[1:2])
    end
    predictions2 = let rng = MersenneTwister(42)
        DynamicPPL.predict(rng, m_lin_reg_test, chain_lin_reg[1:2])
    end
    @test all(Array(predictions1) .== Array(predictions2))

    # Predict on two last indices for vectorized
    m_lin_reg_test = linear_reg_vec(xs_test, missing)
    predictions_vec = DynamicPPL.predict(m_lin_reg_test, chain_lin_reg)
    ys_pred_vec = vec(mean(Array(group(predictions_vec, :y)); dims=1))

    @test sum(abs2, ys_test - ys_pred_vec) ≤ 0.1

    # Multiple chains
    chain_lin_reg = sample(
        DynamicPPL.LogDensityFunction(m_lin_reg, DynamicPPL.VarInfo(m_lin_reg)),
        AdvancedHMC.NUTS(0.65),
        MCMCThreads(),
        200,
        2;
        chain_type=MCMCChains.Chains,
        param_names=[:β],
    )
    m_lin_reg_test = linear_reg(xs_test, fill(missing, length(ys_test)))
    predictions = DynamicPPL.predict(m_lin_reg_test, chain_lin_reg)

    @test size(chain_lin_reg, 3) == size(predictions, 3)

    for chain_idx in MCMCChains.chains(chain_lin_reg)
        ys_pred = vec(mean(Array(group(predictions[:, :, chain_idx], :y)); dims=1))
        @test sum(abs2, ys_test - ys_pred) ≤ 0.1
    end

    # Predict on two last indices for vectorized
    m_lin_reg_test = linear_reg_vec(xs_test, missing)
    predictions_vec = DynamicPPL.predict(m_lin_reg_test, chain_lin_reg)

    for chain_idx in MCMCChains.chains(chain_lin_reg)
        ys_pred_vec = vec(mean(Array(group(predictions_vec[:, :, chain_idx], :y)); dims=1))
        @test sum(abs2, ys_test - ys_pred_vec) ≤ 0.1
    end

    # https://github.com/TuringLang/Turing.jl/issues/1352
    @model function simple_linear1(x, y)
        intercept ~ Normal(0, 1)
        coef ~ MvNormal(zeros(2), I)
        coef = reshape(coef, 1, size(x, 1))

        mu = vec(intercept .+ coef * x)
        error ~ truncated(Normal(0, 1), 0, Inf)
        return y ~ MvNormal(mu, error^2 * I)
    end

    @model function simple_linear2(x, y)
        intercept ~ Normal(0, 1)
        coef ~ filldist(Normal(0, 1), 2)
        coef = reshape(coef, 1, size(x, 1))

        mu = vec(intercept .+ coef * x)
        error ~ truncated(Normal(0, 1), 0, Inf)
        return y ~ MvNormal(mu, error^2 * I)
    end

    @model function simple_linear3(x, y)
        intercept ~ Normal(0, 1)
        coef = Vector(undef, 2)
        for i in axes(coef, 1)
            coef[i] ~ Normal(0, 1)
        end
        coef = reshape(coef, 1, size(x, 1))

        mu = vec(intercept .+ coef * x)
        error ~ truncated(Normal(0, 1), 0, Inf)
        return y ~ MvNormal(mu, error^2 * I)
    end

    @model function simple_linear4(x, y)
        intercept ~ Normal(0, 1)
        coef1 ~ Normal(0, 1)
        coef2 ~ Normal(0, 1)
        coef = [coef1, coef2]
        coef = reshape(coef, 1, size(x, 1))

        mu = vec(intercept .+ coef * x)
        error ~ truncated(Normal(0, 1), 0, Inf)
        return y ~ MvNormal(mu, error^2 * I)
    end

    x = randn(2, 100)
    y = [1 + 2 * a + 3 * b for (a, b) in eachcol(x)]

    param_names = Dict(
        simple_linear1 => [:intercept, Symbol("coef[1]"), Symbol("coef[2]"), :error],
        simple_linear2 => [:intercept, Symbol("coef[1]"), Symbol("coef[2]"), :error],
        simple_linear3 => [:intercept, Symbol("coef[1]"), Symbol("coef[2]"), :error],
        simple_linear4 => [:intercept, :coef1, :coef2, :error],
    )
    @testset "$model" for model in
                          [simple_linear1, simple_linear2, simple_linear3, simple_linear4]
        m = model(x, y)
        chain = sample(
            DynamicPPL.LogDensityFunction(m, DynamicPPL.VarInfo(m)),
            AdvancedHMC.NUTS(0.65),
            100;
            chain_type=MCMCChains.Chains,
            param_names=param_names[model],
        )
        chain_predict = DynamicPPL.predict(model(x, missing), chain)
        mean_prediction = [mean(chain_predict["y[$i]"].data) for i in 1:length(y)]
        @test mean(abs2, mean_prediction - y) ≤ 1e-3
    end
end
