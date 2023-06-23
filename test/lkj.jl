using Bijectors: pd_from_upper

@model lkj_prior_demo() = x ~ LKJ(2, 1)
@model lkj_chol_prior_demo() = x ~ LKJCholesky(2, 1, 'U')

# Same for both distributions
target_mean = vec(Matrix{Float64}(I, 2, 2))

_lkj_atol = 0.05

@testset "Sample from x ~ LKJ(2, 1)" begin
    model = lkj_prior_demo()
    # `SampleFromPrior` will sample in constrained space.
    @testset "SampleFromPrior" begin
        samples = sample(model, SampleFromPrior(), 1_000)
        @test mean(map(Base.Fix2(getindex, Colon()), samples)) ≈ target_mean atol =
            _lkj_atol
    end

    # `SampleFromUniform` will sample in unconstrained space.
    @testset "SampleFromUniform" begin
        samples = sample(model, SampleFromUniform(), 1_000)
        @test mean(map(Base.Fix2(getindex, Colon()), samples)) ≈ target_mean atol =
            _lkj_atol
    end
end

@testset "Sample from x ~ LKJCholesky(2, 1, U)" begin
    model = lkj_chol_prior_demo()
    # `SampleFromPrior` will sample in unconstrained space.
    @testset "SampleFromPrior" begin
        samples = sample(model, SampleFromPrior(), 1_000)
        # Build correlation matrix from factor
        corr_matrices = map(samples) do s
            M = Float64.(reshape(s.metadata.vals, (2, 2)))
            pd_from_upper(M)
        end
        @test vec(mean(corr_matrices)) ≈ target_mean atol = _lkj_atol
    end

    # `SampleFromUniform` will sample in unconstrained space.
    @testset "SampleFromUniform" begin
        samples = sample(model, SampleFromUniform(), 1_000)
        # Build correlation matrix from factor
        corr_matrices = map(samples) do s
            M = Float64.(reshape(s.metadata.vals, (2, 2)))
            pd_from_upper(M)
        end
        @test vec(mean(corr_matrices)) ≈ target_mean atol = _lkj_atol
    end
end
