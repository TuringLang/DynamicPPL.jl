using Bijectors: pd_from_upper, pd_from_lower

pd_from_triangular(X::AbstractMatrix, uplo::Char) = uplo == 'U' ? pd_from_upper(X) : pd_from_lower(X)

@model lkj_prior_demo() = x ~ LKJ(2, 1)
@model lkj_chol_prior_demo(uplo) = x ~ LKJCholesky(2, 1, uplo)

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

@testset "Sample from x ~ LKJCholesky(2, 1, $(uplo))" for uplo in ['U', 'L']
    model = lkj_chol_prior_demo(uplo)
    # `SampleFromPrior` will sample in unconstrained space.
    @testset "SampleFromPrior" begin
        samples = sample(model, SampleFromPrior(), 1_000)
        # Build correlation matrix from factor
        corr_matrices = map(samples) do s
            M = float.(reshape(s.metadata.vals, (2, 2)))
            pd_from_triangular(M, uplo)
        end
        @test vec(mean(corr_matrices)) ≈ target_mean atol = _lkj_atol
    end

    # `SampleFromUniform` will sample in unconstrained space.
    @testset "SampleFromUniform" begin
        samples = sample(model, SampleFromUniform(), 1_000)
        # Build correlation matrix from factor
        corr_matrices = map(samples) do s
            M = float.(reshape(s.metadata.vals, (2, 2)))
            pd_from_triangular(M, uplo)
        end
        @test vec(mean(corr_matrices)) ≈ target_mean atol = _lkj_atol
    end
end
