@model lkj_prior_demo() = x ~ LKJ(2, 1)
@model lkj_chol_prior_demo() = x ~ LKJCholesky(2, 1)

target_mean(::Model{typeof(lkj_prior_demo)}) = vec(mean(LKJ(2, 1)))
# NOTE: Is this an unbiased estimate?
target_mean(::Model{typeof(lkj_chol_prior_demo)}) = vec(cholesky(mean(LKJ(2, 1))).UL)

_lkj_atol(::Model{typeof(lkj_prior_demo)}) = 0.05
# HACK: Need much larger tolerance.
_lkj_atol(::Model{typeof(lkj_chol_prior_demo)}) = 0.25

@testset "$(model.f)" for model in [lkj_prior_demo(), lkj_chol_prior_demo()]
    # `SampleFromPrior` will sample in unconstrained space.
    @testset "SampleFromPrior" begin
        samples = sample(model, SampleFromPrior(), 1_000)
        @test mean(map(Base.Fix2(getindex, Colon()), samples)) ≈ target_mean(model) atol = _lkj_atol(
            model
        )
    end

    # `SampleFromUniform` will sample in unconstrained space.
    @testset "SampleFromUniform" begin
        samples = sample(model, SampleFromUniform(), 1_000)
        @test mean(map(Base.Fix2(getindex, Colon()), samples)) ≈ target_mean(model) atol=_lkj_atol(model)
    end
end
