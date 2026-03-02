module DynamicPPLLKJTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Distributions: LKJCholesky, LKJ, mean
using DynamicPPL
using LinearAlgebra
using Test

function pd_from_triangular(X::AbstractMatrix, uplo::Char)
    # Pre-allocation fixes a problem with abstract element types in Julia 1.10
    # Ref https://github.com/TuringLang/DynamicPPL.jl/pull/570#issue-2092729916
    out = similar(X, Base.promote_op(*, eltype(X), eltype(X)))
    if uplo === 'U'
        mul!(out, UpperTriangular(X)', UpperTriangular(X))
    else
        mul!(out, LowerTriangular(X), LowerTriangular(X)')
    end
    return out
end

@model lkj_prior_demo() = x ~ LKJ(2, 1)
@model lkj_chol_prior_demo(uplo) = x ~ LKJCholesky(2, 1, uplo)

# Same for both distributions
target_mean = vec(Matrix{Float64}(I, 2, 2))

n_samples = 1000
_lkj_atol = 0.05

@testset "Sample from x ~ LKJ(2, 1)" begin
    model = lkj_prior_demo()
    for init_strategy in [InitFromPrior(), InitFromUniform()]
        corr_matrices = map(1:n_samples) do _
            accs = DynamicPPL.OnlyAccsVarInfo(RawValueAccumulator(false))
            _, accs = DynamicPPL.init!!(model, accs, init_strategy, UnlinkAll())
            corr_sample = DynamicPPL.get_raw_values(accs)[@varname(x)]
        end
        @test vec(mean(corr_matrices)) ≈ target_mean atol = _lkj_atol
    end
end

@testset "Sample from x ~ LKJCholesky(2, 1, $(uplo))" for uplo in ['U', 'L']
    model = lkj_chol_prior_demo(uplo)
    for init_strategy in [InitFromPrior(), InitFromUniform()]
        corr_matrices = map(1:n_samples) do _
            accs = DynamicPPL.OnlyAccsVarInfo(RawValueAccumulator(false))
            _, accs = DynamicPPL.init!!(model, accs, init_strategy, UnlinkAll())
            chol_sample = DynamicPPL.get_raw_values(accs)[@varname(x)]
            pd_from_triangular(chol_sample.UL.data, uplo)
        end
        @test vec(mean(corr_matrices)) ≈ target_mean atol = _lkj_atol
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
