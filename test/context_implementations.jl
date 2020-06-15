@testset "context_implementations.jl" begin
    # https://github.com/TuringLang/DynamicPPL.jl/issues/129
    @testset "#129" begin
        @model function test(x)
            μ ~ MvNormal(fill(0, 2), 2.0)
            z = Vector{Int}(undef, length(x))
            z .~ Categorical.(fill([0.5, 0.5], length(x)))
            for i in 1:length(x)
                x[i] ~ Normal(μ[z[i]], 0.1)
            end
        end

        test([1, 1, -1])(VarInfo(), SampleFromPrior(), LikelihoodContext())
    end
end
