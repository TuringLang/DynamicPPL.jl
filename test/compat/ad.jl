@testset "ad.jl" begin
    @testset "logp" begin
        # Hand-written log probabilities for vector `x = [s, m]`.
        function logp_gdemo_default(x)
            s = x[1]
            m = x[2]
            dist = Normal(m, sqrt(s))

            return logpdf(InverseGamma(2, 3), s) +
                   logpdf(Normal(0, sqrt(s)), m) +
                   logpdf(dist, 1.5) +
                   logpdf(dist, 2.0)
        end

        test_model_ad(gdemo_default, logp_gdemo_default)

        @model function wishart_ad()
            return v ~ Wishart(7, [1 0.5; 0.5 1])
        end

        # Hand-written log probabilities for `x = [v]`.
        function logp_wishart_ad(x)
            dist = Wishart(7, [1 0.5; 0.5 1])
            return logpdf(dist, reshape(x, 2, 2))
        end

        test_model_ad(wishart_ad(), logp_wishart_ad)
    end
end
