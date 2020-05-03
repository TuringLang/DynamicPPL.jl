using DynamicPPL
using Distributions

using ForwardDiff
using Zygote
using Tracker

@testset "logp" begin
    @model function admodel()
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        1.5 ~ Normal(m, sqrt(s))
        2.0 ~ Normal(m, sqrt(s))
        return s, m
    end

    model = admodel()
    vi = VarInfo(model)
    model(vi, SampleFromPrior())
    x = [vi[@varname(s)], vi[@varname(m)]]

    dist_s = InverseGamma(2,3)

    # Hand-written log probabilities for vector `x = [s, m]`.
    function logp_manual(x)
        s = x[1]
        m = x[2]
        dist = Normal(m, sqrt(s))

        return logpdf(InverseGamma(2, 3), s) + logpdf(Normal(0, sqrt(s)), m) +
            logpdf(dist, 1.5) + logpdf(dist, 2.0)
    end

    # Log probabilities for vector `x = [s, m]` using the model.
    function logp_model(x)
        new_vi = VarInfo(vi, SampleFromPrior(), x)
        model(new_vi, SampleFromPrior())
        return getlogp(new_vi)
    end

    # Check that both functions return the same values.
    lp = logp_manual(x)
    @test logp_model(x) ≈ lp

    # Gradients based on the manual implementation.
    grad = ForwardDiff.gradient(logp_manual, x)

    y, back = Tracker.forward(logp_manual, x)
    @test Tracker.data(y) ≈ lp
    @test Tracker.data(back(1)[1]) ≈ grad

    y, back = Zygote.pullback(logp_manual, x)
    @test y ≈ lp
    @test back(1)[1] ≈ grad

    # Gradients based on the model.
    @test ForwardDiff.gradient(logp_model, x) ≈ grad

    y, back = Tracker.forward(logp_model, x)
    @test Tracker.data(y) ≈ lp
    @test Tracker.data(back(1)[1]) ≈ grad

    y, back = Zygote.pullback(logp_model, x)
    @test y ≈ lp
    @test back(1) ≈ grad
end

