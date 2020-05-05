using DynamicPPL: TypedVarInfo

function test_model_ad(model, logp_manual)
    vi = TypedVarInfo(model)
    model(vi, SampleFromPrior())
    x = DynamicPPL.getall(vi)

    # Log probabilities using the model.
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
    @test back(1)[1] ≈ grad
end
