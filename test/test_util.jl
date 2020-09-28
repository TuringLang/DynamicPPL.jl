function test_model_ad(model, logp_manual)
    vi = VarInfo(model)
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


"""
    test_setval!(model, chain; sample_idx = 1, chain_idx = 1)

Test `setval!` on `model` and `chain`.

Worth noting that this only supports models containing symbols of the forms
`m`, `m[1]`, `m[1, 2]`, not `m[1][1]`, etc.
"""
function test_setval!(model, chain; sample_idx = 1, chain_idx = 1)
    var_info = VarInfo(model)
    spl = SampleFromPrior()
    θ_old = var_info[spl]
    DynamicPPL.setval!(var_info, chain, sample_idx, chain_idx)
    θ_new = var_info[spl]
    @test θ_old != θ_new
    nt = DynamicPPL.tonamedtuple(var_info)
    for (k, (vals, names)) in pairs(nt)
        for (n, v) in zip(names, vals)
            chain_val = if Symbol(n) ∉ keys(chain)
                # Assume it's a group
                vec(MCMCChains.group(chain, Symbol(n)).value[sample_idx, :, chain_idx])
            else
                chain[sample_idx, n, chain_idx]
            end
            @test v == chain_val
        end
    end
end
