@testset "loglikelihoods.jl" begin
    for m in DynamicPPL.TestUtils.demo_models
        vi = VarInfo(m)

        vns = vi.metadata.m.vns
        if length(vns) == 1 && length(vi[vns[1]]) == 1
            # Only have one latent variable.
            DynamicPPL.setval!(vi, [1.0], ["m"])
        else
            DynamicPPL.setval!(vi, [1.0, 1.0], ["m[1]", "m[2]"])
        end

        lls = pointwise_loglikelihoods(m, vi)

        if isempty(lls)
            # One of the models with literal observations, so we just skip.
            continue
        end

        loglikelihood = if length(keys(lls)) == 1 && length(m.args.x) == 1
            # Only have one observation, so we need to double it
            # for comparison with other models.
            2 * sum(lls[first(keys(lls))])
        else
            sum(sum, values(lls))
        end

        @test loglikelihood ≈ -324.45158270528947
    end
end
