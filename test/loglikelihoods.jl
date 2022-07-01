@testset "loglikelihoods.jl" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
        vi = VarInfo(m)

        for vn in DynamicPPL.TestUtils.varnames(m)
            if vi[vn] isa Real
                vi = DynamicPPL.setindex!!(vi, 1.0, vn)
            else
                vi = DynamicPPL.setindex!!(vi, ones(size(vi[vn])), vn)
            end
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
