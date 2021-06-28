# A collection of models for which the mean-of-means for the posterior should
# be same.
@model function gdemo1(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and `observe`
    m = TV(undef, length(x))
    m .~ Normal()
    return x ~ MvNormal(m, 0.5 * ones(length(x)))
end

@model function gdemo2(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    # `assume` with indexing and `observe`
    m = TV(undef, length(x))
    for i in eachindex(m)
        m[i] ~ Normal()
    end
    return x ~ MvNormal(m, 0.5 * ones(length(x)))
end

@model function gdemo3(x=10 * ones(2))
    # Multivariate `assume` and `observe`
    m ~ MvNormal(length(x), 1.0)
    return x ~ MvNormal(m, 0.5 * ones(length(x)))
end

@model function gdemo4(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and `observe` with indexing
    m = TV(undef, length(x))
    m .~ Normal()
    for i in eachindex(x)
        x[i] ~ Normal(m[i], 0.5)
    end
end

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function gdemo5(x=10 * ones(1))
    # `assume` and `dot_observe`
    m ~ Normal()
    return x .~ Normal(m, 0.5)
end

# @model function gdemo6(::Type{TV} = Vector{Float64}) where {TV}
#     # `assume` and literal `observe`
#     m ~ MvNormal(length(x), 1.0)
#     [10.0, 10.0] ~ MvNormal(m, 0.5 * ones(2))
# end

@model function gdemo7(::Type{TV}=Vector{Float64}) where {TV}
    # `dot_assume` and literal `observe` with indexing
    m = TV(undef, 2)
    m .~ Normal()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end
end

# @model function gdemo8(::Type{TV} = Vector{Float64}) where {TV}
#     # `assume` and literal `dot_observe`
#     m ~ Normal()
#     [10.0, ] .~ Normal(m, 0.5)
# end

@model function _prior_dot_assume(::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, 2)
    m .~ Normal()

    return m
end

@model function gdemo9()
    # Submodel prior
    m = @submodel _prior_dot_assume()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end
end

@model function _likelihood_dot_observe(m, x)
    return x ~ MvNormal(m, 0.5 * ones(length(m)))
end

@model function gdemo10(x=10 * ones(2), ::Type{TV}=Vector{Float64}) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Submodel likelihood
    @submodel _likelihood_dot_observe(m, x)
end

const gdemo_models = (
    gdemo1(), gdemo2(), gdemo3(), gdemo4(), gdemo5(), gdemo7(), gdemo9(), gdemo10()
)

@testset "loglikelihoods.jl" begin
    for m in gdemo_models
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

        args = DynamicPPL.getarguments(m)
        loglikelihood = if length(keys(lls)) == 1 && length(args.x) == 1
            # Only have one observation, so we need to double it
            # for comparison with other models.
            2 * sum(lls[first(keys(lls))])
        else
            sum(sum, values(lls))
        end

        @test loglikelihood ≈ -324.45158270528947
    end
end
