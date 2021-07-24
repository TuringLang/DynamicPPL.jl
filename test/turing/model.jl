@testset "model.jl" begin
    @testset "setval! & generated_quantities" begin
        @model function demo1(xs, ::Type{TV}=Vector{Float64}) where {TV}
            m = TV(undef, 2)
            for i in 1:2
                m[i] ~ Normal(0, 1)
            end

            for i in eachindex(xs)
                xs[i] ~ Normal(m[1], 1.0)
            end

            return (m,)
        end

        @model function demo2(xs)
            m ~ MvNormal(2, 1.0)

            for i in eachindex(xs)
                xs[i] ~ Normal(m[1], 1.0)
            end

            return (m,)
        end

        xs = randn(3)
        model1 = demo1(xs)
        model2 = demo2(xs)

        chain1 = sample(model1, MH(), 100)
        chain2 = sample(model2, MH(), 100)

        res11 = generated_quantities(model1, MCMCChains.get_sections(chain1, :parameters))
        res21 = generated_quantities(model2, MCMCChains.get_sections(chain1, :parameters))

        res12 = generated_quantities(model1, MCMCChains.get_sections(chain2, :parameters))
        res22 = generated_quantities(model2, MCMCChains.get_sections(chain2, :parameters))

        # Check that the two different models produce the same values for
        # the same chains.
        @test all(res11 .== res21)
        @test all(res12 .== res22)
        # Ensure that they're not all the same (some can be, because rejected samples)
        @test any(res12[1:(end - 1)] .!= res12[2:end])

        test_setval!(model1, MCMCChains.get_sections(chain1, :parameters))
        test_setval!(model2, MCMCChains.get_sections(chain2, :parameters))

        # Next level
        @model function demo3(xs, ::Type{TV}=Vector{Float64}) where {TV}
            m = Vector{TV}(undef, 2)
            for i in 1:length(m)
                m[i] ~ MvNormal(2, 1.0)
            end

            for i in eachindex(xs)
                xs[i] ~ Normal(m[1][1], 1.0)
            end

            return (m,)
        end

        @model function demo4(xs, ::Type{TV}=Vector{Vector{Float64}}) where {TV}
            m = TV(undef, 2)
            for i in 1:length(m)
                m[i] ~ MvNormal(2, 1.0)
            end

            for i in eachindex(xs)
                xs[i] ~ Normal(m[1][1], 1.0)
            end

            return (m,)
        end

        model3 = demo3(xs)
        model4 = demo4(xs)

        chain3 = sample(model3, MH(), 100)
        chain4 = sample(model4, MH(), 100)

        res33 = generated_quantities(model3, MCMCChains.get_sections(chain3, :parameters))
        res43 = generated_quantities(model4, MCMCChains.get_sections(chain3, :parameters))

        res34 = generated_quantities(model3, MCMCChains.get_sections(chain4, :parameters))
        res44 = generated_quantities(model4, MCMCChains.get_sections(chain4, :parameters))

        # Check that the two different models produce the same values for
        # the same chains.
        @test all(res33 .== res43)
        @test all(res34 .== res44)
        # Ensure that they're not all the same (some can be, because rejected samples)
        @test any(res34[1:(end - 1)] .!= res34[2:end])
    end
end
