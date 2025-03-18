@testset "values_as_in_model" begin
    @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        vns = DynamicPPL.TestUtils.varnames(model)
        example_values = DynamicPPL.TestUtils.rand_prior_true(model)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(model, example_values, vns)
        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            # We can set the include_colon_eq arg to false because none of
            # the demo models contain :=. The behaviour when
            # include_colon_eq is true is tested in test/compiler.jl
            realizations = values_as_in_model(model, false, varinfo)
            # Ensure that all variables are found.
            vns_found = collect(keys(realizations))
            @test vns ∩ vns_found == vns ∪ vns_found
            # Ensure that the values are the same.
            for vn in vns
                @test realizations[vn] == varinfo[vn]
            end
        end
    end

    @testset "support for :=" begin
        @model function demo_tracked()
            x ~ Normal()
            y := 100 + x
            return (; x, y)
        end
        @model function demo_tracked_submodel()
            return vals ~ to_submodel(demo_tracked(), false)
        end

        for model in [demo_tracked(), demo_tracked_submodel()]
            values = values_as_in_model(model, true, VarInfo(model))
            @test haskey(values, @varname(x))
            @test haskey(values, @varname(y))

            values = values_as_in_model(model, false, VarInfo(model))
            @test haskey(values, @varname(x))
            @test !haskey(values, @varname(y))
        end
    end

    @testset "Prefixing" begin
        @model inner() = x ~ Normal()

        @model function outer_auto_prefix()
            a ~ to_submodel(inner(), true)
            b ~ to_submodel(inner(), true)
            return nothing
        end
        @model function outer_manual_prefix()
            a ~ to_submodel(prefix(inner(), :a), false)
            b ~ to_submodel(prefix(inner(), :b), false)
            return nothing
        end

        for model in (outer_auto_prefix(), outer_manual_prefix())
            vi = VarInfo(model)
            vns = Set(keys(values_as_in_model(model, false, vi)))
            @test vns == Set([@varname(var"a.x"), @varname(var"b.x")])
        end
    end

    @testset "Track only specific varnames" begin
        @model function track_specific()
            x = Vector{Float64}(undef, 2)
            # Include a vector x to test for correct subsumption behaviour
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            y ~ Normal(x[1], 1)
            return z := sum(x)
        end

        model = track_specific()
        vi = VarInfo(model)

        # Specify varnames to be tracked directly as an argument to `values_as_in_model`
        values = values_as_in_model(model, true, vi, [@varname(x)])
        # Since x subsumes both x[1] and x[2], they should be included
        @test haskey(values, @varname(x[1]))
        @test haskey(values, @varname(x[2]))
        @test !haskey(values, @varname(y))
        @test haskey(values, @varname(z))  # := is always included

        # Specify instead using `set_tracked_varnames` method
        model2 = DynamicPPL.set_tracked_varnames(model, [@varname(y)])
        values = values_as_in_model(model2, true, vi)
        @test !haskey(values, @varname(x[1]))
        @test !haskey(values, @varname(x[2]))
        @test haskey(values, @varname(y))
        @test haskey(values, @varname(z))  # := is always included
    end
end
