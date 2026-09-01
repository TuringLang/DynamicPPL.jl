module RawValueAccTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Test
using Distributions
using DynamicPPL
using LinearAlgebra

@testset "RawValueAccumulator" begin
    @testset "stored values do not alias data in model" begin
        # This testset checks that RawValueAccumulator always stores the values
        # at the point in which they are seen in the model
        @model function f()
            x ~ MvNormal(zeros(4), I) # sample
            x .= NaN                  # mutate
            y := randn(5)             # same but with := instead
            y .= NaN
            return vcat(x, y)
        end
        model = f()

        # Return value should have been mutated
        @test all(isnan, model())
        # rand uses RawValueAcc under the hood
        @test !any(isnan, rand(model)[@varname(x)])
        # Explicitly spelled out
        accs = OnlyAccsVarInfo(RawValueAccumulator(true))
        _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
        vnt = get_raw_values(accs)
        @test !any(isnan, vnt[@varname(x)])
        @test !any(isnan, vnt[@varname(y)])
        @test !any(isnan, get_colon_eq_values(accs)[@varname(y)])
        # with LDF
        ldf = LogDensityFunction(model)
        p = rand(ldf)
        pws = ParamsWithStats(p, ldf; include_colon_eq=true)
        @test !any(isnan, pws.params[@varname(x)])
        @test !any(isnan, pws.params[@varname(y)])
    end

    @testset "evaluation order" begin
        @model function f()
            y := 1.0
            return x ~ Normal()
        end
        accs = OnlyAccsVarInfo(RawValueAccumulator(true))
        _, accs = init!!(f(), accs, InitFromPrior(), UnlinkAll())
        @test collect(keys(get_raw_values(accs))) == [@varname(y), @varname(x)]
    end

    @testset "indexed provenance" begin
        acc = RawValueAccumulator(true)
        acc = DynamicPPL.accumulate_assume!!(
            acc, 1.0, 1.0, 0.0, @varname(x[1]), Normal(), zeros(3)
        )
        acc = DynamicPPL.store_colon_eq!!(acc, @varname(x[2:3]), [2.0, 3.0], zeros(3))
        accs = OnlyAccsVarInfo(acc)
        @test keys(get_parameter_values(accs)) == [@varname(x[1])]
        @test keys(get_colon_eq_values(accs)) == [@varname(x[2]), @varname(x[3])]
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
