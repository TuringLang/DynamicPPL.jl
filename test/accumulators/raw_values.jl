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
        # with LDF
        ldf = LogDensityFunction(model)
        p = rand(ldf)
        pws = ParamsWithStats(p, ldf; include_colon_eq=true)
        @test !any(isnan, pws.params[@varname(x)])
        @test !any(isnan, pws.params[@varname(y)])
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
