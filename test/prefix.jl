"""
Note that `test/submodel.jl` also contains a number of tests which make use of
prefixing functionality (more like end-to-end tests). This file contains what
are essentially unit tests for prefixing functions.
"""
module DPPLPrefixTests

using DynamicPPL
# not exported
using DynamicPPL: FixedContext, prefix_cond_and_fixed_variables, childcontext
using Distributions
using Test

@testset "prefix.jl" begin
    @testset "prefix_cond_and_fixed_variables" begin
        @testset "ConditionContext" begin
            c1 = ConditionContext((c=1, d=2))
            c1_prefixed = prefix_cond_and_fixed_variables(c1, @varname(a))
            @test c1_prefixed isa ConditionContext
            @test childcontext(c1_prefixed) isa DefaultContext
            @test length(c1_prefixed.values) == 2
            @test c1_prefixed.values[@varname(a.c)] == 1
            @test c1_prefixed.values[@varname(a.d)] == 2
        end

        @testset "FixedContext" begin
            c2 = FixedContext((f=1, g=2))
            c2_prefixed = prefix_cond_and_fixed_variables(c2, @varname(a))
            @test c2_prefixed isa FixedContext
            @test childcontext(c2_prefixed) isa DefaultContext
            @test length(c2_prefixed.values) == 2
            @test c2_prefixed.values[@varname(a.f)] == 1
            @test c2_prefixed.values[@varname(a.g)] == 2
        end

        @testset "Nested ConditionContext and FixedContext" begin
            c3 = ConditionContext((c=1, d=2), FixedContext((f=1, g=2)))
            c3_prefixed = prefix_cond_and_fixed_variables(c3, @varname(a))
            c3_prefixed_child = childcontext(c3_prefixed)
            @test c3_prefixed isa ConditionContext
            @test length(c3_prefixed.values) == 2
            @test c3_prefixed.values[@varname(a.c)] == 1
            @test c3_prefixed.values[@varname(a.d)] == 2
            @test c3_prefixed_child isa FixedContext
            @test length(c3_prefixed_child.values) == 2
            @test c3_prefixed_child.values[@varname(a.f)] == 1
            @test c3_prefixed_child.values[@varname(a.g)] == 2
            @test childcontext(c3_prefixed_child) isa DefaultContext
        end
    end

    @testset "DynamicPPL.prefix(::Model, x)" begin
        @model function demo()
            x ~ Normal()
            return y ~ Normal()
        end
        model = demo()

        @testset "No conditioning / fixing" begin
            pmodel = DynamicPPL.prefix(model, @varname(a))
            @test pmodel.prefix == @varname(a)
            vi = VarInfo(pmodel)
            @test Set(keys(vi)) == Set([@varname(a.x), @varname(a.y)])
        end

        @testset "Prefixing a conditioned model" begin
            cmodel = model | (; x=1.0)
            # Sanity check.
            vi = VarInfo(cmodel)
            @test Set(keys(vi)) == Set([@varname(y)])
            # Now prefix.
            pcmodel = DynamicPPL.prefix(cmodel, @varname(a))
            @test pcmodel.prefix == @varname(a)
            # Because the model was conditioned on `x` _prior_ to prefixing,
            # the resulting `a.x` variable should also be conditioned. In
            # other words, which variables are treated as conditioned should be
            # invariant to prefixing.
            vi = VarInfo(pcmodel)
            @test Set(keys(vi)) == Set([@varname(a.y)])
        end

        @testset "Prefixing a fixed model" begin
            # Same as above but for FixedContext rather than Condition.
            fmodel = fix(model, (; y=1.0))
            # Sanity check.
            vi = VarInfo(fmodel)
            @test Set(keys(vi)) == Set([@varname(x)])
            # Now prefix.
            pfmodel = DynamicPPL.prefix(fmodel, @varname(a))
            @test pfmodel.prefix == @varname(a)
            # Because the model was conditioned on `x` _prior_ to prefixing,
            # the resulting `a.x` variable should also be conditioned. In
            # other words, which variables are treated as conditioned should be
            # invariant to prefixing.
            vi = VarInfo(pfmodel)
            @test Set(keys(vi)) == Set([@varname(a.x)])
        end

        @testset "Conditioning a prefixed model" begin
            # If the prefixing happens first, then we want to make sure that the
            # user is forced to apply conditioning WITH the prefix.
            pmodel = DynamicPPL.prefix(model, @varname(a))

            # If this doesn't happen...
            cpmodel_wrong = pmodel | (; x=1.0)
            @test cpmodel_wrong.prefix == @varname(a)
            vi = VarInfo(cpmodel_wrong)
            # Then `a.x` will be `assume`d
            @test Set(keys(vi)) == Set([@varname(a.x), @varname(a.y)])

            # If it does...
            cpmodel_right = pmodel | (@varname(a.x) => 1.0)
            @test cpmodel_right.prefix == @varname(a)
            vi = VarInfo(cpmodel_right)
            # Then `a.x` will be `observe`d
            @test Set(keys(vi)) == Set([@varname(a.y)])
        end
    end
end

end
