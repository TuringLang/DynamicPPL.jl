module DynamicPPLConditionFixTests

using Dates: now
using Distributions
using DynamicPPL
using LinearAlgebra: I
using Test

@info "Testing $(@__FILE__)..."
__now__ = now()

@testset "condition and fix" begin
    @model function demo_cond_fix()
        x ~ Normal()
        return y ~ Normal(x)
    end
    model = demo_cond_fix()

    function test_logp_correct(
        op::Union{typeof(condition),typeof(fix)}, transformed::Model, x
    )
        y = 1.0
        values = VarNamedTuple(; y)
        @test logprior(transformed, values) == logpdf(Normal(x), y)
        if op === condition
            @test loglikelihood(transformed, values) == logpdf(Normal(), x)
        else
            @test iszero(loglikelihood(transformed, values))
        end
    end

    @testset "$op input forms" for op in (condition, fix)
        x = 0.5
        transformed_models = (
            op(model; x),
            op(model, VarNamedTuple(; x)),
            op(model, (; x)),
            op(model, Dict(@varname(x) => x)),
            op(model, @varname(x) => x),
        )
        for transformed in transformed_models
            test_logp_correct(op, transformed, x)
        end
        if op === condition
            test_logp_correct(condition, model | VarNamedTuple(; x), x)
            test_logp_correct(condition, model | (; x), x)
            test_logp_correct(condition, model | Dict(@varname(x) => x), x)
            test_logp_correct(condition, model | (@varname(x) => x), x)
        end
    end

    @testset "values are model fields" begin
        conditioned_model = condition(condition(model; x=1.0); x=2.0, y=3.0)
        @test conditioned_model.context === model.context
        @test conditioned(conditioned_model)[@varname(x)] == 2.0
        @test conditioned(conditioned_model)[@varname(y)] == 3.0
        @test conditioned_model() == 3.0

        fixed_model = fix(fix(model; x=1.0); x=2.0, y=3.0)
        @test fixed_model.context === model.context
        @test fixed(fixed_model)[@varname(x)] == 2.0
        @test fixed(fixed_model)[@varname(y)] == 3.0
        @test fixed_model() == 3.0

        @model return_x() = x ~ Normal()
        @test fix(condition(return_x(); x=1.0); x=2.0)() == 2.0
    end

    @testset "missing values leave variables latent" begin
        conditioned_model = condition(model; x=missing)
        fixed_model = fix(model; x=missing)
        @test keys(VarInfo(conditioned_model)) == [@varname(x), @varname(y)]
        @test keys(VarInfo(fixed_model)) == [@varname(x), @varname(y)]
    end

    @testset "decondition and unfix" begin
        conditioned_model = condition(model; x=1.0, y=2.0)
        @test isempty(keys(conditioned(decondition(conditioned_model))))
        @test keys(conditioned(decondition(conditioned_model, :x))) == [@varname(y)]
        @test keys(conditioned(decondition(conditioned_model, @varname(x)))) ==
            [@varname(y)]

        fixed_model = fix(model; x=1.0, y=2.0)
        @test isempty(keys(fixed(unfix(fixed_model))))
        @test keys(fixed(unfix(fixed_model, :x))) == [@varname(y)]
        @test keys(fixed(unfix(fixed_model, @varname(x)))) == [@varname(y)]

        nested_values = VarNamedTuple((@varname(a.x) => 1.0, @varname(b) => 2.0))
        @test keys(
            conditioned(decondition(condition(model, nested_values), @varname(a)))
        ) == [@varname(b)]
        @test keys(fixed(unfix(fix(model, nested_values), @varname(a)))) == [@varname(b)]
    end

    @testset "parent model values override submodel values" begin
        @model inner() = x ~ Normal()
        @model function outer(inner_model)
            return a ~ to_submodel(inner_model)
        end

        conditioned_inner = condition(inner(); x=1.0)
        @test outer(conditioned_inner)() == 1.0
        @test condition(outer(conditioned_inner), @varname(a.x) => 2.0)() == 2.0

        fixed_inner = fix(inner(); x=1.0)
        @test outer(fixed_inner)() == 1.0
        @test fix(outer(fixed_inner), @varname(a.x) => 2.0)() == 2.0
    end

    @testset "immutable data can be fixed" begin
        @model function ntfix()
            m ~ Normal()
            data = (; x=undef)
            data.x ~ Normal(m, 1.0)
            return data.x
        end
        fixed_model = fix(ntfix(), (; data=(; x=5.0)))
        accs = OnlyAccsVarInfo(RawValueAccumulator(false))
        retval, accs = init!!(fixed_model, accs, InitFromPrior(), UnlinkAll())
        @test retval == 5.0
        @test get_raw_values(accs)[@varname(m)] isa Real
    end

    @testset "multivariate values" begin
        @model function mvnorm()
            x ~ MvNormal(zeros(3), I)
            return x
        end
        templated = @vnt begin
            @template x = zeros(3)
            x[1] := 1.0
            x[2] := 2.0
            x[3] := 3.0
        end
        untemplated = @vnt begin
            x[1] := 1.0
            x[2] := 2.0
            x[3] := 3.0
        end
        for op in (condition, fix), values in (templated, untemplated)
            @test op(mvnorm(), values)() == [1.0, 2.0, 3.0]
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
