module DPPLSubmodelTests

using DynamicPPL
using Distributions
using Test

@testset "submodels.jl" begin
    @testset "$op with AbstractPPL API" for op in [condition, fix]
        x_val = 1.0
        x_logp = op == condition ? logpdf(Normal(), x_val) : 0.0

        @testset "Auto prefix" begin
            @model function inner()
                x ~ Normal()
                y ~ Normal()
                return (x, y)
            end
            @model function outer()
                return a ~ to_submodel(inner())
            end
            inner_op = op(inner(), (@varname(x) => x_val))
            @model function outer2()
                return a ~ to_submodel(inner_op)
            end
            with_inner_op = outer2()
            with_outer_op = op(outer(), (@varname(a.x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(a.x), @varname(a.y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                vi = VarInfo(model)
                @test getlogp(vi) == x_logp + logpdf(Normal(), vi[@varname(a.y)])
                # Check the keys
                @test Set(keys(VarInfo(model))) == Set([@varname(a.y)])
            end
        end

        @testset "No prefix" begin
            @model function inner()
                x ~ Normal()
                y ~ Normal()
                return (x, y)
            end
            @model function outer()
                return a ~ to_submodel(inner(), false)
            end
            @model function outer2()
                return a ~ to_submodel(inner_op, false)
            end
            with_inner_op = outer2()
            inner_op = op(inner(), (@varname(x) => x_val))
            with_outer_op = op(outer(), (@varname(x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(x), @varname(y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                vi = VarInfo(model)
                @test getlogp(vi) == x_logp + logpdf(Normal(), vi[@varname(y)])
                # Check the keys
                @test Set(keys(VarInfo(model))) == Set([@varname(y)])
            end
        end

        @testset "Manual prefix" begin
            @model function inner()
                x ~ Normal()
                y ~ Normal()
                return (x, y)
            end
            @model function outer()
                return a ~ to_submodel(prefix(inner(), :b), false)
            end
            inner_op = op(inner(), (@varname(x) => x_val))
            @model function outer2()
                return a ~ to_submodel(prefix(inner_op, :b), false)
            end
            with_inner_op = outer2()
            with_outer_op = op(outer(), (@varname(b.x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(b.x), @varname(b.y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                vi = VarInfo(model)
                @test getlogp(vi) == x_logp + logpdf(Normal(), vi[@varname(b.y)])
                # Check the keys
                @test Set(keys(VarInfo(model))) == Set([@varname(b.y)])
            end
        end

        @testset "Complex prefixes" begin
            mutable struct P
                a::Float64
                b::Float64
            end
            @model function f()
                x = Vector{Float64}(undef, 1)
                x[1] ~ Normal()
                y ~ Normal()
                return x[1]
            end
            @model function g()
                p = P(1.0, 2.0)
                p.a ~ to_submodel(f())
                p.b ~ Normal()
                return (p.a, p.b)
            end
            expected_vns = Set([
                @varname(var"p.a".x[1]), @varname(var"p.a".y), @varname(p.b)
            ])
            @test Set(keys(VarInfo(g()))) == expected_vns

            # Check that we can condition/fix on any of them from the outside
            for vn in expected_vns
                op_g = op(g(), (vn => 1.0))
                vi = VarInfo(op_g)
                @test Set(keys(vi)) == symdiff(expected_vns, Set([vn]))
            end
        end

        @testset "Nested submodels" begin
            @model function f()
                x ~ Normal()
                return y ~ Normal()
            end
            @model function g()
                return _unused ~ to_submodel(prefix(f(), :b), false)
            end
            @model function h()
                return a ~ to_submodel(g())
            end

            # No conditioning
            vi = VarInfo(h())
            @test Set(keys(vi)) == Set([@varname(a.b.x), @varname(a.b.y)])
            @test getlogp(vi) ==
                logpdf(Normal(), vi[@varname(a.b.x)]) +
                  logpdf(Normal(), vi[@varname(a.b.y)])

            # Conditioning/fixing at the top level
            op_h = op(h(), (@varname(a.b.x) => x_val))

            # Conditioning/fixing at the second level
            op_g = op(g(), (@varname(b.x) => x_val))
            @model function h2()
                return a ~ to_submodel(op_g)
            end

            # Conditioning/fixing at the very bottom
            op_f = op(f(), (@varname(x) => x_val))
            @model function g2()
                return _unused ~ to_submodel(prefix(op_f, :b), false)
            end
            @model function h3()
                return a ~ to_submodel(g2())
            end

            models = [("top", op_h), ("middle", h2()), ("bottom", h3())]
            @testset "$name" for (name, model) in models
                vi = VarInfo(model)
                @test Set(keys(vi)) == Set([@varname(a.b.y)])
                @test getlogp(vi) == x_logp + logpdf(Normal(), vi[@varname(a.b.y)])
            end
        end
    end

    @testset "conditioning via model arguments" begin
        @model function f(x)
            x ~ Normal()
            return y ~ Normal()
        end
        @model function g(inner_x)
            return a ~ to_submodel(f(inner_x))
        end

        vi = VarInfo(g(1.0))
        @test Set(keys(vi)) == Set([@varname(a.y)])

        vi = VarInfo(g(missing))
        @test Set(keys(vi)) == Set([@varname(a.x), @varname(a.y)])
    end
end

end
