module DPPLSubmodelTests

using DynamicPPL
using Distributions
using Test

@testset "submodels.jl" begin
    @testset "Conditioning variables" begin
        @testset "Auto prefix" begin
            @model function inner()
                x ~ Normal()
                return y ~ Normal()
            end
            @model function outer()
                return a ~ to_submodel(inner())
            end
            inner_cond = inner() | (@varname(x) => 1.0)
            with_outer_cond = outer() | (@varname(a.x) => 1.0)

            # No conditioning
            @test Set(keys(VarInfo(outer()))) == Set([@varname(a.x), @varname(a.y)])
            # Conditioning from the outside
            @test Set(keys(VarInfo(with_outer_cond))) == Set([@varname(a.y)])
            # Conditioning from the inside
            @model function outer2()
                return a ~ to_submodel(inner_cond)
            end
            with_inner_cond = outer2()
            @test Set(keys(VarInfo(with_inner_cond))) == Set([@varname(a.y)])
        end

        @testset "No prefix" begin
            @model function inner()
                x ~ Normal()
                return y ~ Normal()
            end
            @model function outer()
                return a ~ to_submodel(inner(), false)
            end
            inner_cond = inner() | (@varname(x) => 1.0)
            with_outer_cond = outer() | (@varname(x) => 1.0)

            # No conditioning
            @test Set(keys(VarInfo(outer()))) == Set([@varname(x), @varname(y)])
            # Conditioning from the outside
            @test Set(keys(VarInfo(with_outer_cond))) == Set([@varname(y)])
            # Conditioning from the inside
            @model function outer2()
                return a ~ to_submodel(inner_cond, false)
            end
            with_inner_cond = outer2()
            @test Set(keys(VarInfo(with_inner_cond))) == Set([@varname(y)])
        end

        @testset "Manual prefix" begin
            @model function inner()
                x ~ Normal()
                return y ~ Normal()
            end
            @model function outer()
                return a ~ to_submodel(prefix(inner(), :b), false)
            end
            inner_cond = inner() | (@varname(x) => 1.0)
            with_outer_cond = outer() | (@varname(b.x) => 1.0)

            # No conditioning
            @test Set(keys(VarInfo(outer()))) == Set([@varname(b.x), @varname(b.y)])
            # Conditioning from the outside
            @test Set(keys(VarInfo(with_outer_cond))) == Set([@varname(b.y)])
            # Conditioning from the inside
            @model function outer2()
                return a ~ to_submodel(prefix(inner_cond, :b), false)
            end
            with_inner_cond = outer2()
            @test Set(keys(VarInfo(with_inner_cond))) == Set([@varname(b.y)])
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
            @test Set(keys(VarInfo(h()))) == Set([@varname(a.b.x), @varname(a.b.y)])

            # Conditioning at the top level
            condition_h = h() | (@varname(a.b.x) => 1.0)
            @test Set(keys(VarInfo(condition_h))) == Set([@varname(a.b.y)])

            # Conditioning at the second level
            condition_g = g() | (@varname(b.x) => 1.0)
            @model function h2()
                return a ~ to_submodel(condition_g)
            end
            @test Set(keys(VarInfo(h2()))) == Set([@varname(a.b.y)])

            # Conditioning at the very bottom
            condition_f = f() | (@varname(x) => 1.0)
            @model function g2()
                return _unused ~ to_submodel(prefix(condition_f, :b), false)
            end
            @model function h3()
                return a ~ to_submodel(g2())
            end
            @test Set(keys(VarInfo(h3()))) == Set([@varname(a.b.y)])
        end
    end
end

end
