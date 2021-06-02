@testset "contexts.jl" begin
    @testset "PrefixContext" begin
        ctx = @inferred Prefix{:a}(Prefix{:b}(Prefix{:c}(Prefix{:d}(Prefix{:e}(Prefix{:f}())))))
        vn = VarName{:x}()
        vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
        @test DynamicPPL.getsym(vn_prefixed) == Symbol("a.b.c.d.e.f.x")
        @test vn_prefixed.indices === vn.indices

        vn = VarName{:x}((1, ))
        vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
        @test DynamicPPL.getsym(vn_prefixed) == Symbol("a.b.c.d.e.f.x")
        @test vn_prefixed.indices === vn.indices
    end
end
