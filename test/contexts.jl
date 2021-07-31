@testset "contexts.jl" begin
    @testset "PrefixContext" begin
        ctx = @inferred PrefixContext{:f}(
            PrefixContext{:e}(
                PrefixContext{:d}(
                    PrefixContext{:c}(
                        PrefixContext{:b}(PrefixContext{:a}(DefaultContext()))
                    ),
                ),
            ),
        )
        vn = VarName{:x}()
        vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
        @test DynamicPPL.getsym(vn_prefixed) == Symbol("a.b.c.d.e.f.x")
        @test vn_prefixed.indexing === vn.indexing

        vn = VarName{:x}(((1,),))
        vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
        @test DynamicPPL.getsym(vn_prefixed) == Symbol("a.b.c.d.e.f.x")
        @test vn_prefixed.indexing === vn.indexing
    end
end
