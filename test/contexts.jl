module DynamicPPLContextTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Test, DynamicPPL
using AbstractPPL: AbstractPPL
using Bijectors: inverse, Bijectors
using DynamicPPL:
    leafcontext,
    setleafcontext,
    childcontext,
    setchildcontext,
    AbstractContext,
    AbstractParentContext,
    PrefixContext

struct TestParentContext{C<:AbstractContext} <: AbstractParentContext
    child::C
end
DynamicPPL.childcontext(ctx::TestParentContext) = ctx.child
DynamicPPL.setchildcontext(::TestParentContext, child) = TestParentContext(child)

@testset "contexts.jl" begin
    contexts = Dict(
        :default => DefaultContext(),
        :testparent => TestParentContext(DefaultContext()),
        :prefix => PrefixContext(@varname(x)),
    )

    @testset "$(name)" for (name, context) in contexts
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            if context isa AbstractParentContext
                leaf = InitContext(InitFromPrior(), UnlinkAll())
                @test leafcontext(setleafcontext(context, leaf)) === leaf
                child = TestParentContext(DefaultContext())
                @test childcontext(setchildcontext(context, child)) === child
                replaced = setchildcontext(context, leaf)
                @test childcontext(replaced) === leafcontext(replaced) === leaf
            end
            prepared = contextualize(model, context)
            _, vi = init!!(prepared, VarInfo(VectorValueAccumulator()))
            @test vi isa VarInfo
            _, vi = evaluate!!(
                prepared,
                InitContext(InitFromParams(get_values(vi), nothing), UnlinkAll()),
                vi,
            )
            @test vi isa VarInfo
        end
    end

    @testset "PrefixContext" begin
        @testset "prefixing" begin
            ctx = @inferred PrefixContext(
                @varname(a),
                PrefixContext(
                    @varname(b),
                    PrefixContext(
                        @varname(c),
                        PrefixContext(
                            @varname(d),
                            PrefixContext(
                                @varname(e), PrefixContext(@varname(f), DefaultContext())
                            ),
                        ),
                    ),
                ),
            )
            vn = @varname(x)
            vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
            @test vn_prefixed == @varname(a.b.c.d.e.f.x)

            vn = @varname(x[1])
            vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
            @test vn_prefixed == @varname(a.b.c.d.e.f.x[1])
        end

        @testset "nested within arbitrary context stacks" begin
            vn = @varname(x[1])
            ctx1 = PrefixContext(@varname(a))
            @test DynamicPPL.prefix(ctx1, vn) == @varname(a.x[1])
            ctx2 = TestParentContext(ctx1)
            @test DynamicPPL.prefix(ctx2, vn) == @varname(a.x[1])
            ctx3 = PrefixContext(@varname(b), ctx2)
            @test DynamicPPL.prefix(ctx3, vn) == @varname(b.a.x[1])
            ctx4 = TestParentContext(ctx3)
            @test DynamicPPL.prefix(ctx4, vn) == @varname(b.a.x[1])
        end

        @testset "prefix_and_strip_contexts + extract_prefixes" begin
            vn = @varname(x[1])
            ctx1 = PrefixContext(@varname(a))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx1, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == DefaultContext()
            new_ctx, prefixes = DynamicPPL.extract_prefixes(ctx1)
            @test new_ctx == DefaultContext()
            @test prefixes == @varname(a)

            ctx2 = TestParentContext(PrefixContext(@varname(a)))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx2, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == TestParentContext(DefaultContext())
            new_ctx, prefixes = DynamicPPL.extract_prefixes(ctx2)
            @test new_ctx == TestParentContext(DefaultContext())
            @test prefixes == @varname(a)

            ctx3 = PrefixContext(@varname(a), TestParentContext(DefaultContext()))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx3, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == TestParentContext(DefaultContext())
            new_ctx, prefixes = DynamicPPL.extract_prefixes(ctx3)
            @test new_ctx == TestParentContext(DefaultContext())
            @test prefixes == @varname(a)

            ctx4 = TestParentContext(
                PrefixContext(@varname(a), TestParentContext(DefaultContext()))
            )
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx4, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == TestParentContext(TestParentContext(DefaultContext()))
            new_ctx, prefixes = DynamicPPL.extract_prefixes(ctx4)
            @test new_ctx == TestParentContext(TestParentContext(DefaultContext()))
            @test prefixes == @varname(a)

            ctx5 = DefaultContext()
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx5, vn)
            @test new_vn == vn
            @test new_ctx == DefaultContext()
            new_ctx, prefixes = DynamicPPL.extract_prefixes(ctx5)
            @test new_ctx == DefaultContext()
            @test prefixes == nothing
        end

        @testset "evaluation: $(model.f)" for model in DynamicPPL.TestUtils.ALL_MODELS
            prefix_vn = @varname(my_prefix)
            context = DynamicPPL.PrefixContext(prefix_vn, DefaultContext())
            new_model = contextualize(model, context)
            # Initialize a new varinfo with the prefixed model
            _, varinfo = DynamicPPL.init!!(
                new_model, DynamicPPL.VarInfo(VectorValueAccumulator())
            )
            # Extract the resulting varnames
            vns_actual = Set(keys(varinfo))

            # Extract the ground truth varnames
            vns_expected = Set([
                AbstractPPL.prefix(vn, prefix_vn) for
                vn in DynamicPPL.TestUtils.varnames(model)
            ])

            # Check that all variables are prefixed correctly.
            @test vns_actual == vns_expected
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
