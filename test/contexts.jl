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
    AbstractParentContext

struct TestParentContext{C<:AbstractContext} <: AbstractParentContext
    child::C
end
DynamicPPL.childcontext(ctx::TestParentContext) = ctx.child
DynamicPPL.setchildcontext(::TestParentContext, child) = TestParentContext(child)

@testset "contexts.jl" begin
    contexts = Dict(
        :default => DefaultContext(), :testparent => TestParentContext(DefaultContext())
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

    @testset "evaluation: $(model.f)" for model in DynamicPPL.TestUtils.ALL_MODELS
        prefix_vn = @varname(my_prefix)
        new_model = DynamicPPL.prefix(model, prefix_vn)
        # Initialize a new varinfo with the prefixed model
        _, varinfo = DynamicPPL.init!!(
            new_model, DynamicPPL.VarInfo(VectorValueAccumulator())
        )
        # Extract the resulting varnames
        vns_actual = Set(keys(varinfo))

        # Extract the ground truth varnames
        vns_expected = Set([
            AbstractPPL.prefix(vn, prefix_vn) for vn in DynamicPPL.TestUtils.varnames(model)
        ])

        # Check that all variables are prefixed correctly.
        @test vns_actual == vns_expected
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
