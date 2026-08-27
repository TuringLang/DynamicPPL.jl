# contexts.jl
# -----------
#
# Utilities for testing contexts.

"""
    test_context(context::AbstractContext, model::Model)

Test that `context` correctly implements the `AbstractContext` interface for `model`.

This method ensures that `context`
- Correctly implements the `AbstractContext` interface.
- Correctly implements the tilde-pipeline.
"""
function test_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
    return test_leaf_context(context, model)
end
function test_context(context::DynamicPPL.AbstractParentContext, model::DynamicPPL.Model)
    return test_parent_context(context, model)
end

function test_leaf_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
    # Note that for a leaf context we can't assume that it will work with an
    # empty VarInfo. (For example, DefaultContext will error with empty
    # varinfos.) Thus we only test evaluation with VarInfos that are already
    # filled with values.
    @testset "evaluation" begin
        # Generate a new filled varinfo
        vi = DynamicPPL.VarInfo(model)
        # Set the test context as the new leaf context
        new_model = DynamicPPL.setleafcontext(model, context)
        # It might seem a bit ugly that we have to use `evaluate_nowarn!!` here. Essentially
        # we want to test that low-level evaluation works with the context, so this is the
        # right thing to do.
        _, vi = DynamicPPL.evaluate_nowarn!!(new_model, vi)
        @test vi isa DynamicPPL.VarInfo
    end
end

function test_parent_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
    @testset "get/set leaf and child contexts" begin
        # Ensure we're using a different leaf context than the current.
        leafcontext_new = if DynamicPPL.leafcontext(context) isa DefaultContext
            DynamicPPL.InitContext(Random.MersenneTwister(1234), InitFromPrior(), UnlinkAll())
        else
            DefaultContext()
        end
        @test DynamicPPL.leafcontext(DynamicPPL.setleafcontext(context, leafcontext_new)) ==
            leafcontext_new
        childcontext_new = DynamicPPL.PrefixContext(Val(:test))
        @test DynamicPPL.childcontext(
            DynamicPPL.setchildcontext(context, childcontext_new)
        ) == childcontext_new
        # Setting the child context to a leaf should now change the leafcontext
        # accordingly.
        context_with_new_leaf = DynamicPPL.setchildcontext(context, leafcontext_new)
        @test DynamicPPL.childcontext(context_with_new_leaf) ===
            DynamicPPL.leafcontext(context_with_new_leaf) ===
            leafcontext_new
    end

    @testset "initialisation and evaluation" begin
        new_model = contextualize(model, context)
        vi = DynamicPPL.VarInfo()
        # Initialisation
        _, vi = DynamicPPL.init!!(
            new_model,
            DynamicPPL.VarInfo(),
            DynamicPPL.InitFromPrior(),
            DynamicPPL.UnlinkAll(),
        )
        @test vi isa DynamicPPL.VarInfo
        # Evaluation. See above regarding note about evaluate_nowarn!!.
        _, vi = DynamicPPL.evaluate_nowarn!!(new_model, vi)
        @test vi isa DynamicPPL.VarInfo
    end
end
