# contexts.jl
# -----------
#
# Utilities for testing contexts.

# Dummy context to test nested behaviors.
struct TestParentContext{C<:DynamicPPL.AbstractContext} <: DynamicPPL.AbstractContext
    context::C
end
TestParentContext() = TestParentContext(DefaultContext())
DynamicPPL.NodeTrait(::TestParentContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::TestParentContext) = context.context
DynamicPPL.setchildcontext(::TestParentContext, child) = TestParentContext(child)
function Base.show(io::IO, c::TestParentContext)
    return print(io, "TestParentContext(", DynamicPPL.childcontext(c), ")")
end

"""
    test_context(context::AbstractContext, model::Model)

Test that `context` correctly implements the `AbstractContext` interface for `model`.

This method ensures that `context`
- Correctly implements the `AbstractContext` interface.
- Correctly implements the tilde-pipeline.
"""
function test_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
    node_trait = DynamicPPL.NodeTrait(context)
    if node_trait isa DynamicPPL.IsLeaf
        test_leaf_context(context, model)
    elseif node_trait isa DynamicPPL.IsParent
        test_parent_context(context, model)
    else
        error("Invalid NodeTrait: $node_trait")
    end
end

function test_leaf_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
    @test DynamicPPL.NodeTrait(context) isa DynamicPPL.IsLeaf

    # Note that for a leaf context we can't assume that it will work with an
    # empty VarInfo. (For example, DefaultContext will error with empty
    # varinfos.) Thus we only test evaluation with VarInfos that are already
    # filled with values.
    @testset "evaluation" begin
        # Generate a new filled untyped varinfo
        _, untyped_vi = DynamicPPL.init!!(model, DynamicPPL.VarInfo())
        typed_vi = DynamicPPL.typed_varinfo(untyped_vi)
        # Set the test context as the new leaf context
        new_model = DynamicPPL.setleafcontext(model, context)
        # Check that evaluation works
        for vi in [untyped_vi, typed_vi]
            _, vi = DynamicPPL.evaluate!!(new_model, vi)
            @test vi isa DynamicPPL.VarInfo
        end
    end
end

function test_parent_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
    @test DynamicPPL.NodeTrait(context) isa DynamicPPL.IsParent

    @testset "get/set leaf and child contexts" begin
        # Ensure we're using a different leaf context than the current.
        leafcontext_new = if DynamicPPL.leafcontext(context) isa DefaultContext
            DynamicPPL.DynamicTransformationContext{false}()
        else
            DefaultContext()
        end
        @test DynamicPPL.leafcontext(DynamicPPL.setleafcontext(context, leafcontext_new)) ==
            leafcontext_new
        childcontext_new = TestParentContext()
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
        for vi in [DynamicPPL.VarInfo(), DynamicPPL.typed_varinfo(DynamicPPL.VarInfo())]
            # Initialisation
            _, vi = DynamicPPL.init!!(new_model, DynamicPPL.VarInfo())
            @test vi isa DynamicPPL.VarInfo
            # Evaluation
            _, vi = DynamicPPL.evaluate!!(new_model, vi)
            @test vi isa DynamicPPL.VarInfo
        end
    end
end
