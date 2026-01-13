# contexts.jl
# -----------
#
# Utilities for testing contexts.

# Dummy context to test nested behaviors.
struct TestParentContext{C<:DynamicPPL.AbstractContext} <: DynamicPPL.AbstractParentContext
    context::C
end
TestParentContext() = TestParentContext(DefaultContext())
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
        _, vi = DynamicPPL.init!!(model, DynamicPPL.VarInfo())
        # Set the test context as the new leaf context
        new_model = DynamicPPL.setleafcontext(model, context)
        _, vi = DynamicPPL.evaluate!!(new_model, vi)
        @test vi isa DynamicPPL.VarInfo
    end
end

function test_parent_context(context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
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
        vi = DynamicPPL.VarInfo()
        # Initialisation
        _, vi = DynamicPPL.init!!(new_model, DynamicPPL.VarInfo())
        @test vi isa DynamicPPL.VarInfo
        # Evaluation
        _, vi = DynamicPPL.evaluate!!(new_model, vi)
        @test vi isa DynamicPPL.VarInfo
    end
end
