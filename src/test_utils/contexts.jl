# contexts.jl
# -----------
#
# Utilities for testing contexts.

"""
Context that multiplies each log-prior by mod
used to test whether varwise_logpriors respects child-context.
"""
struct TestLogModifyingChildContext{T,Ctx} <: DynamicPPL.AbstractContext
    mod::T
    context::Ctx
end
function TestLogModifyingChildContext(
    mod=1.2, context::DynamicPPL.AbstractContext=DynamicPPL.DefaultContext()
)
    return TestLogModifyingChildContext{typeof(mod),typeof(context)}(mod, context)
end

DynamicPPL.NodeTrait(::TestLogModifyingChildContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::TestLogModifyingChildContext) = context.context
function DynamicPPL.setchildcontext(context::TestLogModifyingChildContext, child)
    return TestLogModifyingChildContext(context.mod, child)
end
function DynamicPPL.tilde_assume(context::TestLogModifyingChildContext, right, vn, vi)
    value, logp, vi = DynamicPPL.tilde_assume(context.context, right, vn, vi)
    return value, logp * context.mod, vi
end
function DynamicPPL.dot_tilde_assume(
    context::TestLogModifyingChildContext, right, left, vn, vi
)
    value, logp, vi = DynamicPPL.dot_tilde_assume(context.context, right, left, vn, vi)
    return value, logp * context.mod, vi
end
function DynamicPPL.tilde_observe(context::TestLogModifyingChildContext, right, left, vi)
    logp, vi = DynamicPPL.tilde_observe(context.context, right, left, vi)
    return logp * context.mod, vi
end
function DynamicPPL.dot_tilde_observe(
    context::TestLogModifyingChildContext, right, left, vi
)
    logp, vi = DynamicPPL.dot_tilde_observe(context.context, right, left, vi)
    return logp * context.mod, vi
end

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
    # `NodeTrait`.
    node_trait = DynamicPPL.NodeTrait(context)
    # Throw error immediately if it it's missing a `NodeTrait` implementation.
    node_trait isa Union{DynamicPPL.IsLeaf,DynamicPPL.IsParent} ||
        throw(ValueError("Invalid NodeTrait: $node_trait"))

    # The interface methods.
    if node_trait isa DynamicPPL.IsParent
        # `childcontext` and `setchildcontext`
        # With new child context
        childcontext_new = TestParentContext()
        @test DynamicPPL.childcontext(
            DynamicPPL.setchildcontext(context, childcontext_new)
        ) == childcontext_new
    end

    # To see change, let's make sure we're using a different leaf context than the current.
    leafcontext_new = if DynamicPPL.leafcontext(context) isa DefaultContext
        PriorContext()
    else
        DefaultContext()
    end
    @test DynamicPPL.leafcontext(DynamicPPL.setleafcontext(context, leafcontext_new)) ==
        leafcontext_new

    # Setting the child context to a leaf should now change the leafcontext accordingly.
    context_with_new_leaf = DynamicPPL.setchildcontext(context, leafcontext_new)
    @test DynamicPPL.setchildcontext(context_with_new_leaf) ===
        DynamicPPL.setleafcontext(context_with_new_leaf) ===
        leafcontext_new

    # Make sure that the we can evaluate the model with the context (i.e. that none of the tilde-functions are incorrectly overloaded).
    # The tilde-pipeline contains two different paths: with `SamplingContext` as a parent, and without it.
    # NOTE(torfjelde): Need to sample with the untyped varinfo _using_ the context, since the
    # context might alter which variables are present, their names, etc., e.g. `PrefixContext`.
    # TODO(torfjelde): Make the `varinfo` used for testing a kwarg once it makes sense for other varinfos.
    # Untyped varinfo.
    varinfo_untyped = DynamicPPL.VarInfo()
    @test (DynamicPPL.evaluate!!(model, varinfo_untyped, SamplingContext(context)); true)
    @test (DynamicPPL.evaluate!!(model, varinfo_untyped, context); true)
    # Typed varinfo.
    varinfo_typed = DynamicPPL.TypedVarInfo(varinfo_untyped)
    @test (DynamicPPL.evaluate!!(model, varinfo_typed, SamplingContext(context)); true)
    @test (DynamicPPL.evaluate!!(model, varinfo_typed, context); true)
end
