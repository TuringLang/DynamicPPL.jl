# contexts.jl
# -----------
#
# Utilities for testing contexts.

"""
    test_context_interface(context)

Test that `context` implements the `AbstractContext` interface.
"""
function test_context_interface(context)
    # Is a subtype of `AbstractContext`.
    @test context isa DynamicPPL.AbstractContext
    # Should implement `NodeTrait.`
    @test DynamicPPL.NodeTrait(context) isa Union{DynamicPPL.IsParent,DynamicPPL.IsLeaf}
    # If it's a parent.
    if DynamicPPL.NodeTrait(context) == DynamicPPL.IsParent
        # Should implement `childcontext` and `setchildcontext`
        @test DynamicPPL.setchildcontext(context, DynamicPPL.childcontext(context)) ==
            context
    end
end

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
