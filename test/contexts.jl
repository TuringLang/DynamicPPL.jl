using Test, DynamicPPL
using DynamicPPL:
    leafcontext,
    setleafcontext,
    childcontext,
    setchildcontext,
    AbstractContext,
    NodeTrait,
    IsLeaf,
    IsParent,
    PointwiseLikelihoodContext,
    contextual_isassumption,
    ConditionContext,
    hasvalue,
    getvalue,
    hasvalue_nested,
    getvalue_nested

# Dummy context to test nested behaviors.
struct ParentContext{C<:AbstractContext} <: AbstractContext
    context::C
end
ParentContext() = ParentContext(DefaultContext())
DynamicPPL.NodeTrait(::ParentContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::ParentContext) = context.context
DynamicPPL.setchildcontext(::ParentContext, child) = ParentContext(child)
Base.show(io::IO, c::ParentContext) = print(io, "ParentContext(", childcontext(c), ")")

# TODO: Should we maybe put this in DPPL itself?
function Base.iterate(context::AbstractContext)
    if NodeTrait(context) isa IsLeaf
        return nothing
    end

    return context, context
end
function Base.iterate(_::AbstractContext, context::AbstractContext)
    return _iterate(NodeTrait(context), context)
end
_iterate(::IsLeaf, context) = nothing
function _iterate(::IsParent, context)
    child = childcontext(context)
    return child, child
end

Base.IteratorSize(::Type{<:AbstractContext}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:AbstractContext}) = Base.EltypeUnknown()

"""
    remove_prefix(vn::VarName)

Return `vn` but now with the prefix removed.
"""
remove_prefix(vn::VarName) = VarName{Symbol(split(string(vn), ".")[end])}(vn.indexing)

"""
    varnames(vn::VarName, val)

Return iterator over all varnames that are represented by `vn` on `val`,
e.g. `varnames(@varname(x), rand(2))` results in an iterator over `[@varname(x[1]), @varname(x[2])]`.
"""
varnames(vn::VarName, val::Real) = [vn]
function varnames(vn::VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (VarName(vn, (vn.indexing..., Tuple(I))) for I in CartesianIndices(val))
end
function varnames(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varnames(VarName(vn, (vn.indexing..., Tuple(I))), val[I]) for
        I in CartesianIndices(val)
    )
end

@testset "contexts.jl" begin
    child_contexts = [DefaultContext(), PriorContext(), LikelihoodContext()]

    parent_contexts = [
        ParentContext(DefaultContext()),
        SamplingContext(),
        MiniBatchContext(DefaultContext(), 0.0),
        PrefixContext{:x}(DefaultContext()),
        PointwiseLikelihoodContext(),
        ConditionContext((x=1.0,)),
        ConditionContext((x=1.0,), ParentContext(ConditionContext((y=2.0,)))),
        ConditionContext((x=1.0,), PrefixContext{:a}(ConditionContext((var"a.y"=2.0,)))),
        ConditionContext((x=[1.0, missing],)),
    ]

    contexts = vcat(child_contexts, parent_contexts)

    @testset "NodeTrait" begin
        @testset "$context" for context in contexts
            # Every `context` should have a `NodeTrait`.
            @test NodeTrait(context) isa NodeTrait
        end
    end

    @testset "leafcontext" begin
        @testset "$context" for context in child_contexts
            @test leafcontext(context) === context
        end

        @testset "$context" for context in parent_contexts
            @test NodeTrait(leafcontext(context)) isa IsLeaf
        end
    end

    @testset "setleafcontext" begin
        @testset "$context" for context in child_contexts
            # Setting to itself should return itself.
            @test setleafcontext(context, context) === context

            # Setting to a different context should return that context.
            new_leaf = context isa DefaultContext ? PriorContext() : DefaultContext()
            @test setleafcontext(context, new_leaf) === new_leaf

            # Also works for parent contexts.
            new_leaf = ParentContext(context)
            @test setleafcontext(context, new_leaf) === new_leaf
        end

        @testset "$context" for context in parent_contexts
            # Leaf contexts.
            new_leaf =
                leafcontext(context) isa DefaultContext ? PriorContext() : DefaultContext()
            @test leafcontext(setleafcontext(context, new_leaf)) === new_leaf

            # Setting parent contexts as "leaf" means that the new leaf should be
            # the leaf of the parent context we just set as the leaf.
            new_leaf = ParentContext((
                leafcontext(context) isa DefaultContext ? PriorContext() : DefaultContext()
            ))
            @test leafcontext(setleafcontext(context, new_leaf)) === leafcontext(new_leaf)
        end
    end

    # `IsParent` interface.
    @testset "childcontext" begin
        @testset "$context" for context in parent_contexts
            @test childcontext(context) isa AbstractContext
        end
    end

    @testset "setchildcontext" begin
        @testset "nested contexts" begin
            # Both of the following should result in the same context.
            context1 = ParentContext(ParentContext(ParentContext()))
            context2 = setchildcontext(
                ParentContext(), setchildcontext(ParentContext(), ParentContext())
            )
            @test context1 === context2
        end

        @testset "$context" for context in parent_contexts
            # Setting the child context to a leaf should now change the `leafcontext` accordingly.
            new_leaf =
                leafcontext(context) isa DefaultContext ? PriorContext() : DefaultContext()
            new_context = setchildcontext(context, new_leaf)
            @test childcontext(new_context) === leafcontext(new_context) === new_leaf
        end
    end

    @testset "contextual_isassumption" begin
        @testset "$context" for context in contexts
            # Any `context` should return `true` by default.
            @test contextual_isassumption(context, VarName{gensym(:x)}())

            if any(Base.Fix2(isa, ConditionContext), context)
                # We have a `ConditionContext` among us.
                # Let's first extract the conditioned variables.
                conditioned_values = DynamicPPL.conditioned(context)

                for (sym, val) in pairs(conditioned_values)
                    vn = VarName{sym}()

                    # We need to drop the prefix of `var` since in `contextual_isassumption`
                    # it will be threaded through the `PrefixContext` before it reaches
                    # `ConditionContext` with the conditioned variable.
                    vn_without_prefix = remove_prefix(vn)

                    # Let's check elementwise.
                    for vn_child in varnames(vn_without_prefix, val)
                        if DynamicPPL._getindex(val, vn_child.indexing) === missing
                            @test contextual_isassumption(context, vn_child)
                        else
                            @test !contextual_isassumption(context, vn_child)
                        end
                    end
                end
            end
        end
    end

    @testset "getvalue_nested & hasvalue_nested" begin
        @testset "$context" for context in contexts
            fake_vn = VarName{gensym(:x)}()
            @test !hasvalue_nested(context, fake_vn)
            @test_throws ErrorException getvalue_nested(context, fake_vn)

            if any(Base.Fix2(isa, ConditionContext), context)
                # `ConditionContext` specific.

                # Let's first extract the conditioned variables.
                conditioned_values = DynamicPPL.conditioned(context)

                for (sym, val) in pairs(conditioned_values)
                    vn = VarName{sym}()

                    # We need to drop the prefix of `var` since in `contextual_isassumption`
                    # it will be threaded through the `PrefixContext` before it reaches
                    # `ConditionContext` with the conditioned variable.
                    vn_without_prefix = remove_prefix(vn)

                    for vn_child in varnames(vn_without_prefix, val)
                        # `vn_child` should be in `context`.
                        @test hasvalue_nested(context, vn_child)
                        # Value should be the same as extracted above.
                        @test getvalue_nested(context, vn_child) ===
                              DynamicPPL._getindex(val, vn_child.indexing)
                    end
                end
            end
        end
    end

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

        vn = VarName{:x}((1,))
        vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
        @test DynamicPPL.getsym(vn_prefixed) == Symbol("a.b.c.d.e.f.x")
        @test vn_prefixed.indexing === vn.indexing
    end
end
