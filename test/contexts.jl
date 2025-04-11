using Test, DynamicPPL, Accessors
using DynamicPPL:
    leafcontext,
    setleafcontext,
    childcontext,
    setchildcontext,
    AbstractContext,
    NodeTrait,
    IsLeaf,
    IsParent,
    contextual_isassumption,
    ConditionContext,
    decondition_context,
    hasconditioned,
    getconditioned,
    hasconditioned_nested,
    getconditioned_nested

using EnzymeCore

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

@testset "contexts.jl" begin
    child_contexts = Dict(:default => DefaultContext())

    parent_contexts = Dict(
        :testparent => DynamicPPL.TestUtils.TestParentContext(DefaultContext()),
        :sampling => SamplingContext(),
        :prefix => PrefixContext{:x}(DefaultContext()),
        :condition1 => ConditionContext((x=1.0,)),
        :condition2 => ConditionContext(
            (x=1.0,), DynamicPPL.TestUtils.TestParentContext(ConditionContext((y=2.0,)))
        ),
        :condition3 => ConditionContext(
            (x=1.0,), PrefixContext{:a}(ConditionContext(Dict(@varname(a.y) => 2.0)))
        ),
        :condition4 => ConditionContext((x=[1.0, missing],)),
    )

    contexts = merge(child_contexts, parent_contexts)

    @testset "$(name)" for (name, context) in contexts
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            DynamicPPL.TestUtils.test_context(context, model)
        end
    end

    @testset "contextual_isassumption" begin
        @testset "$(name)" for (name, context) in contexts
            # Any `context` should return `true` by default.
            @test contextual_isassumption(context, VarName{gensym(:x)}())

            if any(Base.Fix2(isa, ConditionContext), context)
                # We have a `ConditionContext` among us.
                # Let's first extract the conditioned variables.
                conditioned_values = DynamicPPL.conditioned(context)

                # The conditioned values might be a NamedTuple, or a Dict.
                # We convert to a Dict for consistency
                if conditioned_values isa NamedTuple
                    conditioned_values = Dict(
                        VarName{sym}() => val for (sym, val) in pairs(conditioned_values)
                    )
                end

                for (vn, val) in pairs(conditioned_values)
                    # We need to drop the prefix of `var` since in `contextual_isassumption`
                    # it will be threaded through the `PrefixContext` before it reaches
                    # `ConditionContext` with the conditioned variable.
                    vn_without_prefix = if getoptic(vn) isa PropertyLens
                        # Hacky: This assumes that there is exactly one level of prefixing
                        # that we need to undo. This is appropriate for the :condition3
                        # test case above, but is not generally correct.
                        AbstractPPL.unprefix(vn, VarName{getsym(vn)}())
                    else
                        vn
                    end

                    @show DynamicPPL.TestUtils.varname_leaves(vn_without_prefix, val)
                    # Let's check elementwise.
                    for vn_child in
                        DynamicPPL.TestUtils.varname_leaves(vn_without_prefix, val)
                        if getoptic(vn_child)(val) === missing
                            @test contextual_isassumption(context, vn_child)
                        else
                            @test !contextual_isassumption(context, vn_child)
                        end
                    end
                end
            end
        end
    end

    @testset "getconditioned_nested & hasconditioned_nested" begin
        @testset "$name" for (name, context) in contexts
            fake_vn = VarName{gensym(:x)}()
            @test !hasconditioned_nested(context, fake_vn)
            @test_throws ErrorException getconditioned_nested(context, fake_vn)

            if any(Base.Fix2(isa, ConditionContext), context)
                # `ConditionContext` specific.

                # Let's first extract the conditioned variables.
                conditioned_values = DynamicPPL.conditioned(context)
                # The conditioned values might be a NamedTuple, or a Dict.
                # We convert to a Dict for consistency
                if conditioned_values isa NamedTuple
                    conditioned_values = Dict(
                        VarName{sym}() => val for (sym, val) in pairs(conditioned_values)
                    )
                end

                for (vn, val) in pairs(conditioned_values)
                    # We need to drop the prefix of `var` since in `contextual_isassumption`
                    # it will be threaded through the `PrefixContext` before it reaches
                    # `ConditionContext` with the conditioned variable.
                    vn_without_prefix = if getoptic(vn) isa PropertyLens
                        # Hacky: This assumes that there is exactly one level of prefixing
                        # that we need to undo. This is appropriate for the :condition3
                        # test case above, but is not generally correct.
                        AbstractPPL.unprefix(vn, VarName{getsym(vn)}())
                    else
                        vn
                    end

                    for vn_child in
                        DynamicPPL.TestUtils.varname_leaves(vn_without_prefix, val)
                        # `vn_child` should be in `context`.
                        @test hasconditioned_nested(context, vn_child)
                        # Value should be the same as extracted above.
                        @test getconditioned_nested(context, vn_child) ===
                            getoptic(vn_child)(val)
                    end
                end
            end
        end
    end

    @testset "PrefixContext" begin
        @testset "prefixing" begin
            ctx = @inferred PrefixContext{:a}(
                PrefixContext{:b}(
                    PrefixContext{:c}(
                        PrefixContext{:d}(
                            PrefixContext{:e}(PrefixContext{:f}(DefaultContext()))
                        ),
                    ),
                ),
            )
            vn = VarName{:x}()
            vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
            @test vn_prefixed == @varname(a.b.c.d.e.f.x)

            vn = VarName{:x}(((1,),))
            vn_prefixed = @inferred DynamicPPL.prefix(ctx, vn)
            @test vn_prefixed == @varname(a.b.c.d.e.f.x[1])
        end

        @testset "nested within arbitrary context stacks" begin
            vn = @varname(x[1])
            ctx1 = PrefixContext{:a}(DefaultContext())
            @test DynamicPPL.prefix(ctx1, vn) == @varname(a.x[1])
            ctx2 = SamplingContext(ctx1)
            @test DynamicPPL.prefix(ctx2, vn) == @varname(a.x[1])
            ctx3 = PrefixContext{:b}(ctx2)
            @test DynamicPPL.prefix(ctx3, vn) == @varname(b.a.x[1])
            ctx4 = DynamicPPL.ValuesAsInModelContext(OrderedDict(), false, ctx3)
            @test DynamicPPL.prefix(ctx4, vn) == @varname(b.a.x[1])
        end

        @testset "evaluation: $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            prefix = :my_prefix
            context = DynamicPPL.PrefixContext{prefix}(SamplingContext())
            # Sample with the context.
            varinfo = DynamicPPL.VarInfo()
            DynamicPPL.evaluate!!(model, varinfo, context)
            # Extract the resulting varnames
            vns_actual = Set(keys(varinfo))

            # Extract the ground truth varnames
            vns_expected = Set([
                AbstractPPL.prefix(vn, VarName{prefix}()) for
                vn in DynamicPPL.TestUtils.varnames(model)
            ])

            # Check that all variables are prefixed correctly.
            @test vns_actual == vns_expected
        end
    end

    @testset "SamplingContext" begin
        context = SamplingContext(Random.default_rng(), SampleFromPrior(), DefaultContext())
        @test context isa SamplingContext

        # convenience constructors
        @test SamplingContext() == context
        @test SamplingContext(Random.default_rng()) == context
        @test SamplingContext(SampleFromPrior()) == context
        @test SamplingContext(DefaultContext()) == context
        @test SamplingContext(Random.default_rng(), SampleFromPrior()) == context
        @test SamplingContext(Random.default_rng(), DefaultContext()) == context
        @test SamplingContext(SampleFromPrior(), DefaultContext()) == context
        @test SamplingContext(SampleFromPrior(), DefaultContext()) == context
        @test EnzymeCore.EnzymeRules.inactive_type(typeof(context))
    end

    @testset "ConditionContext" begin
        @testset "Nesting" begin
            @testset "NamedTuple" begin
                n1 = (x=1, y=2)
                n2 = (x=3,)
                # Values from outer context should override inner one
                ctx1 = ConditionContext(n1, ConditionContext(n2))
                @test ctx1.values == (x=1, y=2)
                # Check that the two ConditionContexts are collapsed
                @test childcontext(ctx1) isa DefaultContext
                # Then test the nesting the other way round
                ctx2 = ConditionContext(n2, ConditionContext(n1))
                @test ctx2.values == (x=3, y=2)
                @test childcontext(ctx2) isa DefaultContext
            end

            @testset "Dict" begin
                # Same tests as NamedTuple above
                d1 = Dict(@varname(x) => 1, @varname(y) => 2)
                d2 = Dict(@varname(x) => 3)
                ctx1 = ConditionContext(d1, ConditionContext(d2))
                @test ctx1.values == Dict(@varname(x) => 1, @varname(y) => 2)
                @test childcontext(ctx1) isa DefaultContext
                ctx2 = ConditionContext(d2, ConditionContext(d1))
                @test ctx2.values == Dict(@varname(x) => 3, @varname(y) => 2)
                @test childcontext(ctx2) isa DefaultContext
            end
        end

        @testset "decondition_context" begin
            @testset "NamedTuple" begin
                ctx = ConditionContext((x=1, y=2, z=3))
                # Decondition all variables
                @test decondition_context(ctx) isa DefaultContext
                # Decondition only some variables
                dctx = decondition_context(ctx, :x)
                @test dctx isa ConditionContext
                @test dctx.values == (y=2, z=3)
                dctx = decondition_context(ctx, :y, :z)
                @test dctx isa ConditionContext
                @test dctx.values == (x=1,)
                # Decondition all variables manually
                @test decondition_context(ctx, :x, :y, :z) isa DefaultContext
            end

            @testset "Dict" begin
                ctx = ConditionContext(
                    Dict(@varname(x) => 1, @varname(y) => 2, @varname(z) => 3)
                )
                # Decondition all variables
                @test decondition_context(ctx) isa DefaultContext
                # Decondition only some variables
                dctx = decondition_context(ctx, @varname(x))
                @test dctx isa ConditionContext
                @test dctx.values == Dict(@varname(y) => 2, @varname(z) => 3)
                dctx = decondition_context(ctx, @varname(y), @varname(z))
                @test dctx isa ConditionContext
                @test dctx.values == Dict(@varname(x) => 1)
                # Decondition all variables manually
                @test decondition_context(ctx, @varname(x), @varname(y), @varname(z)) isa
                    DefaultContext
            end

            @testset "Nesting" begin
                ctx = ConditionContext(
                    (x=1, y=2), ConditionContext(Dict(@varname(a) => 3, @varname(b) => 4))
                )
                # Decondition an outer variable
                dctx = decondition_context(ctx, :x)
                @test dctx.values == (y=2,)
                @test childcontext(dctx).values == Dict(@varname(a) => 3, @varname(b) => 4)
                # Decondition an inner variable
                dctx = decondition_context(ctx, @varname(a))
                @test dctx.values == (x=1, y=2)
                @test childcontext(dctx).values == Dict(@varname(b) => 4)
                # Try deconditioning everything
                dctx = decondition_context(ctx)
                @test dctx isa DefaultContext
            end
        end
    end

    @testset "FixedContext" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            retval = model()
            s, m = retval.s, retval.m

            # Keword approach.
            model_fixed = DynamicPPL.fix(model; s=s)
            @test model_fixed().s == s
            @test model_fixed().m != m
            # A fixed variable should not contribute at all to the logjoint.
            # Assuming `condition` is correctly implemented, the following should hold.
            @test logprior(model_fixed, (; m)) == logprior(condition(model; s=s), (; m))

            # Positional approach.
            model_fixed = DynamicPPL.fix(model, (; s))
            @test model_fixed().s == s
            @test model_fixed().m != m
            @test logprior(model_fixed, (; m)) == logprior(condition(model; s=s), (; m))

            # Pairs approach.
            model_fixed = DynamicPPL.fix(model, @varname(s) => s)
            @test model_fixed().s == s
            @test model_fixed().m != m
            @test logprior(model_fixed, (; m)) == logprior(condition(model; s=s), (; m))

            # Dictionary approach.
            model_fixed = DynamicPPL.fix(model, Dict(@varname(s) => s))
            @test model_fixed().s == s
            @test model_fixed().m != m
            @test logprior(model_fixed, (; m)) == logprior(condition(model; s=s), (; m))
        end
    end
end
