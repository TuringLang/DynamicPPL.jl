using Test, DynamicPPL, Accessors
using AbstractPPL: getoptic, hasvalue, getvalue
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
    FixedContext,
    ConditionContext,
    decondition_context,
    hasconditioned,
    getconditioned,
    conditioned,
    fixed,
    hasconditioned_nested,
    getconditioned_nested,
    collapse_prefix_stack,
    prefix_cond_and_fixed_variables
using LinearAlgebra: I
using Random: Xoshiro

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
    contexts = Dict(
        :default => DefaultContext(),
        :testparent => DynamicPPL.TestUtils.TestParentContext(DefaultContext()),
        :sampling => SamplingContext(),
        :prefix => PrefixContext(@varname(x)),
        :condition1 => ConditionContext((x=1.0,)),
        :condition2 => ConditionContext(
            (x=1.0,), DynamicPPL.TestUtils.TestParentContext(ConditionContext((y=2.0,)))
        ),
        :condition3 => ConditionContext(
            (x=1.0,),
            PrefixContext(@varname(a), ConditionContext(Dict(@varname(y) => 2.0))),
        ),
        :condition4 => ConditionContext((x=[1.0, missing],)),
    )

    @testset "$(name)" for (name, context) in contexts
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            DynamicPPL.TestUtils.test_context(context, model)
        end
    end

    @testset "extracting conditioned values" begin
        # This testset tests `contextual_isassumption`, `getconditioned_nested`, and
        # `hasconditioned_nested`.

        @testset "$(name)" for (name, context) in contexts
            # If the varname doesn't exist, it should always be an assumption.
            fake_vn = VarName{gensym(:x)}()
            @test contextual_isassumption(context, fake_vn)
            @test !hasconditioned_nested(context, fake_vn)
            @test_throws ErrorException getconditioned_nested(context, fake_vn)

            if any(Base.Fix2(isa, ConditionContext), context)
                # We have a `ConditionContext` among us.
                # Let's first extract the conditioned variables.
                conditioned_values = DynamicPPL.conditioned(context)

                # The conditioned values might be a NamedTuple, or a Dict.
                # We convert to a Dict for consistency
                conditioned_values = DynamicPPL.to_varname_dict(conditioned_values)

                # Extract all conditioned variables. We also use varname_leaves
                # here to split up arrays which could potentially have some,
                # but not all, elements being `missing`.
                conditioned_vns = mapreduce(
                    p -> DynamicPPL.TestUtils.varname_leaves(p.first, p.second),
                    vcat,
                    pairs(conditioned_values),
                )

                # We can now loop over them to check which ones are missing. We use
                # `getvalue` to handle the awkward case where sometimes
                # `conditioned_values` contains the full Varname (e.g. `a.x`) and
                # sometimes only the main symbol (e.g. it contains `x` when
                # `vn` is `x[1]`)
                for vn in conditioned_vns
                    val = getvalue(conditioned_values, vn)
                    # These VarNames are present in the conditioning values, so
                    # we should always be able to extract the value.
                    @test hasconditioned_nested(context, vn)
                    @test getconditioned_nested(context, vn) === val
                    # However, the return value of contextual_isassumption depends on
                    # whether the value is missing or not.
                    if ismissing(val)
                        @test contextual_isassumption(context, vn)
                    else
                        @test !contextual_isassumption(context, vn)
                    end
                end
            end
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
            ctx2 = SamplingContext(ctx1)
            @test DynamicPPL.prefix(ctx2, vn) == @varname(a.x[1])
            ctx3 = PrefixContext(@varname(b), ctx2)
            @test DynamicPPL.prefix(ctx3, vn) == @varname(b.a.x[1])
            ctx4 = DynamicPPL.SamplingContext(ctx3)
            @test DynamicPPL.prefix(ctx4, vn) == @varname(b.a.x[1])
        end

        @testset "prefix_and_strip_contexts" begin
            vn = @varname(x[1])
            ctx1 = PrefixContext(@varname(a))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx1, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == DefaultContext()

            ctx2 = FixedContext((b=4,), PrefixContext(@varname(a)))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx2, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == FixedContext((b=4,))

            ctx3 = PrefixContext(@varname(a), ConditionContext((a=1,)))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx3, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == ConditionContext((a=1,))

            ctx4 = FixedContext(
                (b=4,), PrefixContext(@varname(a), ConditionContext((a=1,)))
            )
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx4, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == FixedContext((b=4,), ConditionContext((a=1,)))
        end

        @testset "evaluation: $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            prefix_vn = @varname(my_prefix)
            context = DynamicPPL.PrefixContext(prefix_vn, DefaultContext())
            new_model = contextualize(model, context)
            # Initialize a new varinfo with the prefixed model
            _, varinfo = DynamicPPL.init!!(new_model, DynamicPPL.VarInfo())
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

    @testset "PrefixContext + Condition/FixedContext interactions" begin
        @testset "prefix_cond_and_fixed_variables" begin
            c1 = ConditionContext((c=1, d=2))
            c1_prefixed = prefix_cond_and_fixed_variables(c1, @varname(a))
            @test c1_prefixed isa ConditionContext
            @test childcontext(c1_prefixed) isa DefaultContext
            @test c1_prefixed.values[@varname(a.c)] == 1
            @test c1_prefixed.values[@varname(a.d)] == 2

            c2 = FixedContext((f=1, g=2))
            c2_prefixed = prefix_cond_and_fixed_variables(c2, @varname(a))
            @test c2_prefixed isa FixedContext
            @test childcontext(c2_prefixed) isa DefaultContext
            @test c2_prefixed.values[@varname(a.f)] == 1
            @test c2_prefixed.values[@varname(a.g)] == 2

            c3 = ConditionContext((c=1, d=2), FixedContext((f=1, g=2)))
            c3_prefixed = prefix_cond_and_fixed_variables(c3, @varname(a))
            c3_prefixed_child = childcontext(c3_prefixed)
            @test c3_prefixed isa ConditionContext
            @test c3_prefixed.values[@varname(a.c)] == 1
            @test c3_prefixed.values[@varname(a.d)] == 2
            @test c3_prefixed_child isa FixedContext
            @test c3_prefixed_child.values[@varname(a.f)] == 1
            @test c3_prefixed_child.values[@varname(a.g)] == 2
            @test childcontext(c3_prefixed_child) isa DefaultContext
        end

        @testset "collapse_prefix_stack" begin
            # Utility function to make sure that there are no PrefixContexts in
            # the context stack.
            function has_no_prefixcontexts(ctx::AbstractContext)
                return !(ctx isa PrefixContext) && (
                    NodeTrait(ctx) isa IsLeaf || has_no_prefixcontexts(childcontext(ctx))
                )
            end

            # Prefix -> Condition
            c1 = PrefixContext(@varname(a), ConditionContext((c=1, d=2)))
            c1 = collapse_prefix_stack(c1)
            @test has_no_prefixcontexts(c1)
            c1_vals = conditioned(c1)
            @test length(c1_vals) == 2
            @test getvalue(c1_vals, @varname(a.c)) == 1
            @test getvalue(c1_vals, @varname(a.d)) == 2

            # Condition -> Prefix
            c2 = ConditionContext((c=1, d=2), PrefixContext(@varname(a)))
            c2 = collapse_prefix_stack(c2)
            @test has_no_prefixcontexts(c2)
            c2_vals = conditioned(c2)
            @test length(c2_vals) == 2
            @test getvalue(c2_vals, @varname(c)) == 1
            @test getvalue(c2_vals, @varname(d)) == 2

            # Prefix -> Fixed
            c3 = PrefixContext(@varname(a), FixedContext((f=1, g=2)))
            c3 = collapse_prefix_stack(c3)
            c3_vals = fixed(c3)
            @test length(c3_vals) == 2
            @test length(c3_vals) == 2
            @test getvalue(c3_vals, @varname(a.f)) == 1
            @test getvalue(c3_vals, @varname(a.g)) == 2

            # Fixed -> Prefix
            c4 = FixedContext((f=1, g=2), PrefixContext(@varname(a)))
            c4 = collapse_prefix_stack(c4)
            @test has_no_prefixcontexts(c4)
            c4_vals = fixed(c4)
            @test length(c4_vals) == 2
            @test getvalue(c4_vals, @varname(f)) == 1
            @test getvalue(c4_vals, @varname(g)) == 2

            # Prefix -> Condition -> Prefix -> Condition
            c5 = PrefixContext(
                @varname(a),
                ConditionContext(
                    (c=1,), PrefixContext(@varname(b), ConditionContext((d=2,)))
                ),
            )
            c5 = collapse_prefix_stack(c5)
            @test has_no_prefixcontexts(c5)
            c5_vals = conditioned(c5)
            @test length(c5_vals) == 2
            @test getvalue(c5_vals, @varname(a.c)) == 1
            @test getvalue(c5_vals, @varname(a.b.d)) == 2

            # Prefix -> Condition -> Prefix -> Fixed
            c6 = PrefixContext(
                @varname(a),
                ConditionContext((c=1,), PrefixContext(@varname(b), FixedContext((d=2,)))),
            )
            c6 = collapse_prefix_stack(c6)
            @test has_no_prefixcontexts(c6)
            @test conditioned(c6) == Dict(@varname(a.c) => 1)
            @test fixed(c6) == Dict(@varname(a.b.d) => 2)
        end
    end

    @testset "InitContext" begin
        empty_varinfos = [
            VarInfo(),
            DynamicPPL.typed_varinfo(VarInfo()),
            VarInfo(DynamicPPL.VarNamedVector()),
            DynamicPPL.typed_vector_varinfo(DynamicPPL.typed_varinfo(VarInfo())),
            SimpleVarInfo(),
            SimpleVarInfo(Dict{VarName,Any}()),
        ]

        @model function test_init_model()
            x ~ Normal()
            y ~ MvNormal(fill(x, 2), I)
            1.0 ~ Normal()
            return nothing
        end
        function test_generating_new_values(strategy::AbstractInitStrategy)
            @testset "generating new values: $(typeof(strategy))" begin
                # Check that init!! can generate values that weren't there
                # previously.
                model = test_init_model()
                for empty_vi in empty_varinfos
                    this_vi = deepcopy(empty_vi)
                    _, vi = DynamicPPL.init!!(model, this_vi, strategy)
                    @test Set(keys(vi)) == Set([@varname(x), @varname(y)])
                    x, y = vi[@varname(x)], vi[@varname(y)]
                    @test x isa Real
                    @test y isa AbstractVector{<:Real}
                    @test length(y) == 2
                    (; logprior, loglikelihood) = getlogp(vi)
                    @test logpdf(Normal(), x) + logpdf(MvNormal(fill(x, 2), I), y) ==
                        logprior
                    @test logpdf(Normal(), 1.0) == loglikelihood
                end
            end
        end
        function test_replacing_values(strategy::AbstractInitStrategy)
            @testset "replacing old values: $(typeof(strategy))" begin
                # Check that init!! can overwrite values that were already there.
                model = test_init_model()
                for empty_vi in empty_varinfos
                    # start by generating some rubbish values
                    vi = deepcopy(empty_vi)
                    old_x, old_y = 100000.00, [300000.00, 500000.00]
                    push!!(vi, @varname(x), old_x, Normal())
                    push!!(vi, @varname(y), old_y, MvNormal(fill(old_x, 2), I))
                    # then overwrite it
                    _, new_vi = DynamicPPL.init!!(model, vi, strategy)
                    new_x, new_y = new_vi[@varname(x)], new_vi[@varname(y)]
                    # check that the values are (presumably) different
                    @test old_x != new_x
                    @test old_y != new_y
                end
            end
        end
        function test_rng_respected(strategy::AbstractInitStrategy)
            @testset "check that RNG is respected: $(typeof(strategy))" begin
                model = test_init_model()
                for empty_vi in empty_varinfos
                    _, vi1 = DynamicPPL.init!!(
                        Xoshiro(468), model, deepcopy(empty_vi), strategy
                    )
                    _, vi2 = DynamicPPL.init!!(
                        Xoshiro(468), model, deepcopy(empty_vi), strategy
                    )
                    _, vi3 = DynamicPPL.init!!(
                        Xoshiro(469), model, deepcopy(empty_vi), strategy
                    )
                    @test vi1[@varname(x)] == vi2[@varname(x)]
                    @test vi1[@varname(y)] == vi2[@varname(y)]
                    @test vi1[@varname(x)] != vi3[@varname(x)]
                    @test vi1[@varname(y)] != vi3[@varname(y)]
                end
            end
        end
        function test_link_status_respected(strategy::AbstractInitStrategy)
            @testset "check that varinfo linking is preserved: $(typeof(strategy))" begin
                @model logn() = a ~ LogNormal()
                model = logn()
                vi = VarInfo(model)
                linked_vi = DynamicPPL.link!!(vi, model)
                _, new_vi = DynamicPPL.init!!(model, linked_vi, strategy)
                @test DynamicPPL.istrans(new_vi)
                # this is the unlinked value, since it uses `getindex`
                a = new_vi[@varname(a)]
                # internal logjoint should correspond to the transformed value
                @test isapprox(
                    DynamicPPL.getlogjoint_internal(new_vi), logpdf(Normal(), log(a))
                )
                # user logjoint should correspond to the transformed value
                @test isapprox(DynamicPPL.getlogjoint(new_vi), logpdf(LogNormal(), a))
                @test isapprox(
                    only(DynamicPPL.getindex_internal(new_vi, @varname(a))), log(a)
                )
            end
        end

        @testset "PriorInit" begin
            test_generating_new_values(PriorInit())
            test_replacing_values(PriorInit())
            test_rng_respected(PriorInit())
            test_link_status_respected(PriorInit())

            @testset "check that values are within support" begin
                # Not many other sensible checks we can do for priors.
                @model just_unif() = x ~ Uniform(0.0, 1e-7)
                for _ in 1:100
                    _, vi = DynamicPPL.init!!(just_unif(), VarInfo(), PriorInit())
                    @test vi[@varname(x)] isa Real
                    @test 0.0 <= vi[@varname(x)] <= 1e-7
                end
            end
        end

        @testset "UniformInit" begin
            test_generating_new_values(UniformInit())
            test_replacing_values(UniformInit())
            test_rng_respected(UniformInit())
            test_link_status_respected(UniformInit())

            @testset "check that bounds are respected" begin
                @testset "unconstrained" begin
                    umin, umax = -1.0, 1.0
                    @model just_norm() = x ~ Normal()
                    for _ in 1:100
                        _, vi = DynamicPPL.init!!(
                            just_norm(), VarInfo(), UniformInit(umin, umax)
                        )
                        @test vi[@varname(x)] isa Real
                        @test umin <= vi[@varname(x)] <= umax
                    end
                end
                @testset "constrained" begin
                    umin, umax = -1.0, 1.0
                    @model just_beta() = x ~ Beta(2, 2)
                    inv_bijector = inverse(Bijectors.bijector(Beta(2, 2)))
                    tmin, tmax = inv_bijector(umin), inv_bijector(umax)
                    for _ in 1:100
                        _, vi = DynamicPPL.init!!(
                            just_beta(), VarInfo(), UniformInit(umin, umax)
                        )
                        @test vi[@varname(x)] isa Real
                        @test tmin <= vi[@varname(x)] <= tmax
                    end
                end
            end
        end

        @testset "ParamsInit" begin
            test_link_status_respected(ParamsInit((; a=1.0)))
            test_link_status_respected(ParamsInit(Dict(@varname(a) => 1.0)))

            @testset "given full set of parameters" begin
                # test_init_model has x ~ Normal() and y ~ MvNormal(zeros(2), I)
                my_x, my_y = 1.0, [2.0, 3.0]
                params_nt = (; x=my_x, y=my_y)
                params_dict = Dict(@varname(x) => my_x, @varname(y) => my_y)
                model = test_init_model()
                for empty_vi in empty_varinfos
                    _, vi = DynamicPPL.init!!(
                        model, deepcopy(empty_vi), ParamsInit(params_nt)
                    )
                    @test vi[@varname(x)] == my_x
                    @test vi[@varname(y)] == my_y
                    logp_nt = getlogp(vi)
                    _, vi = DynamicPPL.init!!(
                        model, deepcopy(empty_vi), ParamsInit(params_dict)
                    )
                    @test vi[@varname(x)] == my_x
                    @test vi[@varname(y)] == my_y
                    logp_dict = getlogp(vi)
                    @test logp_nt == logp_dict
                end
            end

            @testset "given only partial parameters" begin
                # In this case, we expect `ParamsInit` to use the value of x, and
                # generate a new value for y.
                my_x = 1.0
                params_nt = (; x=my_x)
                params_dict = Dict(@varname(x) => my_x)
                model = test_init_model()
                for empty_vi in empty_varinfos
                    _, vi = DynamicPPL.init!!(
                        Xoshiro(468), model, deepcopy(empty_vi), ParamsInit(params_nt)
                    )
                    @test vi[@varname(x)] == my_x
                    nt_y = vi[@varname(y)]
                    @test nt_y isa AbstractVector{<:Real}
                    @test length(nt_y) == 2
                    _, vi = DynamicPPL.init!!(
                        Xoshiro(469), model, deepcopy(empty_vi), ParamsInit(params_dict)
                    )
                    @test vi[@varname(x)] == my_x
                    dict_y = vi[@varname(y)]
                    @test dict_y isa AbstractVector{<:Real}
                    @test length(dict_y) == 2
                    # the values should be different since we used different seeds
                    @test dict_y != nt_y
                end
            end
        end
    end
end
