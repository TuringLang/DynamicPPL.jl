module DynamicPPLCondFixContextTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using DynamicPPL: CondFixContext, Condition, Fix, PrefixContext
using Distributions
using LinearAlgebra: I
using Test

# Useful test context types. Note we have to overload equality because otherwise two VNTs
# with different masked entries can compare different under the default struct equality.
struct MyLeafContext <: AbstractContext end
struct MyParentContext{C<:AbstractContext} <: AbstractParentContext
    child::C
end
Base.:(==)(c1::MyParentContext, c2::MyParentContext) = c1.child == c2.child
Base.isequal(c1::MyParentContext, c2::MyParentContext) = isequal(c1.child, c2.child)
DynamicPPL.childcontext(ctx::MyParentContext) = ctx.child
DynamicPPL.setchildcontext(::MyParentContext, child) = MyParentContext(child)

@testset "CondFixContext" begin
    @testset "external facing condition & fix API" begin
        @model function demo_cond_fix()
            x ~ Normal()
            return y ~ Normal(x)
        end
        model = demo_cond_fix()

        # Assume that we've either conditioned or fixed `x` inside `op_model`.
        function test_logp_correct(
            op::Union{typeof(condition),typeof(fix)}, op_model::Model, x
        )
            y = 1.0
            eval_values = VarNamedTuple(; y=y)
            if op === condition
                # x should contribute to the log-likelihood, but `y` should still be
                # in the log-prior.
                @test logprior(op_model, eval_values) == logpdf(Normal(x), y)
                @test loglikelihood(op_model, eval_values) == logpdf(Normal(), x)
            else
                # `x` should not contribute at all
                @test logprior(op_model, eval_values) == logpdf(Normal(x), y)
                @test iszero(loglikelihood(op_model, eval_values))
            end
        end

        @testset "$op" for op in (condition, fix)
            @testset "Keyword arguments" begin
                x = 0.5
                op_model = op(model; x=x)
                test_logp_correct(op, op_model, x)
            end

            @testset "VarNamedTuple as positional argument" begin
                x = 0.5
                op_model = op(model, VarNamedTuple(; x=x))
                test_logp_correct(op, op_model, x)
                if op === condition
                    # Also test with `|`
                    test_logp_correct(condition, (model | VarNamedTuple(; x=x)), x)
                end
            end

            @testset "NamedTuple as positional argument" begin
                x = 0.5
                op_model = op(model, (; x))
                test_logp_correct(op, op_model, x)
                if op === condition
                    test_logp_correct(condition, (model | (; x=x)), x)
                end
            end

            @testset "Dict as positional argument" begin
                x = 0.5
                op_model = op(model, Dict(@varname(x) => x))
                test_logp_correct(op, op_model, x)
                if op === condition
                    test_logp_correct(condition, (model | Dict(@varname(x) => x)), x)
                end
            end

            @testset "Pair as positional argument" begin
                x = 0.5
                op_model = op(model, @varname(x) => x)
                test_logp_correct(op, op_model, x)
                if op === condition
                    test_logp_correct(condition, (model | (@varname(x) => x)), x)
                end
            end
        end
    end

    @testset "CondFixContext constructor" begin
        @testset "Doesn't create a new context if values are empty" begin
            for ctx_type in (Condition, Fix)
                ctx = @inferred(CondFixContext{ctx_type}(VarNamedTuple(), MyLeafContext()))
                @test ctx == MyLeafContext()
            end
        end

        @testset "Merging contexts of the same type" begin
            vnt = VarNamedTuple(; x=1, y=2)
            n2 = VarNamedTuple(; y=3, z=4)
            for ctx_type in (Condition, Fix)
                ctx = @inferred(CondFixContext{ctx_type}(vnt, CondFixContext{ctx_type}(n2)))
                @test Set(keys(ctx.values)) == Set([@varname(x), @varname(y), @varname(z)])
                @test ctx.values[@varname(x)] == 1
                @test ctx.values[@varname(y)] == 2  # Outer context takes precedence
                @test ctx.values[@varname(z)] == 4
                @test childcontext(ctx) isa DefaultContext
            end
        end
    end

    @testset "{has,get}_cf_value{,_nested}" begin
        # Note that we test type stability for all of these. This means that we can't have
        # just any random VNT though because type stability is only as good as the VNT
        # inside it. For example if we have `y[2] := "not an int"` the @inferred tests will
        # break.
        @testset "$ctx_type" for (
            ctx_type, nonnested_hasfunc, nested_hasfunc, nonnested_getfunc, nested_getfunc
        ) in [
            (
                Condition,
                DynamicPPL.hasconditioned,
                DynamicPPL.hasconditioned_nested,
                DynamicPPL.getconditioned,
                DynamicPPL.getconditioned_nested,
            ),
            (
                Fix,
                DynamicPPL.hasfixed,
                DynamicPPL.hasfixed_nested,
                DynamicPPL.getfixed,
                DynamicPPL.getfixed_nested,
            ),
        ]
            @testset "can find values in current context" begin
                vnt = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                ctx = CondFixContext{ctx_type}(vnt)
                # Non-nested versions
                @test @inferred(DynamicPPL.has_cf_value(ctx_type, ctx, @varname(x)))
                @test @inferred(DynamicPPL.has_cf_value(ctx_type, ctx, @varname(y[1])))
                @test @inferred(DynamicPPL.get_cf_value(ctx_type, ctx, @varname(x))) == 1
                @test @inferred(DynamicPPL.get_cf_value(ctx_type, ctx, @varname(y[1]))) == 2
                @test !(@inferred(DynamicPPL.has_cf_value(ctx_type, ctx, @varname(z))))
                @test @inferred(nonnested_hasfunc(ctx, @varname(x)))
                @test @inferred(nonnested_hasfunc(ctx, @varname(y[1])))
                @test @inferred(nonnested_getfunc(ctx, @varname(x))) == 1
                @test @inferred(nonnested_getfunc(ctx, @varname(y[1]))) == 2
                @test !(@inferred(nonnested_hasfunc(ctx, @varname(z))))
                # Nested versions
                @test @inferred(DynamicPPL.has_cf_value_nested(ctx_type, ctx, @varname(x)))
                @test @inferred(
                    DynamicPPL.has_cf_value_nested(ctx_type, ctx, @varname(y[1]))
                )
                @test @inferred(
                    DynamicPPL.get_cf_value_nested(ctx_type, ctx, @varname(x))
                ) == 1
                @test @inferred(
                    DynamicPPL.get_cf_value_nested(ctx_type, ctx, @varname(y[1]))
                ) == 2
                @test !(@inferred(
                    DynamicPPL.has_cf_value_nested(ctx_type, ctx, @varname(z))
                ))
                @test @inferred(nested_hasfunc(ctx, @varname(x)))
                @test @inferred(nested_hasfunc(ctx, @varname(y[1])))
                @test @inferred(nested_getfunc(ctx, @varname(x))) == 1
                @test @inferred(nested_getfunc(ctx, @varname(y[1]))) == 2
                @test !(@inferred(nested_hasfunc(ctx, @varname(z))))
            end

            @testset "does (or does not) find values in nested context" begin
                inner_vnt = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                outer_vnt = VarNamedTuple(; a="a")
                # We need to stick a MyParentContext to separate the two CondFixContexts,
                # otherwise they'll be merged into one.
                ctx = CondFixContext{ctx_type}(
                    outer_vnt, MyParentContext(CondFixContext{ctx_type}(inner_vnt))
                )
                # Since the conditioned values are in the inner child context, they should
                # not be found by the non-nested versions.
                @test @inferred(DynamicPPL.has_cf_value(ctx_type, ctx, @varname(a)))
                @test @inferred(DynamicPPL.get_cf_value(ctx_type, ctx, @varname(a))) == "a"
                @test !(@inferred(DynamicPPL.has_cf_value(ctx_type, ctx, @varname(x))))
                @test !(@inferred(DynamicPPL.has_cf_value(ctx_type, ctx, @varname(y[1]))))
                @test @inferred(nonnested_hasfunc(ctx, @varname(a)))
                @test @inferred(nonnested_getfunc(ctx, @varname(a))) == "a"
                @test !(@inferred(nonnested_hasfunc(ctx, @varname(x))))
                @test !(@inferred(nonnested_hasfunc(ctx, @varname(y[1]))))
                # The nested versions should find everything
                @test @inferred(DynamicPPL.has_cf_value_nested(ctx_type, ctx, @varname(a)))
                @test @inferred(
                    DynamicPPL.get_cf_value_nested(ctx_type, ctx, @varname(a))
                ) == "a"
                @test @inferred(DynamicPPL.has_cf_value_nested(ctx_type, ctx, @varname(x)))
                @test @inferred(
                    DynamicPPL.get_cf_value_nested(ctx_type, ctx, @varname(x))
                ) == 1
                @test @inferred(
                    DynamicPPL.has_cf_value_nested(ctx_type, ctx, @varname(y[1]))
                )
                @test @inferred(
                    DynamicPPL.get_cf_value_nested(ctx_type, ctx, @varname(y[1]))
                ) == 2
                @test @inferred(nested_hasfunc(ctx, @varname(a)))
                @test @inferred(nested_getfunc(ctx, @varname(a))) == "a"
                @test @inferred(nested_hasfunc(ctx, @varname(x)))
                @test @inferred(nested_getfunc(ctx, @varname(x))) == 1
                @test @inferred(nested_hasfunc(ctx, @varname(y[1])))
                @test @inferred(nested_getfunc(ctx, @varname(y[1]))) == 2
            end
        end
    end

    @testset "all_cf_values" begin
        @testset "$ctx_type" for (ctx_type, ext_func) in [
            (Condition, DynamicPPL.conditioned), (Fix, DynamicPPL.fixed)
        ]
            @testset "non-nested" begin
                vnt = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                ctx = CondFixContext{ctx_type}(vnt)
                all_vals = @inferred(DynamicPPL.all_cf_values(ctx_type, ctx))
                @test all_vals == vnt
                all_vals = @inferred(ext_func(ctx))
                @test all_vals == vnt
            end

            @testset "nested" begin
                inner_vnt = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                outer_vnt = VarNamedTuple(; a="a")
                # We need to stick a MyParentContext to separate the two CondFixContexts,
                # otherwise they'll be merged into one.
                ctx = CondFixContext{ctx_type}(
                    outer_vnt, MyParentContext(CondFixContext{ctx_type}(inner_vnt))
                )
                all_vals = @inferred(DynamicPPL.all_cf_values(ctx_type, ctx))
                expected = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                    a := "a"
                end
                # The order of keys may differ, so compare as sets
                @test Set(keys(all_vals)) == Set(keys(expected))
                @test all_vals[@varname(x)] == expected[@varname(x)]
                @test all_vals[@varname(y[1])] == expected[@varname(y[1])]
                @test all_vals[@varname(a)] == expected[@varname(a)]
            end
        end
    end

    @testset "remove_cf_values" begin
        @testset "$ctx_type" for (ctx_type, deop) in [
            (Condition, DynamicPPL.decondition_context), (Fix, DynamicPPL.unfix_context)
        ]
            @testset "Remove all values" begin
                values = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                ctx = CondFixContext{ctx_type}(values, MyLeafContext())
                @test (@inferred(DynamicPPL.remove_cf_values(ctx_type, ctx))) ==
                    MyLeafContext()
                @test (@inferred(deop(ctx))) == MyLeafContext()
            end

            @testset "Remove specific values (which happens to be all of them)" begin
                values = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                # This is not type stable because we don't know if we're removing all
                # values.
                # TODO(penelopeysm): In this case it feels like we should be able to make it
                # type stable. I think this will rely on a more advanced implementation of
                # `remove_cf_values` that doesn't iterate through keys, but rather calls
                # delete!! on a VNT (and we can hopefully make delete!! type stable).
                ctx = CondFixContext{ctx_type}(values, MyLeafContext())
                @test DynamicPPL.remove_cf_values(
                    ctx_type, ctx, @varname(x), @varname(y[1])
                ) == MyLeafContext()
                @test deop(ctx, @varname(x), @varname(y[1])) == MyLeafContext()
            end

            @testset "Remove a superset of a key" begin
                values = @vnt begin
                    @template y = zeros(2)
                    y[1] := 2
                    y[2].a := "hello world"
                end
                # Removing `y` should just delete everything.
                ctx = CondFixContext{ctx_type}(values, MyLeafContext())
                @test DynamicPPL.remove_cf_values(ctx_type, ctx, @varname(y)) ==
                    MyLeafContext()
                @test deop(ctx, @varname(y)) == MyLeafContext()
            end

            @testset "Removing all nested values" begin
                inner_vnt = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                outer_vnt = VarNamedTuple(; a="a")
                ctx = CondFixContext{ctx_type}(
                    outer_vnt,
                    MyParentContext(CondFixContext{ctx_type}(inner_vnt, MyLeafContext())),
                )
                @test @inferred(DynamicPPL.remove_cf_values(ctx_type, ctx)) ==
                    MyParentContext(MyLeafContext())
                @test @inferred(deop(ctx)) == MyParentContext(MyLeafContext())
            end

            @testset "Removing some nested values" begin
                inner_vnt = @vnt begin
                    @template y = zeros(2)
                    x := 1
                    y[1] := 2
                end
                inner_vnt_no_x = @vnt begin
                    @template y = zeros(2)
                    y[1] := 2
                end
                outer_vnt = VarNamedTuple(; a="a")
                ctx = CondFixContext{ctx_type}(
                    outer_vnt,
                    MyParentContext(CondFixContext{ctx_type}(inner_vnt, MyLeafContext())),
                )
                expected_ctx = CondFixContext{ctx_type}(
                    outer_vnt,
                    MyParentContext(
                        CondFixContext{ctx_type}(inner_vnt_no_x, MyLeafContext())
                    ),
                )
                # Then remove `x`
                @test DynamicPPL.remove_cf_values(ctx_type, ctx, @varname(x)) ==
                    expected_ctx
                @test deop(ctx, @varname(x)) == expected_ctx
            end
        end
    end

    @testset "Can fix immutable data safely" begin
        # extra regression test for
        # https://github.com/TuringLang/DynamicPPL.jl/issues/1176#issuecomment-3648871018
        @model function ntfix()
            m ~ Normal()
            data = (; x=undef)
            data.x ~ Normal(m, 1.0)
            return data.x
        end
        fixm = DynamicPPL.fix(ntfix(), (; data=(; x=5.0)))
        retval, vi = DynamicPPL.init!!(fixm, VarInfo())
        # The fixed data should overwrite the NamedTuple that came before it
        @test retval == 5.0
        @test vi isa VarInfo
        @test vi[@varname(m)] isa Real
    end

    @testset "can condition/fix on each individual part of a multivariate" begin
        @model function mvnorm()
            x ~ MvNormal(zeros(3), I)
            return x
        end
        @testset "with templating" begin
            vnt = @vnt begin
                @template x = zeros(3)
                x[1] := 1.0
                x[2] := 2.0
                x[3] := 3.0
            end
            for op in (condition, fix)
                op_model = op(mvnorm(), vnt)
                @test op_model() == [1.0, 2.0, 3.0]
            end
        end
        @testset "without templating" begin
            # Note that this is only guaranteed to work as long as rand(distribution)
            # returns a Base.Vector.
            vnt = @vnt begin
                x[1] := 1.0
                x[2] := 2.0
                x[3] := 3.0
            end
            for op in (condition, fix)
                op_model = op(mvnorm(), vnt)
                @test op_model() == [1.0, 2.0, 3.0]
            end
        end
    end
end

@testset "PrefixContext + CondFixContext interactions" begin
    @testset "prefix_cond_and_fixed_variables" begin
        c1 = CondFixContext{Condition}(VarNamedTuple(; c=1, d=2))
        c1_prefixed = DynamicPPL.prefix_cond_and_fixed_variables(c1, @varname(a))
        @test c1_prefixed isa CondFixContext{Condition}
        @test DynamicPPL.childcontext(c1_prefixed) isa DefaultContext
        @test c1_prefixed.values[@varname(a.c)] == 1
        @test c1_prefixed.values[@varname(a.d)] == 2

        c2 = CondFixContext{Fix}(VarNamedTuple(; f=1, g=2))
        c2_prefixed = DynamicPPL.prefix_cond_and_fixed_variables(c2, @varname(a))
        @test c2_prefixed isa CondFixContext{Fix}
        @test DynamicPPL.childcontext(c2_prefixed) isa DefaultContext
        @test c2_prefixed.values[@varname(a.f)] == 1
        @test c2_prefixed.values[@varname(a.g)] == 2

        c3 = CondFixContext{Condition}(
            VarNamedTuple(; c=1, d=2), CondFixContext{Fix}(VarNamedTuple(; f=1, g=2))
        )
        c3_prefixed = DynamicPPL.prefix_cond_and_fixed_variables(c3, @varname(a))
        c3_prefixed_child = DynamicPPL.childcontext(c3_prefixed)
        @test c3_prefixed isa CondFixContext{Condition}
        @test c3_prefixed.values[@varname(a.c)] == 1
        @test c3_prefixed.values[@varname(a.d)] == 2
        @test c3_prefixed_child isa CondFixContext{Fix}
        @test c3_prefixed_child.values[@varname(a.f)] == 1
        @test c3_prefixed_child.values[@varname(a.g)] == 2
        @test DynamicPPL.childcontext(c3_prefixed_child) isa DefaultContext
    end

    @testset "collapse_prefix_stack" begin
        # Utility function to make sure that there are no PrefixContexts in
        # the context stack.
        has_no_prefixcontexts(::PrefixContext) = false
        function has_no_prefixcontexts(ctx::AbstractParentContext)
            return has_no_prefixcontexts(childcontext(ctx))
        end
        has_no_prefixcontexts(::AbstractContext) = true

        @testset "$ctx_type" for ctx_type in (Condition, Fix)
            @testset "Prefix -> CondFix" begin
                vals = VarNamedTuple(; c=1, d=2)
                c1 = PrefixContext(@varname(a), CondFixContext{ctx_type}(vals))
                c1 = @inferred(DynamicPPL.collapse_prefix_stack(c1))
                @test has_no_prefixcontexts(c1)
                c1_vals = DynamicPPL.all_cf_values(ctx_type, c1)
                @test length(keys(c1_vals)) == 2
                @test c1_vals[@varname(a.c)] == 1
                @test c1_vals[@varname(a.d)] == 2
            end

            @testset "CondFix -> Prefix" begin
                vals = VarNamedTuple(; c=1, d=2)
                c2 = CondFixContext{ctx_type}(vals, PrefixContext(@varname(a)))
                c2 = @inferred(DynamicPPL.collapse_prefix_stack(c2))
                @test has_no_prefixcontexts(c2)
                c2_vals = DynamicPPL.all_cf_values(ctx_type, c2)
                @test length(keys(c2_vals)) == 2
                @test c2_vals[@varname(c)] == 1
                @test c2_vals[@varname(d)] == 2
            end

            @testset "Prefix -> CondFix -> Prefix -> CondFix" begin
                c5 = PrefixContext(
                    @varname(a),
                    CondFixContext{ctx_type}(
                        VarNamedTuple(; c=1),
                        PrefixContext(
                            @varname(b), CondFixContext{ctx_type}(VarNamedTuple(; d=2))
                        ),
                    ),
                )
                c5 = @inferred(DynamicPPL.collapse_prefix_stack(c5))
                @test has_no_prefixcontexts(c5)
                c5_vals = DynamicPPL.all_cf_values(ctx_type, c5)
                @test length(keys(c5_vals)) == 2
                @test c5_vals[@varname(a.c)] == 1
                @test c5_vals[@varname(a.b.d)] == 2
            end

            @testset "Prefix -> CondFix -> Prefix -> FixCond" begin
                inner_ctx_type = ctx_type === Condition ? Fix : Condition
                # the inner FixCond is the opposite type from the outer CondFix
                c6 = PrefixContext(
                    @varname(a),
                    CondFixContext{ctx_type}(
                        VarNamedTuple(; c=1),
                        PrefixContext(
                            @varname(b),
                            CondFixContext{inner_ctx_type}(VarNamedTuple(; d=2)),
                        ),
                    ),
                )
                c6 = @inferred(DynamicPPL.collapse_prefix_stack(c6))
                @test has_no_prefixcontexts(c6)
                @test only(keys(DynamicPPL.all_cf_values(ctx_type, c6))) == @varname(a.c)
                @test DynamicPPL.all_cf_values(ctx_type, c6)[@varname(a.c)] == 1
                @test only(keys(DynamicPPL.all_cf_values(inner_ctx_type, c6))) ==
                    @varname(a.b.d)
                @test DynamicPPL.all_cf_values(inner_ctx_type, c6)[@varname(a.b.d)] == 2
            end
        end
    end

    @testset "Deconditioning prefixed variables" begin
        @testset "$ctx_type" for (ctx_type, deop) in [
            (Condition, DynamicPPL.decondition_context), (Fix, DynamicPPL.unfix_context)
        ]
            ctx = PrefixContext(
                @varname(a), CondFixContext{ctx_type}(VarNamedTuple(; x=1), MyLeafContext())
            )
            expected_ctx = PrefixContext(@varname(a), MyLeafContext())
            # Remove everything
            @test @inferred(DynamicPPL.remove_cf_values(ctx_type, ctx)) == expected_ctx
            @test @inferred(deop(ctx)) == expected_ctx
            # Remove the entire prefix
            @test DynamicPPL.remove_cf_values(ctx_type, ctx, @varname(a)) == expected_ctx
            @test deop(ctx, @varname(a)) == expected_ctx
            # Remove the prefixed variable
            @test DynamicPPL.remove_cf_values(ctx_type, ctx, @varname(a.x)) == expected_ctx
            @test deop(ctx, @varname(a.x)) == expected_ctx
            # Remove the unprefixed variable (should do nothing)
            @test DynamicPPL.remove_cf_values(ctx_type, ctx, @varname(x)) == ctx
            @test deop(ctx, @varname(x)) == ctx
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
