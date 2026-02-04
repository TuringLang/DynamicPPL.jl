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
    AbstractParentContext,
    contextual_isassumption,
    PrefixContext,
    CondFixContext,
    Condition,
    Fix,
    decondition_context,
    hasconditioned,
    getconditioned,
    conditioned,
    fixed,
    hasconditioned_nested,
    getconditioned_nested,
    collapse_prefix_stack,
    prefix_cond_and_fixed_variables

# TODO: Should we maybe put this in DPPL itself?
function Base.iterate(context::AbstractParentContext)
    return context, childcontext(context)
end
function Base.iterate(context::AbstractContext)
    return context, nothing
end
function Base.iterate(::AbstractContext, state::AbstractParentContext)
    return state, childcontext(state)
end
function Base.iterate(::AbstractContext, state::AbstractContext)
    return state, nothing
end
function Base.iterate(::AbstractContext, state::Nothing)
    return nothing
end
Base.IteratorSize(::Type{<:AbstractContext}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:AbstractContext}) = Base.EltypeUnknown()

@testset "contexts.jl" begin
    contexts = Dict(
        :default => DefaultContext(),
        :testparent => DynamicPPL.TestUtils.TestParentContext(DefaultContext()),
        :prefix => PrefixContext(@varname(x)),
        :condition1 => CondFixContext{Condition}(VarNamedTuple(; x=1.0)),
        :condition2 => CondFixContext{Condition}(
            VarNamedTuple(; x=1.0),
            DynamicPPL.TestUtils.TestParentContext(
                CondFixContext{Condition}(VarNamedTuple(; y=2.0))
            ),
        ),
        :condition3 => CondFixContext{Condition}(
            VarNamedTuple(; x=1.0),
            PrefixContext(@varname(a), CondFixContext{Condition}(VarNamedTuple(; y=2.0))),
        ),
        :condition4 => CondFixContext{Condition}(VarNamedTuple(; x=[1.0, missing])),
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

            if any(Base.Fix2(isa, CondFixContext{Condition}), context)
                # We have a `ConditionContext` among us.
                # Let's first extract the conditioned variables.
                conditioned_values = DynamicPPL.conditioned(context)

                # Extract all conditioned variables. We also use varname_leaves
                # here to split up arrays which could potentially have some,
                # but not all, elements being `missing`.
                conditioned_vns = mapreduce(
                    p -> AbstractPPL.varname_leaves(p.first, p.second),
                    vcat,
                    pairs(conditioned_values),
                )

                # We can now loop over them to check which ones are missing.
                for vn in conditioned_vns
                    val = conditioned_values[vn]
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
            ctx2 = CondFixContext{Condition}(VarNamedTuple(; b=1), ctx1)
            @test DynamicPPL.prefix(ctx2, vn) == @varname(a.x[1])
            ctx3 = PrefixContext(@varname(b), ctx2)
            @test DynamicPPL.prefix(ctx3, vn) == @varname(b.a.x[1])
            ctx4 = CondFixContext{Fix}(VarNamedTuple(; c=2), ctx3)
            @test DynamicPPL.prefix(ctx4, vn) == @varname(b.a.x[1])
        end

        @testset "prefix_and_strip_contexts" begin
            vn = @varname(x[1])
            ctx1 = PrefixContext(@varname(a))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx1, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == DefaultContext()

            ctx2 = CondFixContext{Fix}(VarNamedTuple(; b=4), PrefixContext(@varname(a)))
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx2, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == CondFixContext{Fix}(VarNamedTuple(; b=4))

            ctx3 = PrefixContext(
                @varname(a), CondFixContext{Condition}(VarNamedTuple(; a=1))
            )
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx3, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == CondFixContext{Condition}(VarNamedTuple(; a=1))

            ctx4 = CondFixContext{Fix}(
                VarNamedTuple(; b=4),
                PrefixContext(@varname(a), CondFixContext{Condition}(VarNamedTuple(; a=1))),
            )
            new_vn, new_ctx = DynamicPPL.prefix_and_strip_contexts(ctx4, vn)
            @test new_vn == @varname(a.x[1])
            @test new_ctx == CondFixContext{Fix}(
                VarNamedTuple(; b=4), CondFixContext{Condition}(VarNamedTuple(; a=1))
            )
        end

        @testset "evaluation: $(model.f)" for model in DynamicPPL.TestUtils.ALL_MODELS
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
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
