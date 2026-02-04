module DynamicPPLContextTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Test, DynamicPPL, Distributions
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
using LinearAlgebra: I
using Random: Xoshiro

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

    @testset "InitContext" begin
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
                empty_vi = VarInfo()
                this_vi = deepcopy(empty_vi)
                _, vi = DynamicPPL.init!!(model, this_vi, strategy)
                @test Set(keys(vi)) == Set([@varname(x), @varname(y)])
                x, y = vi[@varname(x)], vi[@varname(y)]
                @test x isa Real
                @test y isa AbstractVector{<:Real}
                @test length(y) == 2
                (; logprior, loglikelihood) = getlogp(vi)
                @test logpdf(Normal(), x) + logpdf(MvNormal(fill(x, 2), I), y) == logprior
                @test logpdf(Normal(), 1.0) == loglikelihood
            end
        end

        function test_replacing_values(strategy::AbstractInitStrategy)
            @testset "replacing old values: $(typeof(strategy))" begin
                # Check that init!! can overwrite values that were already there.
                model = test_init_model()
                empty_vi = VarInfo()
                # start by generating some rubbish values
                vi = deepcopy(empty_vi)
                old_x, old_y = 100000.00, [300000.00, 500000.00]
                vi = DynamicPPL.setindex_with_dist!!(
                    vi,
                    UntransformedValue(old_x),
                    Normal(),
                    @varname(x),
                    DynamicPPL.NoTemplate(),
                )
                vi = DynamicPPL.setindex_with_dist!!(
                    vi,
                    UntransformedValue(old_y),
                    MvNormal(fill(old_x, 2), I),
                    @varname(y),
                    DynamicPPL.NoTemplate(),
                )
                # then overwrite it
                _, new_vi = DynamicPPL.init!!(model, vi, strategy)
                new_x, new_y = new_vi[@varname(x)], new_vi[@varname(y)]
                # check that the values are (presumably) different
                @test old_x != new_x
                @test old_y != new_y
            end
        end

        function test_rng_respected(strategy::AbstractInitStrategy)
            @testset "check that RNG is respected: $(typeof(strategy))" begin
                model = test_init_model()
                empty_vi = VarInfo()
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

        function test_link_status_respected(strategy::AbstractInitStrategy)
            @testset "check that varinfo linking is preserved: $(typeof(strategy))" begin
                @model logn() = a ~ LogNormal()
                model = logn()
                vi = VarInfo(model)
                linked_vi = DynamicPPL.link!!(vi, model)
                _, new_vi = DynamicPPL.init!!(model, linked_vi, strategy, LinkAll())
                @test DynamicPPL.is_transformed(new_vi)
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

        @testset "InitFromPrior" begin
            test_generating_new_values(InitFromPrior())
            test_replacing_values(InitFromPrior())
            test_rng_respected(InitFromPrior())
            test_link_status_respected(InitFromPrior())

            @testset "check that values are within support" begin
                # Not many other sensible checks we can do for priors.
                @model just_unif() = x ~ Uniform(0.0, 1e-7)
                for _ in 1:100
                    _, vi = DynamicPPL.init!!(just_unif(), VarInfo(), InitFromPrior())
                    @test vi[@varname(x)] isa Real
                    @test 0.0 <= vi[@varname(x)] <= 1e-7
                end
            end
        end

        @testset "InitFromUniform" begin
            test_generating_new_values(InitFromUniform())
            test_replacing_values(InitFromUniform())
            test_rng_respected(InitFromUniform())
            test_link_status_respected(InitFromUniform())

            @testset "check that bounds are respected" begin
                @testset "unconstrained" begin
                    umin, umax = -1.0, 1.0
                    @model just_norm() = x ~ Normal()
                    for _ in 1:100
                        _, vi = DynamicPPL.init!!(
                            just_norm(), VarInfo(), InitFromUniform(umin, umax)
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
                            just_beta(), VarInfo(), InitFromUniform(umin, umax)
                        )
                        @test vi[@varname(x)] isa Real
                        @test tmin <= vi[@varname(x)] <= tmax
                    end
                end
            end
        end

        @testset "InitFromParams" begin
            test_link_status_respected(InitFromParams((; a=1.0)))
            test_link_status_respected(InitFromParams(Dict(@varname(a) => 1.0)))

            @testset "given full set of parameters" begin
                # test_init_model has x ~ Normal() and y ~ MvNormal(zeros(2), I)
                my_x, my_y = 1.0, [2.0, 3.0]
                params_nt = (; x=my_x, y=my_y)
                params_dict = Dict(@varname(x) => my_x, @varname(y) => my_y)
                model = test_init_model()
                empty_vi = VarInfo()
                _, vi = DynamicPPL.init!!(
                    model, deepcopy(empty_vi), InitFromParams(params_nt)
                )
                @test vi[@varname(x)] == my_x
                @test vi[@varname(y)] == my_y
                logp_nt = getlogp(vi)
                _, vi = DynamicPPL.init!!(
                    model, deepcopy(empty_vi), InitFromParams(params_dict)
                )
                @test vi[@varname(x)] == my_x
                @test vi[@varname(y)] == my_y
                logp_dict = getlogp(vi)
                @test logp_nt == logp_dict
            end

            @testset "given only partial parameters" begin
                my_x = 1.0
                params_nt = (; x=my_x)
                params_dict = Dict(@varname(x) => my_x)
                model = test_init_model()
                empty_vi = VarInfo()
                @testset "with InitFromPrior fallback" begin
                    _, vi = DynamicPPL.init!!(
                        Xoshiro(468),
                        model,
                        deepcopy(empty_vi),
                        InitFromParams(params_nt, InitFromPrior()),
                    )
                    @test vi[@varname(x)] == my_x
                    nt_y = vi[@varname(y)]
                    @test nt_y isa AbstractVector{<:Real}
                    @test length(nt_y) == 2
                    _, vi = DynamicPPL.init!!(
                        Xoshiro(469),
                        model,
                        deepcopy(empty_vi),
                        InitFromParams(params_dict, InitFromPrior()),
                    )
                    @test vi[@varname(x)] == my_x
                    dict_y = vi[@varname(y)]
                    @test dict_y isa AbstractVector{<:Real}
                    @test length(dict_y) == 2
                    # the values should be different since we used different seeds
                    @test dict_y != nt_y
                end

                @testset "with no fallback" begin
                    # These just don't have an entry for `y`.
                    @test_throws ErrorException DynamicPPL.init!!(
                        model, deepcopy(empty_vi), InitFromParams(params_nt, nothing)
                    )
                    @test_throws ErrorException DynamicPPL.init!!(
                        model, deepcopy(empty_vi), InitFromParams(params_dict, nothing)
                    )
                    # We also explicitly test the case where `y = missing`.
                    params_nt_missing = (; x=my_x, y=missing)
                    params_dict_missing = Dict(@varname(x) => my_x, @varname(y) => missing)
                    @test_throws ErrorException DynamicPPL.init!!(
                        model,
                        deepcopy(empty_vi),
                        InitFromParams(params_nt_missing, nothing),
                    )
                    @test_throws ErrorException DynamicPPL.init!!(
                        model,
                        deepcopy(empty_vi),
                        InitFromParams(params_dict_missing, nothing),
                    )
                end
            end
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
