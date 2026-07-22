module DynamicPPLChainsTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using DynamicPPL
using Distributions
using LinearAlgebra
using Test

@testset "ParamsWithStats from VarInfo" begin
    @model function f(z)
        x ~ Normal()
        y := x + 1
        return z ~ Normal(y)
    end
    z = 1.0
    model = f(z)

    for init_strat in (InitFromPrior(), InitFromParams(VarInfo(model).values))
        @testset "with reevaluation" begin
            ps = ParamsWithStats(init_strat, model)
            @test haskey(ps.params, @varname(x))
            @test haskey(ps.params, @varname(y))
            @test length(ps.params) == 2
            @test haskey(ps.stats, :logprior)
            @test haskey(ps.stats, :loglikelihood)
            @test haskey(ps.stats, :logjoint)
            @test length(ps.stats) == 3
            @test ps.stats.logjoint ≈ ps.stats.logprior + ps.stats.loglikelihood
            @test ps.params[@varname(y)] ≈ ps.params[@varname(x)] + 1
            @test ps.stats.logprior ≈ logpdf(Normal(), ps.params[@varname(x)])
            @test ps.stats.loglikelihood ≈ logpdf(Normal(ps.params[@varname(y)]), z)
        end

        @testset "without colon_eq" begin
            ps = ParamsWithStats(init_strat, model; include_colon_eq=false)
            @test haskey(ps.params, @varname(x))
            @test length(ps.params) == 1
            @test haskey(ps.stats, :logprior)
            @test haskey(ps.stats, :loglikelihood)
            @test haskey(ps.stats, :logjoint)
            @test length(ps.stats) == 3
            @test ps.stats.logjoint ≈ ps.stats.logprior + ps.stats.loglikelihood
            @test ps.stats.logprior ≈ logpdf(Normal(), ps.params[@varname(x)])
            @test ps.stats.loglikelihood ≈ logpdf(Normal(ps.params[@varname(x)] + 1), z)
        end

        @testset "without log probs" begin
            ps = ParamsWithStats(init_strat, model; include_log_probs=false)
            @test haskey(ps.params, @varname(x))
            @test haskey(ps.params, @varname(y))
            @test length(ps.params) == 2
            @test isempty(ps.stats)
        end
    end

    @testset "no reevaluation" begin
        # Without VAIM, it should error
        vi = OnlyAccsVarInfo()
        @test_throws ErrorException get_raw_values(vi) # sanity check that it doesn't have VAIM
        vi = last(DynamicPPL.init!!(model, vi, InitFromPrior(), UnlinkAll()))
        @test_throws ErrorException ParamsWithStats(vi)
        # With VAIM, it should work
        vi = OnlyAccsVarInfo(RawValueAccumulator(true))
        vi = last(DynamicPPL.init!!(model, vi, InitFromPrior(), UnlinkAll()))
        ps = ParamsWithStats(vi)
        @test haskey(ps.params, @varname(x))
        @test haskey(ps.params, @varname(y))
        @test length(ps.params) == 2
        # Because we didn't evaluate with log prob accumulators, there should be no stats
        @test isempty(ps.stats)
    end
end

@testset "ParamsWithStats from LogDensityFunction" begin
    @testset "$(m.f)" for m in DynamicPPL.TestUtils.ALL_MODELS
        @testset "$transform_strategy" for transform_strategy in (UnlinkAll(), LinkAll())
            # Get the ParamsWithStats using LogDensityFunction
            ldf = LogDensityFunction(m, getlogjoint, transform_strategy)
            param_vector = rand(ldf)
            # This will give us a VNT of values.params`.
            actual_vnt = ParamsWithStats(param_vector, ldf).params
            # We should make sure that those values line up with the values inside the vector.
            accs = OnlyAccsVarInfo(RawValueAccumulator(true))
            _, accs = DynamicPPL.init!!(
                m, accs, InitFromVector(param_vector, ldf), transform_strategy
            )
            expected_vnt = DynamicPPL.densify!!(get_raw_values(accs))

            # Iterate over all variables to check that their values match
            @test Set(keys(actual_vnt)) == Set(keys(expected_vnt))
            for vn in keys(actual_vnt)
                @test actual_vnt[vn] == expected_vnt[vn]
            end
        end
    end

    @testset "with fixed transforms" begin
        # Note: can't use ALL_MODELS here because that contains a model with dynamic transforms,
        # which would yield incorrect results with fix_transforms.
        @testset "$(m.f)" for m in DynamicPPL.TestUtils.DEMO_MODELS
            @testset "$transform_strategy" for transform_strategy in
                                               (UnlinkAll(), LinkAll())
                ldf_fixed = LogDensityFunction(
                    m, getlogjoint_internal, transform_strategy; fix_transforms=true
                )
                ldf_dynamic = LogDensityFunction(
                    m, getlogjoint_internal, transform_strategy
                )
                param_vector = rand(ldf_fixed)

                # Fast path (no log probs, no colon eq): should match the model-evaluation path
                fast = ParamsWithStats(
                    param_vector, ldf_fixed; include_log_probs=false, include_colon_eq=false
                )
                slow = ParamsWithStats(
                    param_vector,
                    ldf_dynamic;
                    include_log_probs=false,
                    include_colon_eq=false,
                )
                @test fast == slow
            end
        end

        @testset "check that model is actually not evaluated" begin
            should_error = false
            @model function prickly()
                x ~ Normal()
                return should_error && error("nope")
            end
            # need to construct LDF without erroring
            ldf = LogDensityFunction(
                prickly(), getlogjoint_internal, LinkAll(); fix_transforms=true
            )
            # now make the model error
            should_error = true
            @test_throws ErrorException prickly()()
            # check that ParamsWithStats doesn't error
            @test ParamsWithStats(
                [0.5], ldf; include_log_probs=false, include_colon_eq=false
            ) isa Any
            # but it does if you set either of them to true
            for (ilp, ice) in ((true, false), (false, true), (true, true))
                @test_throws ErrorException ParamsWithStats(
                    [0.5], ldf; include_log_probs=ilp, include_colon_eq=ice
                )
            end
        end
    end

    @testset "errors on invalid length" begin
        @model function f()
            x ~ Normal()
            y ~ Normal()
            return nothing
        end
        for fix_transforms in (false, true)
            ldf = LogDensityFunction(f(), getlogjoint_internal, UnlinkAll(); fix_transforms)
            for vec in (randn(1), randn(3))
                @test_throws ArgumentError ParamsWithStats(
                    vec, ldf; include_log_probs=false, include_colon_eq=false
                )
            end
        end
    end
end

@testset "no Union-typed VarInfo across init!! (Julia 1.12 -O2 GC miscompile guard)" begin
    # Regression guard for a Julia 1.12 `-O2` codegen/GC-rooting bug. Both `ParamsWithStats`
    # model-reevaluation methods choose an accumulator tuple based on `include_log_probs`. If
    # that choice is made *before* `init!!`, the resulting `(retval, varinfo)` tuple is
    # `Union`-typed; 1.12 heap-boxes it, leaves the dead `retval` pointer fields uninitialized
    # across a GC safepoint, and a GC landing there corrupts the heap (a `gc_mark_obj8`
    # segfault, only reproducible under concurrent allocation load). Routing `init!!` through
    # the `@noinline _pws_eval` barrier keeps the accumulator `VarInfo` concretely typed at
    # every call site. The union is a type-inference fact independent of Julia version (only
    # the crash is 1.12-specific), so assert it never reappears in the inferred IR.
    model = DynamicPPL.TestUtils.DEMO_MODELS[1]
    ldf = LogDensityFunction(model)
    param_vector = rand(ldf)
    # Compile both entry points (vector + strategy) so their method bodies are in the IR cache.
    ParamsWithStats(param_vector, ldf)
    ParamsWithStats(DynamicPPL.InitFromPrior(), model)

    is_union_of_oavi(t) =
        t isa Union && all(u -> u <: DynamicPPL.OnlyAccsVarInfo, Base.uniontypes(t))
    function offending_methods(f)
        hits = String[]
        for m in methods(f)
            body = try
                Base.bodyfunction(m)
            catch
                nothing
            end
            body === nothing && continue
            for (ci, _) in code_typed(body; optimize=true)
                any(is_union_of_oavi, ci.ssavaluetypes) && push!(hits, string(m))
            end
        end
        return hits
    end
    offending = [
        offending_methods(DynamicPPL.pws_with_eval)
        offending_methods(ParamsWithStats)
    ]
    @test isempty(offending)
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
