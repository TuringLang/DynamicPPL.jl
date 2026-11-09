module DPPLSubmodelTests

using DynamicPPL
using DynamicPPL: @vnt
using AbstractPPL: @of, of
using Distributions
using Test
using Random: Xoshiro

# Dummy object that we can use to test VarNames with property lenses.
mutable struct P
    a::Float64
    b::Float64
end

function get_logp_and_rawval_accs(model::Model)
    accs = VarInfo()
    accs = setacc!!(accs, RawValueAccumulator(false))
    _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
    return accs
end

# Models for the nested-submodel type-stability tests; see
# https://github.com/TuringLang/DynamicPPL.jl/pull/1427. Each level wraps the previous one in
# a `to_submodel`. They must be defined at module scope: a model defined in local (testset)
# scope is not type-inferrable, which would mask the property under test.
@model t2844_inner() = (x ~ Normal(); return (; x))
@model t2844_middle() = (a ~ to_submodel(t2844_inner()); return (; x=a.x))
@model t2844_outer() = (b ~ to_submodel(t2844_middle()); return (; x=b.x))
@model t2844_deeper() = (c ~ to_submodel(t2844_outer()); return (; x=c.x))

@testset "submodels.jl" begin
    @testset "$op with AbstractPPL API" for op in [condition, fix]
        x_val = 1.0
        x_logp = op == condition ? logpdf(Normal(), x_val) : 0.0

        @testset "Auto prefix" begin
            @model function inner()
                x ~ Normal()
                y ~ Normal()
                return (x, y)
            end
            @model function outer()
                return a ~ to_submodel(inner())
            end
            inner_op = op(inner(), (@varname(x) => x_val))
            @model function outer2()
                return a ~ to_submodel(inner_op)
            end
            with_inner_op = outer2()
            with_outer_op = op(outer(), (@varname(a.x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(a.x), @varname(a.y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                accs = get_logp_and_rawval_accs(model)
                raw_vals = get_raw_values(accs)
                @test getlogjoint(accs) ==
                    x_logp + logpdf(Normal(), raw_vals[@varname(a.y)])
                # Check the keys
                @test Set(keys(raw_vals)) == Set([@varname(a.y)])
            end
        end

        @testset "Explicit true" begin
            @model inner() = (x ~ Normal(); y ~ Normal(); (x, y))
            @model function outer()
                return a ~ to_submodel(inner(), true)
            end
            @model function outer2()
                return a ~ to_submodel(inner_op, true)
            end
            with_inner_op = outer2()
            inner_op = op(inner(), (@varname(x) => x_val))
            with_outer_op = op(outer(), (@varname(a.x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(a.x), @varname(a.y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                accs = get_logp_and_rawval_accs(model)
                raw_vals = get_raw_values(accs)
                @test getlogjoint(accs) ==
                    x_logp + logpdf(Normal(), raw_vals[@varname(a.y)])
                # Check the keys
                @test Set(keys(raw_vals)) == Set([@varname(a.y)])
            end
        end

        @testset "Manual prefix" begin
            @model inner() = (x ~ Normal(); y ~ Normal(); (x, y))
            @model function outer()
                return a ~ to_submodel(prefix(inner(), :b))
            end
            inner_op = op(inner(), (@varname(x) => x_val))
            @model function outer2()
                return a ~ to_submodel(prefix(inner_op, :b))
            end
            with_inner_op = outer2()
            with_outer_op = op(outer(), (@varname(a.b.x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(a.b.x), @varname(a.b.y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                accs = get_logp_and_rawval_accs(model)
                raw_vals = get_raw_values(accs)
                @test getlogjoint(accs) ==
                    x_logp + logpdf(Normal(), raw_vals[@varname(a.b.y)])
                # Check the keys
                @test Set(keys(raw_vals)) == Set([@varname(a.b.y)])
            end
        end

        @testset "Complex prefixes" begin
            @model function f()
                x = Vector{Float64}(undef, 1)
                x[1] ~ Normal()
                y ~ Normal()
                return x[1]
            end
            @model function g()
                p = P(1.0, 2.0)
                p.a ~ to_submodel(f())
                p.b ~ Normal()
                return (p.a, p.b)
            end
            expected_vns = Set([@varname(p.a.x[1]), @varname(p.a.y), @varname(p.b)])
            @test Set(keys(rand(g()))) == expected_vns

            # Check that we can condition/fix on any of them from the outside
            for vn in expected_vns
                op_g = if vn == @varname(p.a.x[1])
                    op(g(), vn => 1.0, @of(p = @of(a = @of(x = of(Array, 1)))))
                else
                    op(g(), vn => 1.0)
                end
                vnt = rand(op_g)
                @test Set(keys(vnt)) == symdiff(expected_vns, Set([vn]))
            end
        end

        @testset "Nested submodels" begin
            @model function f()
                x ~ Normal()
                return y ~ Normal()
            end
            @model function g()
                return b ~ to_submodel(f())
            end
            @model function h()
                return a ~ to_submodel(g())
            end

            # No conditioning
            accs = get_logp_and_rawval_accs(h())
            raw_vals = get_raw_values(accs)
            @test Set(keys(raw_vals)) == Set([@varname(a.b.x), @varname(a.b.y)])
            @test getlogjoint(accs) ==
                logpdf(Normal(), raw_vals[@varname(a.b.x)]) +
                  logpdf(Normal(), raw_vals[@varname(a.b.y)])

            # Conditioning/fixing at the top level
            op_h = op(h(), (@varname(a.b.x) => x_val))

            # Conditioning/fixing at the second level
            op_g = op(g(), (@varname(b.x) => x_val))
            @model function h2()
                return a ~ to_submodel(op_g)
            end

            # Conditioning/fixing at the very bottom
            op_f = op(f(), (@varname(x) => x_val))
            @model function g2()
                return b ~ to_submodel(op_f)
            end
            @model function h3()
                return a ~ to_submodel(g2())
            end

            models = [("top", op_h), ("middle", h2()), ("bottom", h3())]
            @testset "$name" for (name, model) in models
                accs = get_logp_and_rawval_accs(model)
                raw_vals = get_raw_values(accs)
                @test Set(keys(raw_vals)) == Set([@varname(a.b.y)])
                @test getlogjoint(accs) ==
                    x_logp + logpdf(Normal(), raw_vals[@varname(a.b.y)])
            end
        end
    end

    @testset "conditioning argument-backed submodel LHS variables" begin
        @model function f(x)
            x ~ Normal()
            return y ~ Normal()
        end
        @model function g(inner_x)
            return a ~ to_submodel(condition(f(missing); x=inner_x))
        end

        late_bound = condition(g(0.0), @varname(a.x) => 1.0)
        @test_throws ArgumentError late_bound(Xoshiro(1))
        vnt = rand(
            Xoshiro(1), condition(decondition(g(0.0), @varname(a.x)), @varname(a.x) => 1.0)
        )
        @test Set(keys(vnt)) == Set([@varname(a.y)])

        @model latent_g() = a ~ to_submodel(f(missing))
        vnt = rand(Xoshiro(1), latent_g())
        @test Set(keys(vnt)) == Set([@varname(a.x), @varname(a.y)])
    end

    @testset ":= in submodels" begin
        @testset "basic" begin
            @model function inner1()
                a ~ Normal()
                b := a + 1.0
                return a
            end
            @model function outer1()
                x ~ to_submodel(inner1())
                return x
            end

            model = outer1()
            vnt = rand(model)
            @test only(keys(vnt)) == @varname(x.a)

            accs = VarInfo((RawValueAccumulator(true),))
            a, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
            vnt = get_raw_values(accs)
            @test vnt[@varname(x.a)] == a
            @test vnt[@varname(x.b)] == vnt[@varname(x.a)] + 1.0
        end

        @testset "with sub-VarNames" begin
            # This test set also checks that templating is happening correctly for := calls
            # inside submodels. See https://github.com/TuringLang/DynamicPPL.jl/issues/1215.
            @model function inner2()
                a ~ Normal()
                b = zeros(1)
                b[1] := a + 1.0
                return a
            end
            @model function outer2()
                x ~ to_submodel(inner2())
                return x
            end

            model = outer2()
            vnt = rand(model)
            @test only(keys(vnt)) == @varname(x.a)

            accs = VarInfo((RawValueAccumulator(true),))
            a, accs = init!!(model, accs, InitFromPrior(), UnlinkAll())
            vnt = get_raw_values(accs)
            @test vnt[@varname(x.a)] == a
            @test vnt[@varname(x.b[1])] == vnt[@varname(x.a)] + 1.0
            # If the templating fails, then x.b will be stored as a GrowableArray, and
            # trying to access the entire array will fail.
            @test vnt[@varname(x.b)] isa Vector{Float64}
            @test vnt[@varname(x.b)] == [a + 1.0]
            # For good measure.
            @test vnt[@varname(x.b[:])] == [a + 1.0]
        end
    end

    @testset "deconditioning a submodel from outside" begin
        @testset "$op" for (op, deop) in [(condition, decondition), (fix, unfix)]
            @model inner() = x ~ Normal()
            @model function outer()
                return a ~ to_submodel(inner())
            end

            model = outer()
            @test only(keys(VarInfo(model))) == @varname(a.x)
            op_model = op(model, (@varname(a.x) => 1.0))
            @test isempty(keys(VarInfo(op_model)))

            deop_model = deop(op_model)
            @test only(keys(VarInfo(deop_model))) == @varname(a.x)
            deop_model2 = deop(op_model, @varname(a))
            @test only(keys(VarInfo(deop_model2))) == @varname(a.x)
            deop_model3 = deop(op_model, @varname(a.x))
            @test only(keys(VarInfo(deop_model3))) == @varname(a.x)
        end
    end

    @testset "submodels with indexed prefixes" begin
        # These submodels briefly failed when VNT was implemented, due to GrowableArray
        # issues (see example in https://github.com/TuringLang/DynamicPPL.jl/issues/1221).
        # They're included here to prevent regressions.
        #
        @model function inner()
            return a ~ Normal()
        end
        @model function outer()
            x = zeros(4)
            for i in eachindex(x)
                x[i] ~ to_submodel(inner())
            end
        end
        model = outer()
        vnt = rand(model)
        @test Set(keys(vnt)) == Set([@varname(x[i].a) for i in 1:4])
        for i in 1:4
            @test vnt[@varname(x[i])] isa VarNamedTuple
            @test vnt[@varname(x[i].a)] isa Float64
        end
    end

    @testset "(nested) submodels with arrays inside" begin
        # This mostly tests that templates work correctly and are propagated upwards
        # correctly.
        @model function inner()
            x = zeros(2, 2)
            x[1] ~ Normal()
            return x
        end
        @model function middle()
            return b ~ to_submodel(inner())
        end
        @model function outer()
            return a ~ to_submodel(middle())
        end

        model = middle()
        vnt = rand(model)
        @test Set(keys(vnt)) == Set([@varname(b.x[1, 1])])
        @test vnt.data.b.data.x.data isa Matrix{Float64}
        @test size(vnt.data.b.data.x.data) == (2, 2)

        model = outer()
        vnt = rand(model)
        @test Set(keys(vnt)) == Set([@varname(a.b.x[1, 1])])
        @test vnt.data.a.data.b.data.x.data isa Matrix{Float64}
        @test size(vnt.data.a.data.b.data.x.data) == (2, 2)
    end

    @testset "type stability of nested submodels (issue #2844)" begin
        # See https://github.com/TuringLang/DynamicPPL.jl/pull/1427.
        @testset "$(nameof(model.f))" for model in (
            t2844_inner(), t2844_middle(), t2844_outer(), t2844_deeper()
        )
            # Check inference for `init!!` into a `VarInfo`, under both
            # transform strategies.
            @testset "$tfm" for tfm in (UnlinkAll(), LinkAll())
                accs = setacc!!(VarInfo(), LogPriorAccumulator())
                @test @inferred(init!!(model, accs, InitFromPrior(), tfm)) isa Tuple
            end
            # Evaluating a pre-populated `VarInfo` must also stay type stable.
            vi = VarInfo(model)
            @test @inferred(
                evaluate!!(
                    model,
                    InitContext(InitFromParams(get_values(vi), nothing), UnlinkAll()),
                    vi,
                )
            ) isa Tuple
        end
    end
end

@model contract_leaf(x=missing, mu=0.0) = (x ~ Normal(mu); x)
@model contract_parent(child) = a ~ to_submodel(child)
@model function runtime_middle(bind, entered)
    entered[] += 1
    b ~ to_submodel(bind(contract_leaf(); x=2.0))
    c ~ to_submodel(bind(contract_leaf(); x=3.0))
    return (; b, c)
end
@model runtime_parent(bind, entered) = a ~ to_submodel(runtime_middle(bind, entered))
@model dormant_parent(run) = (run && (a ~ to_submodel(contract_leaf())); q ~ Normal())
@model child_array(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
@model child_fields(x) = (x.a ~ Normal(); x)
@model function indexed_observations(y)
    mu ~ Normal()
    x = similar(y)
    for i in eachindex(y)
        x[i] ~ to_submodel(condition(contract_leaf(missing, mu); x=y[i]))
    end
    return x
end

@testset "child binding contract" begin
    @testset "evaluation inference after partial removal" begin
        cleared = decondition(indexed_observations(zeros(2)), @varname(x[1].x))
        strategy = InitFromParams(
            VarNamedTuple((@varname(mu) => 0.1, @varname(x[1].x) => 0.0)), nothing
        )
        accs = VarInfo(LogPriorAccumulator(), LogLikelihoodAccumulator())
        for model in
            (cleared, condition(cleared, @varname(x[1].x) => 0.0, @of(x = of(Array, 2))))
            @test @inferred(init!!(Xoshiro(1), model, accs, strategy, UnlinkAll())) isa
                Tuple
        end
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            model = bind(
                contract_parent(decondition(child_array(zeros(2)))),
                @varname(a.x[1]) => 3.0,
                @of(a = @of(x = of(Array, 2))),
            )
            updated = remove(bind(model, @varname(a.x[2]) => 4.0), @varname(a.x[1]))
            @test @inferred(updated(Xoshiro(1))) isa Vector{Float64}
        end
        parent = condition(
            contract_parent(decondition(child_fields((a=0.0, b=0.0)))),
            VarNamedTuple((a=VarNamedTuple((x=(a=1.0, b=2.0),)),)),
        )
        parent = fix(decondition(parent, @varname(a.x.a)), @varname(a.x.a) => 3.0)
        @test first(
            @inferred init!!(
                Xoshiro(1),
                parent,
                VarInfo(),
                InitFromParams(VarNamedTuple(), nothing),
                UnlinkAll(),
            )
        ) == (a=3.0, b=2.0)
    end
    @test_throws ArgumentError to_submodel(contract_leaf(), false)
    @test_throws ArgumentError contract_parent(contract_leaf(1.0))(Xoshiro(1))
    @test loglikelihood(
        contract_parent(condition(contract_leaf(missing, 0.5); x=1.0)), (;)
    ) == logpdf(Normal(0.5), 1.0)
    @test loglikelihood(contract_leaf(1.0, 0.5), (;)) == logpdf(Normal(0.5), 1.0)
    @test returned(contract_parent(contract_leaf(missing, 0.5)), (; a=(x=1.0,))) == 1.0
    @test Set(
        keys(rand(Xoshiro(1), contract_parent(prefix(contract_leaf(), @varname(p)))))
    ) == Set([@varname(a.p.x)])
    for (bind, remove, listing) in
        ((condition, decondition, conditioned), (fix, unfix, fixed))
        prefixed = prefix(contract_leaf(), @varname(p))
        @test_throws ArgumentError bind(prefixed; x=1.0)
        @test bind(prefixed, @varname(p.x) => 1.0)(Xoshiro(1)) == 1.0
        @test isempty(listing(remove(bind(prefixed, @varname(p.x) => 1.0), @varname(p.x))))
        @test_throws ArgumentError bind(contract_parent(contract_leaf()); a=1.0)(Xoshiro(1))
        dormant = bind(dormant_parent(false), @varname(a.x) => 1.0)
        @test dormant(Xoshiro(1)) isa Real
        @test listing(dormant)[@varname(a.x)] == 1.0
        @test isempty(listing(dormant_parent(false)))
        for names in ((), (@varname(a),), (@varname(a.b.x), @varname(a.c.x)))
            entered = Ref(0)
            m = runtime_parent(bind, entered)
            @test isempty(listing(m))
            @test entered[] == 0
            cleared = remove(m, names...)
            params = (; a=(b=(x=0.5,), c=(x=0.25,)))
            @test returned(cleared, params) == (b=0.5, c=0.25)
            @test logprior(cleared, params) ≈ logpdf(Normal(), 0.5) + logpdf(Normal(), 0.25)
            @test loglikelihood(cleared, params) == 0.0
            @test Set(keys(rand(Xoshiro(1), cleared))) ==
                Set([@varname(a.b.x), @varname(a.c.x)])
            rebound = bind(cleared, @varname(a.b.x) => 4.0)
            @test rebound(Xoshiro(1)).b == 4.0
            @test remove(rebound, @varname(a.b.x))(Xoshiro(1)).b != 2.0
        end
        m = runtime_parent(bind, Ref(0))
        partial = remove(m, @varname(a.b.x))
        @test returned(partial, (; a=(b=(x=0.5,),))) == (b=0.5, c=3.0)
        absent = remove(m, @varname(a.unknown))
        @test absent(Xoshiro(1)) == (b=2.0, c=3.0)
        @test remove(m, @varname(a.b.x), @varname(a.c.x)) ==
            remove(m, @varname(a.c.x), @varname(a.b.x))
        @test remove(m) != m
        @test listing(remove(m)) == listing(m)
        entered = Ref(false)
        @model entered_child(entered) = (entered[] = true; x ~ Normal(); x)
        child = bind(entered_child(entered); x=1.0)
        conflict = bind(contract_parent(child), @varname(a.x) => 2.0)
        @test_throws ArgumentError conflict(Xoshiro(1))
        @test !entered[]
        @test bind(remove(contract_parent(child), @varname(a.x)), @varname(a.x) => 2.0)(
            Xoshiro(1)
        ) == 2.0
        @test entered[]
    end
end

@model fields_child() = (x ~ Normal(); y ~ Normal(); (; x, y))
@model fields_middle() = b ~ to_submodel(fields_child())
@model fields_parent() = a ~ to_submodel(fields_middle())
@model indexed_child(i) = (a = zeros(2, 2); a[i] ~ to_submodel(contract_leaf()); a)
@model argument_child(a) = a[1] ~ to_submodel(contract_leaf())
@model whole_fields() = b ~ product_distribution((x=Normal(), y=Normal()))
@model runtime_fields(namespace) =
    a ~ to_submodel(namespace ? fields_middle() : whole_fields())
@model function storage_leaf(n=2)
    z = zeros(n)
    for i in eachindex(z)
        z[i] ~ Normal()
    end
    x ~ Normal()
    return (; z, x)
end
@model storage_middle(n=2) = b ~ to_submodel(storage_leaf(n))
@model storage_parent(n=2) = a ~ to_submodel(storage_middle(n))
@model function indexed_storage()
    a = Vector{Any}(undef, 1)
    a[1] ~ to_submodel(storage_middle())
    return a
end
@testset "child namespace geometry and structural storage" begin
    @testset "storage survives role changes and listings" begin
        for (bind, remove, other) in
            ((condition, decondition, fix), (fix, unfix, condition))
            stored = @vnt begin
                @template a = zeros(1)
                a[1] := VarNamedTuple((b=VarNamedTuple((z=[1.0, 2.0],)),))
            end
            model = other(
                remove(bind(indexed_storage(), stored), @varname(a[1].b.z[2])),
                @varname(a[1].b.z[2]) => 5.0,
            )
            @test fix(condition(indexed_storage(), conditioned(model)), fixed(model))(
                Xoshiro(1)
            ) == model(Xoshiro(1))
            model = bind(
                storage_parent(3),
                VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((z=ones(2),)),)),)),
            )
            masked = remove(remove(model, @varname(a.b.z[1])), @varname(a.b.z[2]))
            @test_throws ArgumentError bind(masked; a=(b=(z=ones(3),),))
        end
        parent = contract_parent(decondition(child_array(zeros(2))))
        model = condition(
            parent, @varname(a.x[falses(3)]) => Float64[], @of(a = @of(x = of(Array, 3)))
        )
        @test_throws ArgumentError model(Xoshiro(1))
        @test_throws ArgumentError condition(parent, conditioned(model))(Xoshiro(1))
        child = condition(storage_leaf(), @varname(z[1]) => 1.0, @of(z = of(Array, 2)))
        ancestor = condition(
            contract_parent(child),
            @varname(a.z[2]) => 2,
            @of(a = @of(z = of(Array, Int, 2))),
        )
        @test_throws ArgumentError ancestor(Xoshiro(1))
    end
    @testset "whole values and namespace boundaries" begin
        for bind in (condition, fix), namespace in (false, true)
            model = runtime_fields(namespace)
            field = bind(model, @varname(a.b.y) => 2.0)
            @test_throws ArgumentError bind(field, @varname(a.b) => (x=1.0,))
            shape = bind(
                model, @varname(a.b) => (x=1.0, y=2.0), of((a=(b=(x=0.0, y=0.0),),))
            )
            @test_throws ArgumentError bind(shape, @varname(a.b.y) => 3.0)
        end
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            roots = VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=0.0, y=0.0)),)),))
            cleared = remove(bind(fields_parent(), roots))
            @test bind(bind(cleared, @varname(a.b) => (x=1.0,)), @varname(a.b) => (y=2.0,))(
                Xoshiro(1)
            ) == (x=1.0, y=2.0)
        end
    end
    for (bind, remove, listing) in
        ((condition, decondition, conditioned), (fix, unfix, fixed))
        ambiguous = bind(fields_parent(); a=(b=(x=1.0, y=2.0),))
        @test_throws ArgumentError remove(ambiguous, @varname(a.b.x))
        @test_throws ArgumentError bind(ambiguous, @varname(a.b) => (x=1.0,))
        structured = VarNamedTuple((a=VarNamedTuple((b=(x=1.0, y=2.0),)),))
        bound = bind(fields_parent(), structured)
        @test remove(bound, @varname(a.b.x))(Xoshiro(1)).y == 2.0
        @test bind(remove(bound, @varname(a.b)), @varname(a.b.x) => 3.0)(Xoshiro(1)).x ==
            3.0
        model = indexed_child(CartesianIndex(1, 2))
        @test Set(keys(rand(Xoshiro(1), model))) == Set([@varname(a[1, 2].x)])
        schema = @of(a = of(Array, 2, 2))
        @test_throws ArgumentError bind(model, @varname(a[1, 2].x) => 1.0)
        @test bind(model, @varname(a[1, 2].x) => 1.0, schema)(Xoshiro(1))[1, 2] == 1.0
        @test_throws ArgumentError bind(model, @varname(a[3, 1].x) => 1.0, schema)
        for index in (true, 1:2, [1, 2], [true, false])
            @test_throws ArgumentError indexed_child(index)(Xoshiro(1))
        end
        @test_throws ArgumentError argument_child(zeros(2))(Xoshiro(1))
    end
end

@testset "child entry validates full addresses" begin
    @model entry_child(entered) = (entered[] = true; x ~ Normal(); x)
    @model entry_parent(child) = a ~ to_submodel(child)
    for bind in (condition, fix), prefixed in (false, true)
        entered = Ref(false)
        child = entry_child(entered)
        prefixed && (child = prefix(child, @varname(b)))
        parent = entry_parent(child)
        names = if prefixed
            (@varname(a.x), @varname(a.typo), @varname(a.b.typo))
        else
            (@varname(a.typo),)
        end
        for vn in names
            err = try
                bind(parent, vn => 1.0)(Xoshiro(1))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("`$vn`", sprint(showerror, err))
            @test !entered[]
        end
        vn = prefixed ? @varname(a.b.x) : @varname(a.x)
        @test bind(parent, vn => 1.0)(Xoshiro(1)) == 1.0
    end
end

@testset "recursive removal normalizes submodel indices" begin
    @model removal_leaf() = (x ~ Normal(); x)
    @model function removal_grid(child, linear)
        z = zeros(2, 2)
        for i in (linear ? eachindex(z) : CartesianIndices(z))
            z[i] ~ to_submodel(child)
        end
        return z
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix)), linear in (false, true)
        m = removal_grid(bind(removal_leaf(); x=1.0), linear)
        for vn in
            (@varname(z[3].x), @varname(z[1, 2].x), @varname(z[CartesianIndex(1, 2)].x))
            draws = rand(Xoshiro(1), remove(m, vn))
            @test length(draws) == 1
            @test haskey(draws, linear ? @varname(z[3].x) : @varname(z[1, 2].x))
        end
    end
end

end
