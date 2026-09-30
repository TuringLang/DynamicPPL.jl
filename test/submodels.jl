module DPPLSubmodelTests

using DynamicPPL
using Distributions
using DimensionalData: DimArray, X, Y
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems
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

@model indexed_observation(y, mu) = y ~ Normal(mu, 1)
@model function indexed_observations(y)
    mu ~ Normal()
    x = similar(y)
    for i in eachindex(y)
        x[i] ~ to_submodel(indexed_observation(y[i], mu))
    end
    return x
end
@model nested_observations(y) = a ~ to_submodel(indexed_observations(y))

@testset "submodels.jl" begin
    @testset "implicit argument observations stay in their model" begin
        @model scalar_likelihood(y, mu) = y ~ Normal(mu)
        @model function overlap(y)
            mu ~ Normal()
            y[1] ~ Normal(mu)
            return a ~ to_submodel(scalar_likelihood(y[2], mu), false)
        end
        @model function renamed(z)
            mu ~ Normal()
            z[1] ~ Normal(mu)
            return a ~ to_submodel(scalar_likelihood(z[2], mu), false)
        end
        expected = sum(logpdf.(Normal(), [0.0, 1.0, 2.0]))
        @test logjoint(overlap([1.0, 2.0]), (; mu=0.0)) ≈ expected
        @test logjoint(renamed([1.0, 2.0]), (; mu=0.0)) ≈ expected
        for bind in (condition, fix)
            model = bind(renamed([1.0, 2.0]), @varname(y) => 5.0)
            expected = logpdf(Normal(), 1.0)
            bind === condition && (expected += logpdf(Normal(), 5.0))
            @test loglikelihood(model, (; mu=0.0)) ≈ expected
        end
    end

    @testset "arguments supply submodel return buffers" begin
        @model child() = (x ~ Normal(); x)
        @model function dynamic_buffer(a)
            sub = to_submodel(fix(child(); x=2.0))
            a ~ sub
            return a
        end
        @model function indexed_buffer(a)
            a[1] ~ to_submodel(fix(child(); x=2.0))
            return a
        end
        @model keyword_buffer(; a=0.0) = a ~ to_submodel(fix(child(); x=2.0))
        @model nested(m) = b ~ to_submodel(m)
        for bind in (condition, fix)
            for model in (
                bind(dynamic_buffer((; x=0.0)), @varname(a.x) => 2.0),
                bind(dynamic_buffer(0.0); a=(; x=2.0)),
                bind(indexed_buffer([(; x=0.0)]), @varname(a[1].x) => 2.0),
            )
                @test_throws r"ArgumentError: .*return-value buffer" model(Xoshiro(1))
            end
        end
        for (model, expected) in (
            (dynamic_buffer(0.0), 2.0),
            (indexed_buffer([0.0]), [2.0]),
            (keyword_buffer(), 2.0),
        )
            for wrapped in (model, nested(model))
                @test wrapped(Xoshiro(1)) == expected
                @test isempty(keys(VarInfo(Xoshiro(1), wrapped)))
            end
            @test decondition(model, :a)(Xoshiro(1)) == expected
            for bind in (condition, fix)
                address = expected isa AbstractArray ? @varname(a[1]) : @varname(a)
                bound = bind(model, address => 3.0)
                @test_throws r"ArgumentError: .*submodel's return value" bound(Xoshiro(1))
            end
        end
    end

    @testset "bindings below argument return buffers are rejected" begin
        @model child(mu) = x ~ Normal(mu)
        @model function buffer_parent(a; rhs=child)
            mu = a.x
            a ~ to_submodel(rhs(mu))
            return mu, a
        end
        @model function indexed_parent(a, i)
            mu = a[i].x
            a[i] ~ to_submodel(child(mu))
            return mu, a
        end
        @model keyword_parent(; a=(; x=0.0)) = a ~ to_submodel(child(a.x))
        @model dynamic_parent(a, rhs) = a ~ rhs
        @model outer(m) = b ~ to_submodel(m)
        @model local_parent() = a ~ to_submodel(child(0.0))
        for bind in (condition, fix)
            models = (
                bind(decondition(buffer_parent((; x=0.0))), @varname(a.x) => 2.0),
                bind(decondition(buffer_parent((; x=0.0))); a=(; x=2.0)),
                bind(decondition(keyword_parent()); a=(; x=2.0)),
                bind(decondition(indexed_parent([(; x=0.0)], 1)), @varname(a[1].x) => 2.0),
                bind(decondition(indexed_parent([(; x=0.0)], 1)); a=[(; x=2.0)]),
                bind(
                    decondition(dynamic_parent((; x=0.0), to_submodel(child(0.0)))),
                    @varname(a.x) => 2.0,
                ),
            )
            for model in models, wrapped in (model, outer(model))
                @test_throws r"ArgumentError: .*return-value buffer.*Condition.*child.*to_submodel" wrapped(
                    Xoshiro(1)
                )
            end
            @test bind(local_parent(), @varname(a.x) => 2.0)(Xoshiro(1)) == 2.0
            @test bind(local_parent(); a=(; x=2.0))(Xoshiro(1)) == 2.0
            @test_throws r"ArgumentError: .*return-value buffer" bind(
                outer(decondition(buffer_parent((; x=0.0)))), @varname(b.a.x) => 2.0
            )(
                Xoshiro(1)
            )
            @test bind(dynamic_parent(0.0, Normal()); a=2.0)(Xoshiro(1)) == 2.0
            replacement = mu -> bind(child(mu); x=2.0)
            model = decondition(buffer_parent((; x=0.0); rhs=replacement))
            @test model(Xoshiro(1)) == (0.0, 2.0)
            @test logjoint(model, VarNamedTuple()) ==
                (bind === condition ? logpdf(Normal(), 2.0) : 0.0)
        end
    end

    @testset "submodel return bindings with manual prefixes" begin
        @model child() = x ~ Normal()
        @model manual(a, m) = a ~ to_submodel(m, false)
        @model nested(m) = outer ~ to_submodel(m)
        for child_model in (child(), prefix(child(), @varname(b)))
            for m in (
                condition(decondition(manual(0.0, child_model)); a=3.0),
                fix(decondition(manual(0.0, child_model)); a=3.0),
            )
                for model in (m, nested(m))
                    @test_throws "Cannot condition or fix a submodel's return value" model(
                        Xoshiro(1)
                    )
                end
            end
        end
    end

    @testset "indexed child bindings stay local" begin
        for n in (1_000, 2_000)
            model = indexed_observations(zeros(n))
            params = (; mu=0.25)
            @test logjoint(model, params) ≈
                logpdf(Normal(), params.mu) + n * logpdf(Normal(params.mu, 1), 0)
            @test (@allocated logjoint(model, params)) < 64n
        end

        model = nested_observations(zeros(2))
        model = condition(model, @varname(a.x[1].y) => 2.0)
        model = fix(model, @varname(a.x[2].y) => 3.0)
        params = mu -> VarNamedTuple(; a=VarNamedTuple(; mu))
        density = mu -> logjoint(model, params(mu))
        @test density(0.25) ≈ logpdf(Normal(), 0.25) + logpdf(Normal(0.25, 1), 2.0)
        @test ForwardDiff.derivative(density, 0.25) ≈ 1.5
        result, _ = init!!(model, VarInfo(), InitFromParams(params(0.25), nothing))
        @test result == [2.0, 3.0]
        likelihoods = pointwise_loglikelihoods(model, InitFromParams(params(0.25), nothing))
        @test keys(likelihoods) == [@varname(a.x[1].y)]
        @test size(likelihoods.data.a.data.x) == (2,)
    end

    @testset "dynamic prefixes with stored observations" begin
        @model observed_child(x=2.0) = x ~ Normal()
        @model function dynamic_parent(child)
            a = zeros(2, 3)
            a[begin] ~ to_submodel(child)
            a[end, end] ~ to_submodel(child)
            return a
        end
        @model nested_parent(child) = b ~ to_submodel(dynamic_parent(child))
        @model explicit_child(child) =
            unused ~ to_submodel(prefix(child, @varname(inner)), false)
        for op in (condition, fix), parent in (dynamic_parent, nested_parent)
            child = op(observed_child(); x=2.0)
            model = parent(child)
            @test model() == [2.0 0.0 0.0; 0.0 0.0 2.0]
            @test isempty(keys(VarInfo(model)))
            @test logjoint(model, VarNamedTuple()) ==
                (op === condition ? 2 * logpdf(Normal(), 2.0) : 0.0)
            vn = parent === dynamic_parent ? @varname(a[2, 3].x) : @varname(b.a[2, 3].x)
            if op === condition
                likelihoods = pointwise_loglikelihoods(model, InitFromPrior())
                @test likelihoods[vn] == logpdf(Normal(), 2.0)
            end
            changed = condition(model, vn => 3.0)
            @test changed() == [2.0 0.0 0.0; 0.0 0.0 3.0]

            model = parent(explicit_child(child))
            vn = if parent === dynamic_parent
                @varname(a[1].inner.x)
            else
                @varname(b.a[1].inner.x)
            end
            changed = condition(model, vn => 3.0)
            @test changed() == [3.0 0.0 0.0; 0.0 0.0 2.0]
        end
    end

    @testset "parent array templates" begin
        @model leaf_template() = x ~ Normal()
        @model function matrix_template(a)
            a[1] ~ to_submodel(leaf_template())
            a[2, 2] ~ to_submodel(leaf_template())
            return a
        end
        @model function nested_template(a)
            b ~ to_submodel(decondition(matrix_template(a)))
            return b
        end
        for container in (zeros(2, 2), zeros(Float32, 2, 2), DimArray(zeros(2, 2), (X, Y))),
            wrap in (identity, nested_template)

            model = if wrap === identity
                decondition(matrix_template(container))
            else
                wrap(container)
            end
            vi = VarInfo(RawValueAccumulator(false))
            result, vi = init!!(Xoshiro(1), model, vi, InitFromPrior(), UnlinkAll())
            raw = get_raw_values(vi)
            a = wrap === identity ? raw.data.a : raw.data.b.data.a
            @test size(a.data) == size(container)
            @test count(a.mask) == 2
            @test a.data[1].data.x == result[1]
            @test a.data[2, 2].data.x == result[2, 2]
        end

        ldf = LogDensityFunction(nested_template(zeros(2, 2)))
        parameters = [0.25, 0.5]
        logdensity = p -> LogDensityProblems.logdensity(ldf, p)
        @test logdensity(parameters) ≈ sum(logpdf.(Normal(), parameters))
        @test ForwardDiff.gradient(logdensity, parameters) ≈ -parameters

        @model function child_matrix()
            x = zeros(2, 3)
            x[1] ~ Normal()
            x[2, 3] ~ Normal()
            return sum(x)
        end
        @model function parent_matrix()
            a = zeros(2, 2)
            a[1] ~ to_submodel(child_matrix())
            a[2, 2] ~ to_submodel(child_matrix())
            return a
        end
        vi = VarInfo(RawValueAccumulator(false))
        _, vi = init!!(Xoshiro(1), parent_matrix(), vi, InitFromPrior(), UnlinkAll())
        a = get_raw_values(vi).data.a
        @test size(a.data) == (2, 2)
        @test size(a.data[1].data.x.data) == (2, 3)
        @test size(a.data[2, 2].data.x.data) == (2, 3)

        @model middle_matrix() =
            unused ~ to_submodel(prefix(child_matrix(), @varname(inner)), false)
        @model function outer_matrix()
            a = zeros(2, 2)
            a[1] ~ to_submodel(middle_matrix())
            a[2, 2] ~ to_submodel(middle_matrix())
            return a
        end
        _, vi = init!!(Xoshiro(1), outer_matrix(), vi, InitFromPrior(), UnlinkAll())
        a = get_raw_values(vi).data.a
        @test size(a.data) == (2, 2)
        @test size(a.data[1].data.inner.data.x.data) == (2, 3)
        @test size(a.data[2, 2].data.inner.data.x.data) == (2, 3)
    end

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

        @testset "No prefix" begin
            @model function inner()
                x ~ Normal()
                y ~ Normal()
                return (x, y)
            end
            @model function outer()
                return a ~ to_submodel(inner(), false)
            end
            @model function outer2()
                return a ~ to_submodel(inner_op, false)
            end
            with_inner_op = outer2()
            inner_op = op(inner(), (@varname(x) => x_val))
            with_outer_op = op(outer(), (@varname(x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(x), @varname(y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                accs = get_logp_and_rawval_accs(model)
                raw_vals = get_raw_values(accs)
                @test getlogjoint(accs) == x_logp + logpdf(Normal(), raw_vals[@varname(y)])
                # Check the keys
                @test Set(keys(raw_vals)) == Set([@varname(y)])
            end
        end

        @testset "Manual prefix" begin
            @model function inner()
                x ~ Normal()
                y ~ Normal()
                return (x, y)
            end
            @model function outer()
                return a ~ to_submodel(prefix(inner(), :b), false)
            end
            inner_op = op(inner(), (@varname(x) => x_val))
            @model function outer2()
                return a ~ to_submodel(prefix(inner_op, :b), false)
            end
            with_inner_op = outer2()
            with_outer_op = op(outer(), (@varname(b.x) => x_val))

            # No conditioning/fixing
            @test Set(keys(VarInfo(outer()))) == Set([@varname(b.x), @varname(b.y)])

            # With conditioning/fixing
            models = [("inner", with_inner_op), ("outer", with_outer_op)]
            @testset "$name" for (name, model) in models
                # Test that the value was correctly set
                @test model()[1] == x_val
                # Test that the logp was correctly set
                accs = get_logp_and_rawval_accs(model)
                raw_vals = get_raw_values(accs)
                @test getlogjoint(accs) ==
                    x_logp + logpdf(Normal(), raw_vals[@varname(b.y)])
                # Check the keys
                @test Set(keys(raw_vals)) == Set([@varname(b.y)])
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
                op_g = op(g(), (vn => 1.0))
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
                return _unused ~ to_submodel(prefix(f(), :b), false)
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
                return _unused ~ to_submodel(prefix(op_f, :b), false)
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

    @testset "conditioning argument-backed submodel sites" begin
        @model function f(x)
            x ~ Normal()
            return y ~ Normal()
        end
        @model function g(inner_x)
            return a ~ to_submodel(f(inner_x))
        end

        vnt = rand(Xoshiro(1), condition(g(0.0), @varname(a.x) => 1.0))
        @test Set(keys(vnt)) == Set([@varname(a.y)])

        @model latent_g() = a ~ to_submodel(decondition(f(0.0)))
        vnt = rand(Xoshiro(1), latent_g())
        @test Set(keys(vnt)) == Set([@varname(a.x), @varname(a.y)])

        @model observed_child(x=2.0) = x ~ Normal()
        @model function parent_with_buffer(a)
            a[1] ~ to_submodel(observed_child())
            return a
        end
        @test parent_with_buffer(zeros(1))() == [2.0]
        @test decondition(parent_with_buffer(zeros(1)))() == [2.0]
        @test isempty(keys(VarInfo(decondition(parent_with_buffer(zeros(1))))))
    end

    @testset "extending named-tuple submodel namespaces" begin
        @model namespace_inner() = (x ~ Normal(); y ~ Normal(); z ~ Normal(); (x, y, z))
        @model namespace_outer() = a ~ to_submodel(namespace_inner())
        for first_bind in (condition, fix), bind in (condition, fix)
            original = first_bind(namespace_outer(); a=(; x=1.0))
            extended = bind(original, @varname(a.y) => 2.0)
            @test bind(extended, @varname(a.z) => 3.0)(Xoshiro(1)) == (1.0, 2.0, 3.0)
            expanded = bind(original, @varname(a.x) => 1.0)
            @test bind(bind(expanded, @varname(a.y) => 2.0), @varname(a.z) => 3.0)(
                Xoshiro(1)
            ) == (1.0, 2.0, 3.0)
        end
    end

    @testset "submodel namespaces reject argument buffers" begin
        @model inner_buffer() = x ~ Normal()
        @model scalar_buffer(a) = a ~ to_submodel(inner_buffer())
        @model function indexed_buffer(a)
            one(a[1])
            return a[1] ~ to_submodel(inner_buffer())
        end
        namespace = @vnt begin
            @template a = [(; x=0.0)]
            a[1].x := 2.0
        end
        for bind in (condition, fix)
            @test_throws r"ArgumentError: .*return-value buffer" bind(
                decondition(scalar_buffer(0.0)), @varname(a.x) => 2.0
            )(
                Xoshiro(1)
            )
            @test_throws r"ArgumentError: .*return-value buffer" bind(
                decondition(indexed_buffer([0.0])), @varname(a[1].x) => 2.0
            )(
                Xoshiro(1)
            )
            @test_throws r"ArgumentError: .*return-value buffer" bind(
                decondition(indexed_buffer([0.0])), namespace
            )(
                Xoshiro(1)
            )
        end
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
            # The fast evaluation path: `init!!` into a `VarInfo`, under both
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

end
