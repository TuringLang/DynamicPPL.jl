module DPPLSubmodelTests

using DynamicPPL
using AbstractPPL: AbstractPPL
using Distributions
using DimensionalData: DimArray, X, Y
using ForwardDiff: ForwardDiff
using ADTypes: AutoForwardDiff, AutoMooncake
using Mooncake: Mooncake
using LogDensityProblems: LogDensityProblems
using Test
using Random: Xoshiro
using OffsetArrays: OffsetArray

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
    @testset "explicit bindings of submodel return values explain recovery" begin
        @model child() = z ~ Normal()
        @model parent(y) = (a ~ to_submodel(child()); y ~ Normal(a))
        for bind in (condition, fix)
            bound = bind(parent(0.0); a=1.0)
            @test_throws r"Cannot explicitly bind a submodel return value\. Remove the explicit binding.*@varname\(a.z\)" bound(
                Xoshiro(1)
            )
            model = bind(parent(0.0), @varname(a.z) => 1.0)
            @test loglikelihood(model, VarNamedTuple()) ≈
                logpdf(Normal(1.0), 0.0) +
                  (bind === condition ? logpdf(Normal(), 1.0) : 0.0)
        end
    end

    @testset "implicit argument-supplied observations stay in their model" begin
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

    @testset "child namespace binding APIs" begin
        @model function inspect_child(x=[0.0, 1.0]; metadata=0)
            for i in eachindex(x)
                x[i] ~ Normal()
            end
            return __model__
        end
        @model inspect_parent(m) = a ~ to_submodel(m)
        @model function inspect_outer(m)
            b = Vector{Any}(undef, 2)
            b[2] ~ to_submodel(m)
            return b[2]
        end
        child = inspect_child()
        parent = condition(inspect_parent(child), @varname(a.x[1]) => 2.0)
        for model in (parent, inspect_outer(parent))
            local_model = model(Xoshiro(1))
            @test conditioned(local_model) ==
                conditioned(condition(child, @varname(x[1]) => 2.0))
            @test isempty(fixed(local_model))
            for (bind, remove, accessor) in
                ((condition, decondition, conditioned), (fix, unfix, fixed))
                edited = bind(local_model, @varname(x[2]) => 3.0)
                @test accessor(edited)[@varname(x[2])] == 3.0
                @test accessor(edited(Xoshiro(1)))[@varname(x[2])] == 3.0
                removed = remove(edited, @varname(x[2]))
                @test !haskey(accessor(removed(Xoshiro(1))), @varname(x[2]))
                @test isempty(accessor(remove(edited)))
                values = rand(Xoshiro(1), removed)
                vn = model === parent ? @varname(a.x[2]) : @varname(b[2].a.x[2])
                @test haskey(values, vn) == (remove === decondition)
                @test_throws r"ArgumentError: Argument `metadata`" bind(
                    local_model; metadata=1
                )
            end
        end
    end

    @testset "submodel tildes reject model arguments" begin
        @model child() = z ~ Normal()
        @model argument_return(x=0.0) = x ~ to_submodel(child())
        @model indexed_return(x) = x[1] ~ to_submodel(child())
        @model property_return(x) = x.a ~ to_submodel(child())
        @model keyword_return(; x=0.0) = x ~ to_submodel(child())
        @model runtime_return(x, rhs) = x ~ rhs
        @model unprefixed_return(x) = x ~ to_submodel(child(), false)
        @model splatted_return(x...) = x[1] ~ to_submodel(child())
        @model keyword_splatted_return(; x...) = x.a ~ to_submodel(child())
        @model local_return() = x ~ to_submodel(child())
        @model outer(m) = b ~ to_submodel(m)
        err = r"ArgumentError: Submodel tilde .*model argument `x`.*local LHS.*condition"
        for (model, address, value) in (
            (argument_return(), @varname(x), 2.0),
            (argument_return(missing), @varname(x), 2.0),
            (argument_return(nothing), @varname(x), 2.0),
            (indexed_return([0.0]), @varname(x[1]), 2.0),
            (property_return((a=0.0,)), @varname(x.a), 2.0),
            (keyword_return(), @varname(x), 2.0),
            (runtime_return(0.0, to_submodel(child())), @varname(x), 2.0),
            (unprefixed_return(0.0), @varname(x), 2.0),
            (splatted_return(0.0), @varname(x[1]), 2.0),
            (keyword_splatted_return(; a=0.0), @varname(x), (a=2.0,)),
        )
            for wrapped in (model, prefix(model, @varname(p)), outer(model))
                @test_throws err wrapped(Xoshiro(1))
                @test_throws err decondition(wrapped)(Xoshiro(1))
                @test outer(local_return())(Xoshiro(1)) isa Real
            end
            for bind in (condition, fix)
                bound = bind(model, address => value)
                @test_throws err bound(Xoshiro(1))
                local_model = bind(local_return(), @varname(x.z) => 2.0)
                @test local_model(Xoshiro(1)) == 2.0
            end
        end
        # A runtime RHS is classified at the tilde, and ordinary assignment stays Julia.
        @test runtime_return(2.0, Normal())(Xoshiro(1)) == 2.0
        @model function assigned_return(x)
            a ~ to_submodel(fix(child(); z=2.0))
            x = a
            return x ~ Normal()
        end
        @test loglikelihood(assigned_return(0.0), VarNamedTuple()) == logpdf(Normal(), 2.0)
    end

    @testset "submodel return bindings with manual prefixes" begin
        @model child() = x ~ Normal()
        @model manual(a, m) = a ~ to_submodel(m, false)
        @model nested(m) = outer ~ to_submodel(m)
        @model local_manual(m) = a ~ to_submodel(m, false)
        for child_model in (child(), prefix(child(), @varname(b)))
            for m in (
                condition(decondition(manual(0.0, child_model)); a=3.0),
                fix(decondition(manual(0.0, child_model)); a=3.0),
            )
                for model in (m, nested(m))
                    @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" model(
                        Xoshiro(1)
                    )
                    @test nested(local_manual(child_model))(Xoshiro(1)) isa Real
                end
            end
        end
    end

    @testset "indexed submodel namespace bindings avoid per-call allocations" begin
        unbound = indexed_observations(zeros(100))
        accs = VarInfo(LogPriorAccumulator(), LogLikelihoodAccumulator())
        strategy = InitFromParams((; mu=0.1), nothing)
        init!!(unbound, accs, strategy, UnlinkAll())
        baseline = @allocated init!!(unbound, accs, strategy, UnlinkAll())
        for bindings in
            ((@varname(x[1].y) => 0.0,), Tuple(@varname(x[i].y) => 0.0 for i in 1:100))
            model = condition(unbound, bindings...)
            # Keep the conditioned indexed path concrete as well as allocation-bounded.
            @test @inferred(init!!(model, accs, strategy, UnlinkAll())) isa Tuple
            @test (@allocated init!!(model, accs, strategy, UnlinkAll())) <= baseline + 128
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

    @testset "range-prefixed submodel namespace bindings" begin
        @model range_child() = (x ~ Normal(); [x, x])
        @model range_parent() = (a = zeros(2); a[1:2] ~ to_submodel(range_child()); a)
        @model range_nested(child) = outer ~ to_submodel(child)
        for bind in (condition, fix)
            pair = @varname(a[1:2].x) => 2.0
            bindings = DynamicPPL.templated_setindex!!(
                VarNamedTuple(), 2.0, pair.first, zeros(2)
            )
            for form in (pair, bindings), wrap in (identity, range_nested)
                @test_throws r"a\[.*before prefixing" wrap(bind(range_parent(), form))(
                    Xoshiro(1)
                )
            end
            @test_throws "Cannot explicitly bind a submodel return value" bind(
                range_parent(); a=zeros(2)
            )(
                Xoshiro(1)
            )
        end
    end

    @testset "binding construction uses prefix templates" begin
        @model template_leaf() = x ~ Normal()
        @model template_covariate(z) = x ~ Normal(z[1])
        @model template_parent(child) = outer ~ to_submodel(child)
        for (name, template) in
            ((@varname(p[:]), zeros(2)), (@varname(p[0]), OffsetArray(zeros(2), 0:1))),
            bind in (condition, fix),
            leaf in (template_leaf(), template_covariate(zeros(2)))

            original = prefix(leaf, name; template)
            vn = AbstractPPL.prefix(@varname(x), name)
            if name == @varname(p[:])
                @test_throws r"p\[.*before prefixing" bind(original, vn => 4.0)
                continue
            end
            bound = bind(original, vn => 4.0)
            @test bound(Xoshiro(1)) == 4.0
            @test template_parent(bound)(Xoshiro(1)) == 4.0
            @test loglikelihood(bound, VarNamedTuple()) ≈
                (bind === condition ? logpdf(Normal(), 4.0) : 0.0)
        end
    end

    @testset "slice prefixes resolve explicit bindings" begin
        @model slice_leaf() = x ~ Normal()
        @model slice_parent(child) = unused ~ to_submodel(child, false)
        @model slice_nested(child) = outer ~ to_submodel(child)
        for name in (@varname(a), @varname(a[1]), @varname(a[:]), @varname(a[1:2])),
            (bind, accessor) in ((condition, conditioned), (fix, fixed))

            child = prefix(bind(slice_leaf(); x=3.0), name; template=zeros(2))
            bindings = accessor(child)
            @test keys(bindings) == [AbstractPPL.prefix(@varname(x), name)]
            @test bindings[AbstractPPL.prefix(@varname(x), name)] == 3.0
            @test isempty(rand(Xoshiro(1), child))
            @test child(Xoshiro(1)) == 3.0
            @test slice_parent(child)(Xoshiro(1)) == 3.0
            parent = bind(
                slice_parent(prefix(slice_leaf(), name; template=zeros(2))),
                DynamicPPL.templated_setindex!!(VarNamedTuple(), (; x=4.0), name, zeros(2)),
            )
            @test parent(Xoshiro(1)) == 4.0
            @test slice_nested(parent)(Xoshiro(1)) == 4.0
        end
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
        @model function matrix_template(storage)
            a = copy(storage)
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

    @testset "conditioning argument-backed submodel LHS variables" begin
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
        @model function parent_with_return_value(a)
            a[1] ~ to_submodel(observed_child())
            return a
        end
        for model in (
            parent_with_return_value(zeros(1)),
            decondition(parent_with_return_value(zeros(1))),
        )
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" model(
                Xoshiro(1)
            )
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" VarInfo(
                Xoshiro(1), model
            )
            @test g(2.0)(Xoshiro(1)) isa Real
        end
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

@testset "shared unprefixed binding addresses" begin
    @model shared_left() = x ~ Normal()
    @model shared_right() = y ~ Normal()
    @model function shared_parent()
        a ~ to_submodel(shared_left(), false)
        b ~ to_submodel(shared_right(), false)
        return (a, b)
    end
    for bind in (condition, fix)
        @test bind(shared_parent(); x=1.0, y=2.0)(Xoshiro(1)) == (1.0, 2.0)
    end
end

@testset "recursive removals" begin
    rec = DynamicPPL.Recursive()
    @model observation(counts, mu) = counts ~ Normal(mu)
    @model function generator()
        mu ~ Normal()
        y ~ to_submodel(observation(missing, mu))
        return (mu, y)
    end
    latent = decondition(generator(), rec, @varname(y.counts))
    @test keys(rand(Xoshiro(1), latent)) == [@varname(mu), @varname(y.counts)]
    @test returned(
        latent, VarNamedTuple((@varname(mu) => 1.0, @varname(y.counts) => 2.0))
    ) == (1.0, 2.0)
    @model leaf(x=2.0) = x ~ Normal()
    @model parent(m) = a ~ to_submodel(m)
    @model outer(m) = b ~ to_submodel(m)
    fixed_child = parent(fix(leaf(); x=5.0))
    @test unfix(fixed_child, rec, @varname(a.x))() == 2.0
    @test decondition(fixed_child, rec, @varname(a.x))() == 5.0
    @test unfix(fixed_child, rec)() == 2.0
    @test isempty(conditioned(decondition(parent(leaf()), rec)))
    @test keys(rand(Xoshiro(1), decondition(parent(leaf()), rec))) == [@varname(a.x)]
    removed = decondition(parent(leaf()), rec, @varname(a.x))
    @test keys(rand(Xoshiro(1), decondition(removed, rec, @varname(a.x)))) ==
        [@varname(a.x)]
    @test condition(removed, @varname(a.x) => 3.0)() == 3.0
    @test condition(outer(removed), @varname(b.a.x) => 4.0)() == 4.0
    @test keys(rand(Xoshiro(1), prefix(removed, @varname(p)))) == [@varname(p.a.x)]
    @test keys(rand(Xoshiro(1), outer(removed))) == [@varname(b.a.x)]
    @test unfix(leaf(), rec, @varname(x))(Xoshiro(1)) == 2.0
    @test unfix(parent(leaf()), rec, @varname(a.x))(Xoshiro(1)) == 2.0
    @test keys(
        rand(Xoshiro(1), decondition(parent(decondition(leaf())), rec, @varname(a.x)))
    ) == [@varname(a.x)]
    @test fixed(unfix(fixed_child, rec)) == VarNamedTuple()
    @model function runtime()
        mu ~ Normal()
        a ~ to_submodel(condition(leaf(); x=2mu))
        return a
    end
    @test keys(rand(Xoshiro(2), decondition(runtime(), rec, @varname(a.x)))) ==
        [@varname(mu), @varname(a.x)]
    for backend in (AutoForwardDiff(), AutoMooncake(; config=nothing))
        ldf = LogDensityFunction(latent; adtype=backend)
        @test LogDensityProblems.dimension(ldf) == 2
        value, grad = LogDensityProblems.logdensity_and_gradient(ldf, [1.0, 2.0])
        @test value ≈ logpdf(Normal(), 1.0) + logpdf(Normal(1.0), 2.0)
        @test grad ≈ [0.0, -1.0]
    end
end

@testset "unmatched recursive removals of local LHS variables" begin
    rec = DynamicPPL.Recursive()
    @model removal_scalar() = x ~ Normal()
    @model function removal_parent()
        m ~ Normal()
        a ~ to_submodel(removal_scalar())
        return m
    end
    @model function removal_fields()
        m = (x=zeros(1),)
        m.x[1] ~ Normal()
        a ~ to_submodel(removal_scalar())
        return m
    end
    @model removal_wrapper(child) = b ~ to_submodel(child)
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        for (model, name) in
            ((removal_parent(), @varname(m)), (removal_fields(), @varname(m.x[1])))
            @test name in keys(rand(Xoshiro(1), remove(model, rec, name)))
            prefixed_name = DynamicPPL.maybe_prefix(name, @varname(p))
            @test prefixed_name in keys(
                rand(Xoshiro(1), remove(prefix(model, @varname(p)), rec, prefixed_name))
            )
            deferred = remove(
                removal_wrapper(model), rec, DynamicPPL.maybe_prefix(name, @varname(b))
            )
            @test DynamicPPL.maybe_prefix(name, @varname(b)) in
                keys(rand(Xoshiro(1), deferred))
            matched = remove(bind(model, name => 2.0), rec, name)
            @test name in keys(rand(Xoshiro(1), matched))
        end
    end
end

@testset "recursive removals overlap whole LHS variables" begin
    rec = DynamicPPL.Recursive()
    @model function removal_whole(t, run=true)
        run && (t ~ product_distribution((a=Normal(),)))
        return nothing
    end
    @model removal_whole_local() = t ~ product_distribution((a=Normal(),))
    @model function removal_runtime(child)
        rhs = to_submodel(child)
        return a ~ rhs
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = decondition(removal_whole((a=2.0,)))
        for (model, name, whole_name) in (
            (child, @varname(t.a), @varname(t)),
            (removal_whole_local(), @varname(t.a), @varname(t)),
            (prefix(child, @varname(p)), @varname(p.t.a), @varname(p.t)),
            (removal_runtime(removal_runtime(child)), @varname(a.a.t.a), @varname(a.a.t)),
        )
            @test whole_name in keys(rand(Xoshiro(1), remove(model, rec, name)))
            matched = remove(bind(model, whole_name => (a=2.0,)), rec, name)
            @test whole_name in keys(rand(Xoshiro(1), matched))
        end
        skipped = decondition(removal_whole((a=2.0,), false))
        for (model, name, whole_name) in (
            (skipped, @varname(t.a), @varname(t)),
            (removal_runtime(skipped), @varname(a.t.a), @varname(a.t)),
        )
            @test remove(model, rec, name)(Xoshiro(1)) === nothing
            @test remove(bind(model, whole_name => (a=2.0,)), rec, name)(Xoshiro(1)) ===
                nothing
        end
        # A child's removal cannot clear an enclosing binding.
        removed_child = remove(removal_runtime(child), rec, @varname(a.t.a))
        enclosing = bind(removal_runtime(removed_child), @varname(a.a.t) => (a=2.0,))
        @test enclosing(Xoshiro(1)) === nothing
        @test isempty(keys(rand(Xoshiro(1), enclosing)))
        @test logjoint(enclosing, (;)) ≈ (bind === condition ? logpdf(Normal(), 2.0) : 0.0)
    end
end

@testset "argument submodel tildes reject sibling edits only when reached" begin
    @model removal_sibling_leaf(x=2.0) = x ~ Normal()
    @model function removal_siblings(a, run)
        run && (a.child ~ to_submodel(removal_sibling_leaf()))
        a.obs ~ Normal()
        return a
    end
    @model removal_sibling_parent(child) = b ~ to_submodel(child)
    for (bind, remove) in ((condition, decondition), (fix, unfix)),
        wrap in (identity, m -> prefix(m, @varname(p)), removal_sibling_parent)

        reached = remove(
            bind(removal_siblings((child=0.0, obs=1.0), true), @varname(a.obs) => 3.0),
            @varname(a.obs)
        )
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" wrap(
            reached
        )(
            Xoshiro(1)
        )
        skipped = remove(
            bind(removal_siblings((child=0.0, obs=1.0), false), @varname(a.obs) => 3.0),
            @varname(a.obs)
        )
        @test wrap(skipped)(Xoshiro(1)).obs isa Real
    end
end

@testset "recursive removals through computed and branched submodels" begin
    rec = DynamicPPL.Recursive()
    @model removal_child() = t ~ Normal()
    @model function removal_mixed(child, c)
        if c
            a ~ Normal()
        else
            a ~ to_submodel(child)
        end
        return a
    end
    @model function removal_rhs(child)
        rhs = to_submodel(child)
        return a ~ rhs
    end
    my_sub(child) = to_submodel(child)
    @model removal_call(child) = a ~ my_sub(child)
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = bind(removal_child(); t=2.0)
        for parent in (m -> removal_mixed(m, false), removal_rhs, removal_call)
            model = parent(child)
            @test model(Xoshiro(1)) == 2.0
            removed = remove(model, rec, @varname(a.t))
            @test keys(rand(Xoshiro(1), removed)) == [@varname(a.t)]
            @test logjoint(removed, (; a=(; t=0.5))) ≈ logpdf(Normal(), 0.5)
            @test keys(rand(Xoshiro(1), remove(removed, rec, @varname(a.t)))) ==
                [@varname(a.t)]
        end
        @test keys(
            rand(Xoshiro(1), remove(removal_mixed(child, true), rec, @varname(a.t)))
        ) == [@varname(a)]
    end
end

@testset "matched recursive removal at evaluation" begin
    rec = DynamicPPL.Recursive()
    @model leaf(x) = x ~ Normal()
    @model parent(m) = a ~ to_submodel(m)
    m = decondition(condition(parent(leaf(2.0)), @varname(a.x) => 3.0), rec, @varname(a.x))
    @test logjoint(m, (; a=(; x=0.5))) ≈ logpdf(Normal(), 0.5)
end

@testset "recursive removal storage and scope" begin
    rec = DynamicPPL.Recursive()
    @model function indexed(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @model parent(m) = a ~ to_submodel(m)
    child = condition(indexed(zeros(2)); x=ones(3))
    @test length(decondition(parent(child), rec, @varname(a.x))(Xoshiro(1))) == 2
    @test unfix(parent(fix(indexed(zeros(2)); x=ones(3))), rec, @varname(a.x))() == zeros(2)
    partial = decondition(parent(child), rec, @varname(a.x[2]))
    @test returned(partial, VarNamedTuple((@varname(a.x[2]) => 5.0,))) == [1.0, 5.0, 1.0]
    @test condition(decondition(parent(child), rec), @varname(a.x) => [4.0, 5.0])() ==
        [4.0, 5.0]
    @test returned(
        condition(decondition(parent(indexed(zeros(2))), rec), @varname(a.x[1]) => 4.0),
        VarNamedTuple((@varname(a.x[2]) => 5.0,)),
    ) == [4.0, 5.0]
    @model scalar(x) = x ~ Normal()
    @model function shared(x)
        x ~ Normal()
        a ~ to_submodel(scalar(2.0), false)
        return (x, a)
    end
    both = decondition(shared(1.0), rec, @varname(x))
    @test_throws r"x.*Prefix.*Recursive" both(Xoshiro(1))
    @test condition(both; x=3.0)(Xoshiro(1)) == (3.0, 3.0)
    @test isempty(conditioned(decondition(decondition(parent(child), rec), rec)))
    once = decondition(parent(child), rec, @varname(a.x))
    twice = decondition(once, rec, @varname(a.x))
    @test typeof(twice) == typeof(once)
    @test_logs DynamicPPL.DebugUtils.check_model(Xoshiro(1), twice)
    @model branch(run) = (run && (a ~ to_submodel(scalar(1.0))); nothing)
    @test decondition(branch(false), rec, @varname(a.x))() === nothing
    @model function repeated()
        for i in 1:2
            unused ~ to_submodel(i == 1 ? scalar(1.0) : decondition(scalar(1.0)), false)
        end
    end
    @test decondition(repeated(), rec, @varname(x))(Xoshiro(1)) === nothing
end

@testset "recursive removal namespace boundaries" begin
    rec = DynamicPPL.Recursive()
    @model leaf(x) = x ~ Normal()
    @model parent(a) = a ~ to_submodel(leaf(missing))
    @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" decondition(
        parent(missing), rec, @varname(a.x)
    )(
        Xoshiro(1)
    )
    @model local_parent() = a ~ to_submodel(decondition(leaf(missing)))
    @test local_parent()(Xoshiro(1)) isa Real
    @model function sharearr(x)
        x[1] ~ Normal()
        x[2] ~ Normal()
        a ~ to_submodel(arr([2.0, 2.0]), false)
        return (x, a)
    end
    @model function arr(x)
        x[1] ~ Normal()
        x[2] ~ Normal()
        return x
    end
    m = condition(sharearr([0.0, 0.0]); x=[4.0, 4.0])
    @test condition(m, @varname(x[1]) => 3.0)() == ([3.0, 4.0], [3.0, 4.0])
    @model ownprefix() = a ~ to_submodel(leaf(1.0))
    @test_throws ArgumentError decondition(
        prefix(ownprefix(), @varname(p)), rec, @varname(q.x)
    )
end

@testset "recursive unfix releases whole shape owners" begin
    rec = DynamicPPL.Recursive()
    @model owner_parent(m) = a ~ to_submodel(m)
    @model function owner_grows(x)
        x = vcat(x, 0.0)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    old = fix(owner_grows(zeros(2)); x=ones(2))
    expected = fix(unfix(old), @varname(x[1]) => 2.0)(Xoshiro(1))
    for names in ((), (@varname(a.x),))
        model = fix(unfix(owner_parent(old), rec, names...), @varname(a.x[1]) => 2.0)
        @test model(Xoshiro(1)) == expected
    end
end

@testset "recursive removal resolves indexed prefixes" begin
    rec = DynamicPPL.Recursive()
    @model prefix_removal_leaf(x=2.0) = x ~ Normal()
    @model function prefix_removal_array(x)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    @test keys(
        rand(
            Xoshiro(1), decondition(prefix_removal_array([1.0, 2.0]), rec, @varname(x[end]))
        ),
    ) == [@varname(x[2])]
    @model prefix_removal_parent(m) = a ~ to_submodel(m)
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = prefix(
            bind(prefix_removal_leaf(); x=3.0), @varname(p[end]); template=zeros(2)
        )
        removed = remove(child, rec, @varname(p[end].x))
        @test isempty(bind === condition ? conditioned(removed) : fixed(removed))
        @test remove(removed, rec, @varname(p[2].x))(Xoshiro(1)) == removed(Xoshiro(1))
        if bind === condition
            @test keys(rand(Xoshiro(1), removed)) == [@varname(p[2].x)]
        else
            @test removed(Xoshiro(1)) == 2.0
        end
    end
    model = prefix(
        prefix_removal_parent(prefix_removal_leaf()), @varname(p[end]); template=zeros(2)
    )
    removed = decondition(model, rec, @varname(p[end].a.x))
    @test keys(rand(Xoshiro(1), removed)) == [@varname(p[2].a.x)]
end

@testset "recursive removal classifies the executed RHS" begin
    rec = DynamicPPL.Recursive()
    @model removal_leaf(x) = x ~ Normal()
    @model function removal_branch(a, observed)
        if observed
            a ~ Normal()
        else
            a ~ to_submodel(removal_leaf(missing))
        end
        return a
    end
    @model function removal_dynamic(a)
        child = to_submodel(removal_leaf(missing))
        a ~ child
        return a
    end
    @model removal_outer(m) = b ~ to_submodel(m)
    m = decondition(removal_branch(1.0, true), rec, @varname(a))
    @test keys(rand(Xoshiro(1), m)) == [@varname(a)]
    @test keys(
        rand(
            Xoshiro(1),
            unfix(decondition(fix(removal_branch(1.0, true); a=2.0), rec, @varname(a))),
        ),
    ) == [@varname(a)]
    m = decondition(removal_outer(removal_branch(1.0, true)), rec, @varname(b.a))
    @test keys(rand(Xoshiro(1), m)) == [@varname(b.a)]
    @test_throws ArgumentError decondition(
        removal_branch(missing, true), rec, @varname(a.x)
    )(
        Xoshiro(1)
    )
    for child in (removal_dynamic(missing), removal_branch(missing, false))
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" decondition(
            child, rec, @varname(a.x)
        )(
            Xoshiro(1)
        )
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" decondition(
            removal_outer(child), rec, @varname(b.a.x)
        )(
            Xoshiro(1)
        )
    end
end

@testset "recursive partial edits preserve replacement shape" begin
    rec = DynamicPPL.Recursive()
    @model function shaped_child(y, mu)
        for i in eachindex(y)
            y[i] ~ Normal(mu)
        end
        return y
    end
    @model function shaped_parent(n)
        mu ~ Normal()
        a ~ to_submodel(shaped_child(zeros(n), mu))
        return a
    end
    for (n, replacement) in ((3, [1.0, 2.0]), (2, [1.0, 2.0, 3.0]))
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            whole = bind(shaped_parent(n), @varname(a.y) => replacement)
            edited = bind(whole, @varname(a.y[1]) => 4.0)
            expected = [4.0; replacement[2:end]]
            @test returned(edited, (mu=0.3,)) == expected
            partial = remove(whole, rec, @varname(a.y[1]))
            @test returned(
                partial, VarNamedTuple((@varname(mu) => 0.3, @varname(a.y[1]) => 4.0))
            ) == (bind === condition ? expected : [0.0; replacement[2:end]])
            for backend in (AutoForwardDiff(), AutoMooncake(; config=nothing))
                ldf = LogDensityFunction(edited; adtype=backend)
                value, gradient = LogDensityProblems.logdensity_and_gradient(ldf, [0.3])
                @test value ≈
                    logpdf(Normal(), 0.3) +
                      (bind === condition ? sum(logpdf.(Normal(0.3), expected)) : 0.0)
                @test gradient ≈ [bind === condition ? sum(expected .- 0.3) - 0.3 : -0.3]
            end
        end
    end
end

@model depth_leaf(x) = x ~ Normal()
@model depth_parent(x, child) = (x ~ Normal(); a ~ to_submodel(child, false); a)
@testset "bindings replace recursive removals at their address" begin
    m = condition(
        decondition(depth_parent(2.0, depth_leaf(3.0)), DynamicPPL.Recursive()); x=7.0
    )
    @test loglikelihood(m, (;)) ≈ 2 * logpdf(Normal(), 7.0)
    m = fix(
        unfix(depth_parent(2.0, fix(depth_leaf(3.0); x=9.0)), DynamicPPL.Recursive()); x=7.0
    )
    @test loglikelihood(m, (;)) == 0.0
end

@model function skipped_removal_leaf(run)
    run && (x ~ Normal())
    return :ok
end
@model skipped_removal_parent(run) = a ~ to_submodel(skipped_removal_leaf(run))
@testset "unmatched removals ignore whether the tilde is reached" begin
    for remove in (decondition, unfix)
        m = remove(skipped_removal_parent(false), DynamicPPL.Recursive(), @varname(a.x))
        @test m(Xoshiro(1)) == :ok
        @test_logs (:warn, r"removal") match_mode = :any DynamicPPL.DebugUtils.check_model(
            m
        )
        m = remove(skipped_removal_parent(true), DynamicPPL.Recursive(), @varname(a.x))
        @test m(Xoshiro(1)) == :ok
    end
end

@model address_leaf() = x ~ Normal()
@model address_parent(m) = a ~ to_submodel(m)
@testset "unused nested removal addresses" begin
    for remove in (decondition, unfix), name in (@varname(a.a.x), @varname(a.a.absent))
        m = remove(
            address_parent(address_parent(address_leaf())), DynamicPPL.Recursive(), name
        )
        @test keys(rand(Xoshiro(1), m)) == [@varname(a.a.x)]
        @test_logs (:warn, r"Recursive removal.*unused") DynamicPPL.DebugUtils.check_model(
            Xoshiro(1), m
        )
    end
end

@model closed_leaf(x=3.0) = x ~ Normal()
@model closed_shared(x, child) = (x ~ Normal(); a ~ to_submodel(child, false); a)
@model closed_shared_runtime(x, child, dist) = (x ~ dist; a ~ to_submodel(child, false); a)
@testset "shared removal" begin
    for remove in (decondition, unfix),
        m in (
            closed_shared(2.0, fix(closed_leaf(); x=4.0)),
            closed_shared_runtime(2.0, fix(closed_leaf(); x=4.0), Normal()),
        )

        @test_throws r"x.*[Pp]refix.*Recursive" remove(
            m, DynamicPPL.Recursive(), @varname(x)
        )(
            Xoshiro(1)
        )
    end
end

@testset "partial runtime return arguments are rejected" begin
    @model runtime_return_leaf(x=3.0) = x ~ Normal()
    @model runtime_return(a, rhs) = a ~ rhs
    @model runtime_return_outer(m) = b ~ to_submodel(m)
    for arg in ([1.0, 2.0], (1.0, 2.0), (x=1.0, y=2.0))
        vn = arg isa NamedTuple ? @varname(a.x) : @varname(a[1])
        for edit in
            (m -> decondition(m, vn), m -> condition(m, vn => 4.0), m -> fix(m, vn => 4.0))
            m = edit(runtime_return(arg, to_submodel(runtime_return_leaf())))
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" m(
                Xoshiro(1)
            )
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" runtime_return_outer(
                m
            )(
                Xoshiro(1)
            )
            @test runtime_return_outer(runtime_return_leaf())(Xoshiro(1)) == 3.0
        end
    end
end

@testset "produced bindings round-trip through submodels" begin
    @model leaf() = x ~ Normal()
    @model function prefixed_parent()
        z ~ Normal()
        a ~ to_submodel(leaf())
        return (z, a)
    end
    @model function unprefixed_parent()
        z ~ Normal()
        a ~ to_submodel(leaf(), false)
        return (z, a)
    end
    @model outer(child) = b ~ to_submodel(child)
    for (model, child_address) in
        ((prefixed_parent(), @varname(a.x)), (unprefixed_parent(), @varname(x)))
        draw = rand(Xoshiro(42), model)
        for bind in (condition, fix)
            bound = bind(model, draw)
            @test bound(Xoshiro(1)) == (draw[@varname(z)], draw[child_address])
            @test isempty(rand(Xoshiro(1), bound))
        end
    end
    for child_bind in (condition, fix), parent_bind in (condition, fix)
        child = child_bind(leaf(); x=2.0)
        middle = parent_bind(outer(child), @varname(b.x) => 3.0)
        model = condition(outer(middle), @varname(b.b.x) => 1.0)
        @test model(Xoshiro(1)) == 1.0
        @test loglikelihood(model, (;)) ≈ logpdf(Normal(), 1.0)
    end
end

end
