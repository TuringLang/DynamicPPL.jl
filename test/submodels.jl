module DPPLSubmodelTests

using DynamicPPL
using AbstractPPL: AbstractPPL, @of, of
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
@model function indexed_observations(y, unobserved=())
    mu ~ Normal()
    x = similar(y)
    for i in eachindex(y)
        child = condition(indexed_observation(missing, mu); y=y[i])
        x[i] ~ to_submodel(i in unobserved ? decondition(child) : child)
    end
    return x
end
@model nested_observations(y) = a ~ to_submodel(indexed_observations(y))

@model namespace_leaf() = x ~ Normal()
@model single_namespace_parent() = (
    p = zeros(1, 1); p[1] ~ to_submodel(namespace_leaf()); p[1]
)
@model function mixed_namespace_parent()
    p = zeros(2, 2)
    p[1] ~ to_submodel(condition(namespace_leaf(); x=2.0))
    p[2, 2] ~ to_submodel(decondition(condition(namespace_leaf(); x=2.0)))
    return p
end

@model function nested_geometry_child(n)
    x = [zeros(2) for _ in 1:n]
    x[1][1] ~ Normal()
    return x[1][1]
end
@model retained_geometry_child() = (x = zeros(2); x[1] ~ Normal(); __model__)
@noinline function replaced_geometry_model(bind)
    value = zeros(2)
    reference = WeakRef(value)
    child = bind(retained_geometry_child(); x=value)
    remove = bind === condition ? decondition : unfix
    parent = bind(
        remove(argument_geometry_parent(child), @varname(a.x)), @varname(a.x) => ones(2)
    )
    return parent(Xoshiro(1)), reference
end
@model argument_geometry_parent(child) = a ~ to_submodel(child)
@model held_geometry_parent(child) = a ~ to_submodel(child[])
@model placeholder_child(x) = (x[1] ~ Normal(); x[2] ~ Normal(); x)
@model nested_inner() = (x ~ Normal(); y ~ Normal())
@model nested_mid() = b ~ to_submodel(nested_inner())
@model nested_outer(m) = a ~ to_submodel(m)
@model removal_geometry_middle(child) = (z ~ Normal(); b ~ to_submodel(child); __model__)
@model function removal_geometry_child(n)
    x = [zeros(2 + i % 2) for i in 1:n]
    x[1][1] ~ Normal()
    return x[1][1]
end

@model deferred_leaves() = (x ~ Normal(); y ~ Normal(); (x=x, y=y))
@model deferred_structure() = (b ~ product_distribution((x=Normal(), y=Normal())); b)
@model deferred_array_leaves() = (x = zeros(2); x[1] ~ Normal(); x[2] ~ Normal(); x)
@model scalar_z() = z ~ Normal()
@model outer_model(m) = b ~ to_submodel(m)
@model template_parent(child) = outer ~ to_submodel(child)
@model observed_child(x=2.0) = x ~ Normal()
@model function normal_pair()
    x ~ Normal()
    y ~ Normal()
    return (x, y)
end
@model removal_wrapper(child) = b ~ to_submodel(child)
@model function removal_runtime(child)
    rhs = to_submodel(child)
    return a ~ rhs
end
@model depth_leaf(x) = x ~ Normal()
@model function indexed(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end
@model closed_leaf(x=3.0) = x ~ Normal()

@testset "ancestor partial removal uses inherited placeholder storage" begin
    values = [1.0, 2.0]
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        bound = bind(
            argument_geometry_parent(placeholder_child(nothing)), @varname(a.x) => values
        )
        result = remove(bound, @varname(a.x[1]))(Xoshiro(1))
        @test result[2] == 2.0
    end
end

@testset "partial cross-role bindings need placeholder storage" begin
    for (model, root, part) in (
        (placeholder_child(nothing), @varname(x), @varname(x[1])),
        (
            argument_geometry_parent(placeholder_child(nothing)),
            @varname(a.x),
            @varname(a.x[1])
        ),
    )
        partial = decondition(condition(model, root => [1.0, 2.0]), part)
        if root == @varname(x)
            @test_throws r"Cannot make a partial binding of argument `x` with whole `nothing` storage" fix(
                partial, part => 3.0
            )
        else
            stored_parent = fix(partial, part => 3.0)
            @test_throws r"Cannot make a partial binding of argument `a.x` with whole `nothing` storage" stored_parent()
        end
    end
end

@testset "nested submodel bindings split named-tuple roots" begin
    first = condition(
        nested_outer(nested_mid()),
        VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
    )
    @test condition(first, @varname(a.b.y) => 2.0)() == 2.0
    @test @varname(a.b.x) in keys(conditioned(first))
    second = condition(nested_outer(nested_mid()), @varname(a.b.x) => 1.0)
    second = condition(
        second, VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((y=2.0,)),)),))
    )
    @test second() == 2.0
    @test @varname(a.b.y) in keys(conditioned(second))
end
@model body_namespace_parent() = a ~ to_submodel(nested_mid())
@testset "nested leaf density inference" begin
    if VERSION >= v"1.12"
        for model in (body_namespace_parent(), nested_outer(nested_mid())),
            bind in (condition, fix)

            ldf = LogDensityFunction(bind(model, @varname(a.b.y) => 1.0))
            expected =
                logpdf(Normal(), 0.3) + (bind === condition ? logpdf(Normal(), 1.0) : 0.0)
            @test (@inferred LogDensityProblems.logdensity(ldf, [0.3])) ≈ expected
        end
    end
end
function namespace_density_allocations(ldf, params)
    return @allocated LogDensityProblems.logdensity(ldf, params)
end
@testset "deferred namespace density allocations" begin
    model = condition(
        body_namespace_parent(),
        VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((y=1.0,)),)),)),
    )
    for (bound, params) in
        ((model, [0.3]), (condition(model, @varname(a.b.x) => 0.5), Float64[]))
        ldf = LogDensityFunction(bound)
        namespace_density_allocations(ldf, params)
        @test namespace_density_allocations(ldf, params) <= 432
    end
end
@model deeper_namespace_child() = c ~ to_submodel(nested_mid())
@model structured_namespace_child() = (
    b ~ product_distribution((u=Normal(), v=Normal())); b
)
@model function sibling_namespaces(child)
    a ~ to_submodel(structured_namespace_child())
    c ~ to_submodel(child)
    return a
end

@testset "nested roots use the executing child's namespace" begin
    for bind in (condition, fix),
        model in (nested_outer(nested_mid()), body_namespace_parent())

        whole = bind(model, VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)))
        @test bind(whole, @varname(a.b.y) => 2.0)() == 2.0
        partial = bind(model, @varname(a.b.x) => 1.0)
        @test bind(
            partial, VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((y=2.0,)),)),))
        )() == 2.0
        deep = bind(
            nested_outer(deeper_namespace_child()),
            VarNamedTuple((
                a=VarNamedTuple((c=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
            )),
        )
        @test bind(deep, @varname(a.c.b.y) => 2.0)() == 2.0
    end
    for bind in (condition, fix)
        model = bind(sibling_namespaces(nested_mid()); a=(b=(u=1.0, v=2.0),))
        listing = bind === condition ? conditioned : fixed
        @test collect(keys(listing(model))) == [@varname(a.b)]
        @test bind(model, @varname(a.b) => (u=3.0, v=4.0))() == (u=3.0, v=4.0)
    end
    @test fix(sibling_namespaces(nested_mid()); a=(b=(u=1.0,),))() == (u=1.0,)
end

@model deferred_mid() = b ~ to_submodel(deferred_leaves())
@model deferred_parent() = a ~ to_submodel(deferred_mid())
@model deferred_structured_parent() = a ~ to_submodel(deferred_structure())
@testset "child namespace bindings merge and replace fields" begin
    for (bind, remove, list) in ((condition, decondition, conditioned), (fix, unfix, fixed))
        m = bind(
            deferred_parent(),
            VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
        )
        @test @varname(a.b.x) in keys(list(m))
        @test bind(m; a=(b=(y=2.0,),))() == (x=1.0, y=2.0)
        @test bind(
            bind(
                deferred_parent(),
                VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
            );
            a=(b=(y=2.0,),),
        )() == (x=1.0, y=2.0)
        @test bind(
            remove(m, @varname(a.b.x)), @varname(a.b.x) => 3.0, @varname(a.b.y) => 2.0
        )() == (x=3.0, y=2.0)
        @test bind(remove(m, @varname(a.b)), @varname(a.b.y) => 2.0)(Xoshiro(1)).y == 2.0
        s = bind(
            deferred_structured_parent(),
            VarNamedTuple((a=VarNamedTuple((b=(x=1.0, y=2.0),)),)),
        )
        @test bind(s, @varname(a.b) => (x=3.0, y=4.0))() == (x=3.0, y=4.0)
        @test_throws ArgumentError bind(s, @varname(a.b) => (x=3.0,))
        bind === fix &&
            @test bind(remove(s, @varname(a.b)), @varname(a.b) => (x=3.0,))() == (x=3.0,)
        @test bind(s, @varname(a.b.x) => 3.0)() == (x=3.0, y=2.0)
        @test_throws ArgumentError bind(
            deferred_parent(), @varname(a.b) => (x=1.0,), @varname(a.b.x) => 2.0
        )
    end
    @test condition(
        deferred_parent(),
        VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
        @varname(a.b.y) => 2.0,
    )() == (x=1.0, y=2.0)
    @test condition(
        deferred_parent(),
        @varname(a.b.x) => 1.0,
        VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((y=2.0,)),)),)),
    )() == (x=1.0, y=2.0)
    m = condition(
        deferred_parent(), VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),))
    )
    @test fix(m, @varname(a.b.y) => 2.0)() == (x=1.0, y=2.0)
    @test_throws ArgumentError fix(m, @varname(a.b.x) => 2.0)
    @test fix(
        decondition(m, @varname(a.b.x)), @varname(a.b.x) => 3.0, @varname(a.b.y) => 2.0
    )() == (x=3.0, y=2.0)
end
@testset "deferred listing round trips" begin
    for (bind, listing) in ((condition, conditioned), (fix, fixed))
        m = bind(
            bind(
                deferred_parent(),
                VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
            );
            a=(b=(y=2.0,),),
        )
        @test collect(keys(listing(m))) == [@varname(a.b.x), @varname(a.b.y)]
        @test bind(deferred_parent(), listing(m))() == (x=1.0, y=2.0)
        m = bind(
            bind(
                deferred_structured_parent(),
                VarNamedTuple((a=VarNamedTuple((b=(x=1.0, y=2.0),)),)),
            );
            a=(b=(x=3.0, y=4.0),),
        )
        @test bind(deferred_structured_parent(), listing(m))() == (x=3.0, y=4.0)
    end
end
@testset "deferred removal listing changes role" begin
    m = decondition(
        condition(
            deferred_parent(),
            VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0, y=2.0)),)),)),
        ),
        @varname(a.b.x)
    )
    rebound = fix(deferred_parent(), conditioned(m))
    @test rebound(Xoshiro(1)).y == 2.0
    @test rebound(Xoshiro(1)).x != 1.0
end

@model deferred_array_struct() = (
    q = (x=zeros(2),); q.x[1] ~ Normal(); q.x[2] ~ Normal(); q
)
@model deferred_array_parent() = a ~ to_submodel(deferred_array_struct())
@model deferred_array_middle() = b ~ to_submodel(deferred_array_leaves())
@model deferred_arrays_parent() = a ~ to_submodel(deferred_array_middle())
@testset "deferred array storage" begin
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        m = bind(
            deferred_array_parent(), VarNamedTuple((a=VarNamedTuple((q=(x=[1.0, 2.0],),)),))
        )
        @test bind(m, @varname(a.q.x[2]) => 3.0)() == (x=[1.0, 3.0],)
        @test bind(m, @varname(a.q.x[:]) => [3.0, 4.0])() == (x=[3.0, 4.0],)
        @test_throws ArgumentError bind(m, @varname(a.q.x) => ones(3))
        m = bind(
            deferred_arrays_parent(),
            VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=[1.0, 2.0],)),)),)),
        )
        @test bind(m, @varname(a.b.x[2]) => 3.0)() == [1.0, 3.0]
        @test bind(m, @varname(a.b.x[:]) => [3.0, 4.0])() == [3.0, 4.0]
        @test_throws ArgumentError bind(m, @varname(a.b.x) => ones(3))
    end
end

@model listed_namespace_leaves() = (
    x ~ Normal(); y ~ Normal(); z ~ Normal(); (x=x, y=y, z=z)
)
@model listed_namespace_middle() = b ~ to_submodel(listed_namespace_leaves())
@model listed_namespace_parent() = a ~ to_submodel(listed_namespace_middle())
@testset "a listing can be rebound alongside new entries" begin
    for (bind, listing) in ((condition, conditioned), (fix, fixed))
        m = bind(
            bind(
                listed_namespace_parent(),
                VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
            );
            a=(b=(y=2.0,),),
        )
        @test bind(listed_namespace_parent(), listing(m), @varname(a.b.z) => 3.0)() ==
            (x=1.0, y=2.0, z=3.0)
        @test_throws ArgumentError bind(
            listed_namespace_parent(), listing(m), @varname(a.b.x) => 3.0
        )
    end
end

@model owned_mid() = b ~ to_submodel(deferred_leaves())
@model owned_unprefixed() = r ~ to_submodel(condition(owned_mid(), @varname(b.x) => 5.0))
@model owned_outer() = a ~ to_submodel(owned_unprefixed())
@model owned_deep_mid() = b ~ to_submodel(condition(deferred_leaves(); x=5.0))
@model owned_deep_unprefixed() = r ~ to_submodel(owned_deep_mid())
@model owned_deep_outer() = a ~ to_submodel(owned_deep_unprefixed())
@testset "deferred bindings preserve child ownership" begin
    for bind in (condition, fix)
        late_bound = bind(owned_unprefixed(); r=(b=(x=1.0,),))
        @test_throws r"already bound" late_bound()
        late_bound = bind(owned_outer(); a=(r=(b=(x=1.0,),),))
        @test_throws r"already bound" late_bound()
        late_bound = bind(owned_deep_outer(); a=(r=(b=(x=1.0,),),))
        @test_throws r"already bound" late_bound()
        @test bind(owned_outer(); a=(r=(b=(y=1.0,),),))() == (x=5.0, y=1.0)
    end
end

@model listing_mid() = b ~ to_submodel(deferred_leaves())
@model listing_outer() = a ~ to_submodel(listing_mid())
@testset "listings merge successive child namespace bindings" begin
    for (bind, remove, list) in ((condition, decondition, conditioned), (fix, unfix, fixed))
        m = bind(
            bind(
                listing_outer(),
                VarNamedTuple((a=VarNamedTuple((b=VarNamedTuple((x=1.0,)),)),)),
            );
            a=(b=(y=2.0,),),
        )
        L = list(m)
        @test collect(keys(L)) == [@varname(a.b.x), @varname(a.b.y)]
        @test L[@varname(a.b)] == VarNamedTuple((x=1.0, y=2.0))
        @test haskey(L, @varname(a.b.x)) && L[@varname(a.b.x)] == 1.0
        @test values(L) == [1.0, 2.0]
        @test collect(pairs(L)) == [@varname(a.b.x) => 1.0, @varname(a.b.y) => 2.0]
        @test Dict(pairs(L)) == Dict(@varname(a.b.x) => 1.0, @varname(a.b.y) => 2.0)
        @test merge(VarNamedTuple((z=3.0,)), L)[@varname(a.b)] ==
            VarNamedTuple((x=1.0, y=2.0))
        @test !occursin("Deferred", repr(L)) && !occursin("pending", repr(L))
        @test bind(listing_outer(), L)() == (x=1.0, y=2.0)
        @test list(bind(m; a=(b=(x=3.0,),)))[@varname(a.b)] == VarNamedTuple((x=3.0, y=2.0))
        removed = list(remove(m, @varname(a.b.x)))
        @test collect(pairs(removed)) == [@varname(a.b.y) => 2.0]
        @test !haskey(removed, @varname(a.b.x))
    end
end

@model argument_mid() = b ~ to_submodel(deferred_leaves())
@model argument_body() = a ~ to_submodel(argument_mid())
@model argument_siblings(m) = (a ~ to_submodel(deferred_structure()); c ~ to_submodel(m); a)
@testset "argument child listings" begin
    for (bind, list) in ((condition, conditioned), (fix, fixed))
        @test collect(keys(list(bind(nested_outer(argument_mid()); a=(b=(x=1.0,),))))) ==
            [@varname(a.b.x)]
        @test collect(keys(list(bind(argument_body(); a=(b=(x=1.0,),))))) == [@varname(a.b)]
        @test collect(
            keys(list(bind(argument_siblings(argument_mid()); a=(b=(x=1.0,),))))
        ) == [@varname(a.b)]
    end
end

@testset "argument child prefix scope" begin
    for (bind, listing) in ((condition, conditioned), (fix, fixed))
        model = bind(
            nested_outer(prefix(argument_mid(), @varname(q.r))); a=(q=(r=(b=(x=1.0,),),),)
        )
        @test collect(pairs(listing(model))) == [@varname(a.q.r.b.x) => 1.0]
        parent = nested_outer(prefix(deferred_structure(), :b))
        model = bind(parent; a=(b=(b=(x=1.0, y=2.0),),))
        @test collect(pairs(listing(model))) == [@varname(a.b.b) => (x=1.0, y=2.0)]
        @test bind(parent, listing(model))() == (x=1.0, y=2.0)
        parent = nested_outer(prefix(argument_mid(), @varname(q[2])))
        model = bind(
            parent, @varname(a.q[2].b) => (x=1.0, y=2.0), @of(a = @of(q = of(Array, 2)))
        )
        @test collect(pairs(listing(model))) ==
            [@varname(a.q[2].b.x) => 1.0, @varname(a.q[2].b.y) => 2.0]
        @test bind(parent, listing(model))() == (x=1.0, y=2.0)
    end
end

@model function indexed_argument_child(child)
    a = Vector{Any}(undef, 2)
    for i in 1:2
        a[i] ~ to_submodel(child)
    end
    return a
end
@testset "indexed argument child listings" begin
    for (bind, listing) in ((condition, conditioned), (fix, fixed))
        model = bind(
            indexed_argument_child(deferred_leaves()),
            @varname(a[1].x) => 1.0,
            @of(a = of(Array, 2))
        )
        @test listing(model)[@varname(a[1].x)] == 1.0
    end
end

function removal_geometry_model(bind, remove, n)
    storage(v) = [fill(v, 2 + i % 2) for i in 1:n]
    child = bind(removal_geometry_child(n); x=storage(0.0))
    middle = bind(removal_geometry_middle(remove(child)), @varname(b.x) => storage(1.0))
    middle = bind(middle; z=1.0)
    middle = remove(middle, @varname(b.x[1][1]))
    return bind(
        argument_geometry_parent(remove(middle, @varname(b.x))),
        @varname(a.b.x) => storage(2.0),
    )
end

@testset "submodels.jl" begin
    @testset "unprefixed children are rejected" begin
        @model function parent(flag)
            return a ~ to_submodel(namespace_leaf(), flag)
        end
        @test_throws ArgumentError to_submodel(namespace_leaf(), false)
        @test_throws ArgumentError parent(false)()
        @test typeof(to_submodel(namespace_leaf(), true)) ===
            typeof(to_submodel(namespace_leaf()))
        @test Set(keys(rand(parent(true)))) == Set([@varname(a.x)])
    end

    @testset "explicit bindings of submodel return values explain recovery" begin
        @model parent(y) = (a ~ to_submodel(scalar_z()); y ~ Normal(a))
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
            return a ~ to_submodel(condition(scalar_likelihood(missing, mu); y=y[2]))
        end
        @model function renamed(z)
            mu ~ Normal()
            z[1] ~ Normal(mu)
            return a ~ to_submodel(condition(scalar_likelihood(missing, mu); y=z[2]))
        end
        expected = sum(logpdf.(Normal(), [0.0, 1.0, 2.0]))
        @test logjoint(overlap([1.0, 2.0]), (; mu=0.0)) ≈ expected
        @test logjoint(renamed([1.0, 2.0]), (; mu=0.0)) ≈ expected
        for bind in (condition, fix)
            model = bind(
                decondition(renamed([1.0, 2.0]), @varname(a.y)), @varname(a.y) => 5.0
            )
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
        @model function inspect_outer(m)
            b = Vector{Any}(undef, 2)
            b[2] ~ to_submodel(m)
            return b[2]
        end
        child = condition(inspect_child(); x=[0.0, 1.0])
        parent = condition(
            decondition(nested_outer(child), @varname(a.x[1])),
            @varname(a.x[1]) => 2.0,
            @of(a = @of(x = of(Array, 2))),
        )
        for (model, namespace) in
            ((parent, @varname(a)), (inspect_outer(parent), @varname(b[2].a)))
            local_model = model(Xoshiro(1))
            expected = conditioned(
                prefix(condition(child, @varname(x[1]) => 2.0), namespace)
            )
            function bindings(values, prefix=nothing)
                return DynamicPPL._fold_model_indices(
                    Dict{VarName,Any}(), values
                ) do result, value, optic, _
                    vn = if prefix === nothing
                        AbstractPPL.optic_to_varname(optic)
                    else
                        AbstractPPL.append_optic(prefix, optic)
                    end
                    if value isa Union{VarNamedTuple,DynamicPPL.VarNamedTuples.PartialArray}
                        merge!(result, bindings(value, vn))
                    else
                        result[vn] = value
                    end
                    result
                end
            end
            @test bindings(conditioned(local_model)) == bindings(expected)
            @test isempty(fixed(local_model))
            vn = AbstractPPL.prefix(@varname(x[2]), namespace)
            for (bind, remove, accessor) in
                ((condition, decondition, conditioned), (fix, unfix, fixed))
                @test_throws "outside this model's prefix" bind(
                    local_model, @varname(x[2]) => 3.0
                )
                edited = bind(
                    bind === fix ? decondition(local_model, vn) : local_model, vn => 3.0
                )
                @test accessor(edited)[vn] == 3.0
                @test accessor(edited(Xoshiro(1)))[vn] == 3.0
                removed = remove(edited, vn)
                @test !haskey(accessor(removed(Xoshiro(1))), vn)
                @test isempty(accessor(remove(edited)))
                values = rand(Xoshiro(1), removed)
                @test haskey(values, vn)
                @test_throws "outside this model's prefix" bind(local_model; metadata=1)
                @test_throws "Argument `$(AbstractPPL.prefix(@varname(metadata), namespace))`" bind(
                    local_model, AbstractPPL.prefix(@varname(metadata), namespace) => 1
                )
            end
        end
    end

    @testset "submodel tildes reject model arguments" begin
        @model argument_return(x=0.0) = x ~ to_submodel(scalar_z())
        @model indexed_return(x) = x[1] ~ to_submodel(scalar_z())
        @model property_return(x) = x.a ~ to_submodel(scalar_z())
        @model keyword_return(; x=0.0) = x ~ to_submodel(scalar_z())
        @model runtime_return(x, rhs) = x ~ rhs
        @model unprefixed_return(x) = x ~ to_submodel(scalar_z(), true)
        @model splatted_return(x...) = x[1] ~ to_submodel(scalar_z())
        @model keyword_splatted_return(; x...) = x.a ~ to_submodel(scalar_z())
        @model local_return() = x ~ to_submodel(scalar_z())
        err = r"ArgumentError: Submodel tilde .*model argument `x`.*local LHS.*condition"
        for (model, address, value) in (
            (argument_return(), @varname(x), 2.0),
            (argument_return(missing), @varname(x), 2.0),
            (argument_return(nothing), @varname(x), 2.0),
            (indexed_return([0.0]), @varname(x[1]), 2.0),
            (property_return((a=0.0,)), @varname(x.a), 2.0),
            (keyword_return(), @varname(x), 2.0),
            (runtime_return(0.0, to_submodel(scalar_z())), @varname(x), 2.0),
            (unprefixed_return(0.0), @varname(x), 2.0),
            (splatted_return(0.0), @varname(x), (2.0,)),
            (keyword_splatted_return(; a=0.0), @varname(x), (a=2.0,)),
        )
            for wrapped in (
                model,
                prefix(model, @varname(p)),
                outer_model(condition(model, conditioned(model))),
            )
                @test_throws err wrapped(Xoshiro(1))
                @test_throws err decondition(wrapped)(Xoshiro(1))
                @test outer_model(local_return())(Xoshiro(1)) isa Real
            end
            for bind in (condition, fix)
                bound = bind(bind === fix ? decondition(model) : model, address => value)
                @test_throws err bound(Xoshiro(1))
                local_model = bind(local_return(), @varname(x.z) => 2.0)
                @test local_model(Xoshiro(1)) == 2.0
            end
        end
        # A runtime RHS is classified when the tilde executes; ordinary assignment keeps Julia semantics.
        @test runtime_return(2.0, Normal())(Xoshiro(1)) == 2.0
        @model function assigned_return(x)
            a ~ to_submodel(fix(scalar_z(); z=2.0))
            x = a
            return x ~ Normal()
        end
        @test loglikelihood(assigned_return(0.0), VarNamedTuple()) == logpdf(Normal(), 2.0)
    end

    @testset "submodel return bindings with manual prefixes" begin
        @model manual(a, m) = a ~ to_submodel(m)
        @model nested(m) = outer ~ to_submodel(m)
        for child_model in (namespace_leaf(), prefix(namespace_leaf(), @varname(b)))
            for m in (
                condition(decondition(manual(0.0, child_model)); a=3.0),
                fix(decondition(manual(0.0, child_model)); a=3.0),
            )
                for model in (m, nested(m))
                    @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" model(
                        Xoshiro(1)
                    )
                    @test nested(nested_outer(child_model))(Xoshiro(1)) isa Real
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
        for indices in ((1,), Tuple(1:100))
            bindings = Tuple(@varname(x[i].y) => 0.0 for i in indices)
            for templates in ((), (@of(x = of(Array, 100)),))
                model = if isempty(templates)
                    values = VarNamedTuple()
                    for (vn, value) in bindings
                        values = DynamicPPL.templated_setindex!!(values, value, vn, zeros(100))
                    end
                    condition(indexed_observations(zeros(100), indices), values)
                else
                    condition(
                        indexed_observations(zeros(100), indices),
                        bindings...,
                        templates...,
                    )
                end
                @test @inferred(init!!(model, accs, strategy, UnlinkAll())) isa Tuple
                @test (@allocated init!!(model, accs, strategy, UnlinkAll())) <=
                    baseline + 128
            end
        end
    end

    @testset "recursive removal at one indexed child stays concrete" begin
        unbound = indexed_observations(zeros(2))
        accs = VarInfo(LogPriorAccumulator(), LogLikelihoodAccumulator())
        params = VarNamedTuple((@varname(mu) => 0.1, @varname(x[1].y) => 0.0))
        strategy = InitFromParams(params, nothing)
        init!!(unbound, accs, strategy, UnlinkAll())
        baseline = @allocated init!!(unbound, accs, strategy, UnlinkAll())
        cleared = decondition(unbound, @varname(x[1].y))
        for model in
            (cleared, condition(cleared, @varname(x[1].y) => 0.0, @of(x = of(Array, 2))))
            @test @inferred(init!!(model, accs, strategy, UnlinkAll())) isa Tuple
            @test (@allocated init!!(model, accs, strategy, UnlinkAll())) <= baseline + 512
        end
    end

    @testset "inherited geometry does not allocate during evaluation" begin
        rng = Xoshiro(1)
        for bind in (condition, fix)
            reached, reference = replaced_geometry_model(bind)
            GC.@preserve reached begin
                GC.gc()
                @test reference.value === nothing
            end
        end
        for parent in (argument_geometry_parent, child -> held_geometry_parent(Ref(child)))
            deltas = map((1, 1000)) do n
                storage = [ones(2) for _ in 1:n]
                reference = parent(condition(nested_geometry_child(n); x=storage))
                child = condition(nested_geometry_child(n); x=[zeros(2) for _ in 1:n])
                inherited = condition(parent(decondition(child)), @varname(a.x) => storage)
                @test @inferred(reference(rng)) == @inferred(inherited(rng)) == 1.0
                baseline = minimum(@allocated(reference(rng)) for _ in 1:5)
                minimum(@allocated(inherited(rng)) for _ in 1:5) - baseline
            end
            @test deltas[2] <= deltas[1] + 128
        end
    end

    @testset "recursive removal selection does not rebuild geometry" begin
        rng = Xoshiro(1)
        for (bind, remove, role) in
            ((condition, decondition, DynamicPPL.Condition), (fix, unfix, DynamicPPL.Fix))
            small, large = map(n -> removal_geometry_model(bind, remove, n)(rng), (2, 1000))
            for model in (small, large)
                selected = DynamicPPL._submodel_removals(role, model, @varname(b))
                @test map(marker -> marker.name, selected) ==
                    (@varname(x[1][1]), @varname(x))
            end
            baseline = @allocated DynamicPPL._submodel_removals(role, small, @varname(b))
            @test (@allocated DynamicPPL._submodel_removals(role, large, @varname(b))) <=
                baseline
        end
    end

    @testset "indexed namespace template evaluation" begin
        rng = Xoshiro(1)
        # Compare with equivalent concrete storage: its allocation cost differs
        # between supported Julia versions, independently of binding templates.
        single_bindings = DynamicPPL.templated_setindex!!(
            VarNamedTuple(), 3.0, @varname(p[1].x), zeros(1, 1)
        )
        single = condition(single_namespace_parent(), single_bindings)
        templated = condition(
            single_namespace_parent(), @varname(p[1].x) => 3.0, @of(p = of(Array, 1, 1))
        )
        @test single(rng) == templated(rng) == 3.0
        baseline = @allocated single(rng)
        @test (@allocated templated(rng)) <= baseline
        # Only 1.12 meets 48 B: Julia 1.13's bounds-checked fill! retains the zeros matrix.
        if v"1.12" <= VERSION < v"1.13" && Base.JLOptions().check_bounds != 1
            @test (@allocated templated(rng)) <= 48
        end
        @test @inferred(templated(rng)) == 3.0
        mixed_bindings = DynamicPPL.templated_setindex!!(
            VarNamedTuple(), 3.0, @varname(p[2, 2].x), zeros(2, 2)
        )
        cleared = mixed_namespace_parent()
        reference = condition(cleared, mixed_bindings)
        reference(rng)
        mixed = condition(cleared, @varname(p[2, 2].x) => 3.0, @of(p = of(Array, 2, 2)))
        @test @inferred(mixed(rng)) == [2.0 0.0; 0.0 3.0]
        baseline = @allocated reference(rng)
        @test (@allocated mixed(rng)) <= baseline
    end

    @testset "indexed child bindings stay local" begin
        for n in (1_000, 2_000)
            model = indexed_observations(zeros(n))
            params = (; mu=0.25)
            @test logjoint(model, params) ≈
                logpdf(Normal(), params.mu) + n * logpdf(Normal(params.mu, 1), 0)
            @test (@allocated logjoint(model, params)) < 64n
        end

        model = decondition(
            nested_observations(zeros(2)), @varname(a.x[1].y), @varname(a.x[2].y)
        )
        model = condition(model, @varname(a.x[1].y) => 2.0, @of(a = @of(x = of(Array, 2))))
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

    @testset "binding construction uses prefix templates" begin
        @model template_covariate(z) = x ~ Normal(z[1])
        for (name, template) in
            ((@varname(p[2]), zeros(2)), (@varname(p[0]), OffsetArray(zeros(2), 0:1))),
            bind in (condition, fix),
            leaf in (namespace_leaf(), template_covariate(zeros(2)))

            original = prefix(leaf, name; template)
            vn = AbstractPPL.prefix(@varname(x), name)
            bound = bind(original, vn => 4.0)
            @test bound(Xoshiro(1)) == 4.0
            @test template_parent(bound)(Xoshiro(1)) == 4.0
            @test loglikelihood(bound, VarNamedTuple()) ≈
                (bind === condition ? logpdf(Normal(), 4.0) : 0.0)
        end
    end

    @testset "property and integer prefixes resolve explicit bindings" begin
        @model slice_parent(child) = unused ~ to_submodel(child)
        for name in (@varname(a), @varname(a[1])),
            (bind, accessor) in ((condition, conditioned), (fix, fixed))

            child = prefix(bind(namespace_leaf(); x=3.0), name; template=zeros(2))
            bindings = accessor(child)
            @test keys(bindings) == [AbstractPPL.prefix(@varname(x), name)]
            @test bindings[AbstractPPL.prefix(@varname(x), name)] == 3.0
            @test isempty(rand(Xoshiro(1), child))
            @test child(Xoshiro(1)) == 3.0
            @test slice_parent(child)(Xoshiro(1)) == 3.0
            parent = bind(
                slice_parent(prefix(namespace_leaf(), name; template=zeros(2))),
                VarNamedTuple((
                    unused=DynamicPPL.templated_setindex!!(
                        VarNamedTuple(), (; x=4.0), name, zeros(2)
                    ),
                )),
            )
            @test parent(Xoshiro(1)) == 4.0
            @test template_parent(parent)(Xoshiro(1)) == 4.0
        end
    end

    @testset "dynamic prefixes with stored observations" begin
        @model function dynamic_parent(child)
            a = zeros(2, 3)
            a[begin] ~ to_submodel(child)
            a[end, end] ~ to_submodel(child)
            return a
        end
        @model nested_parent(child) = b ~ to_submodel(dynamic_parent(child))
        @model explicit_child(child) = unused ~ to_submodel(prefix(child, @varname(inner)))
        for op in (condition, fix), parent in (dynamic_parent, nested_parent)
            child = op(op === fix ? decondition(observed_child()) : observed_child(); x=2.0)
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
            # Mixed-index prefixes need storage for their shared array symbol.
            remove = op === condition ? decondition : unfix
            cleared = remove(model, vn)
            @test_throws r"indexed edits require concrete storage" condition(
                cleared, vn => 3.0
            )
            bindings = DynamicPPL.templated_setindex!!(
                VarNamedTuple(), (; x=3.0), @varname(a[2, 3]), zeros(2, 3)
            )
            bindings = parent === dynamic_parent ? bindings : VarNamedTuple(; b=bindings)
            rebound = condition(cleared, bindings)
            @test_throws r"recursive removal.*index count" rebound()

            model = parent(explicit_child(child))
            vn = if parent === dynamic_parent
                @varname(a[1].unused.inner.x)
            else
                @varname(b.a[1].unused.inner.x)
            end
            remove = op === condition ? decondition : unfix
            cleared = remove(model, vn)
            @test_throws r"indexed edits require concrete storage" condition(
                cleared, vn => 3.0
            )
            bindings = DynamicPPL.templated_setindex!!(
                VarNamedTuple(), (; unused=(inner=(x=3.0,),)), @varname(a[1]), zeros(2, 3)
            )
            bindings = parent === dynamic_parent ? bindings : VarNamedTuple(; b=bindings)
            rebound = condition(cleared, bindings)
            @test_throws r"recursive removal.*index count" rebound()
        end
    end

    @testset "parent array templates" begin
        @model function matrix_template(storage)
            a = copy(storage)
            a[1] ~ to_submodel(namespace_leaf())
            a[2, 2] ~ to_submodel(namespace_leaf())
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
            unused ~ to_submodel(prefix(child_matrix(), @varname(inner)))
        @model function outer_matrix()
            a = zeros(2, 2)
            a[1] ~ to_submodel(middle_matrix())
            a[2, 2] ~ to_submodel(middle_matrix())
            return a
        end
        _, vi = init!!(Xoshiro(1), outer_matrix(), vi, InitFromPrior(), UnlinkAll())
        a = get_raw_values(vi).data.a
        @test size(a.data) == (2, 2)
        @test size(a.data[1].data.unused.data.inner.data.x.data) == (2, 3)
        @test size(a.data[2, 2].data.unused.data.inner.data.x.data) == (2, 3)
    end

    @testset "$op with AbstractPPL API" for op in [condition, fix]
        x_val = 1.0
        x_logp = op == condition ? logpdf(Normal(), x_val) : 0.0

        @testset "Auto prefix" begin
            @model function outer()
                return a ~ to_submodel(normal_pair())
            end
            inner_op = op(normal_pair(), (@varname(x) => x_val))
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
            @model function outer()
                return a ~ to_submodel(normal_pair(), true)
            end
            @model function outer2()
                return a ~ to_submodel(inner_op, true)
            end
            with_inner_op = outer2()
            inner_op = op(normal_pair(), (@varname(x) => x_val))
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
            @model function outer()
                return a ~ to_submodel(prefix(normal_pair(), :b))
            end
            inner_op = op(normal_pair(), (@varname(x) => x_val))
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

        @model latent_g() = a ~ to_submodel(decondition(f(0.0)))
        vnt = rand(Xoshiro(1), latent_g())
        @test Set(keys(vnt)) == Set([@varname(a.x), @varname(a.y)])

        @model function parent_with_return_value(a)
            a[1] ~ to_submodel(condition(observed_child(missing); x=2.0))
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
            remove = first_bind === condition ? decondition : unfix
            expanded = bind(
                first_bind === bind ? original : remove(original, @varname(a.x)),
                @varname(a.x) => 1.0,
            )
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
            @model function outer()
                return a ~ to_submodel(namespace_leaf())
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

@testset "recursive removals" begin
    @model observation(counts, mu) = counts ~ Normal(mu)
    @model function generator()
        mu ~ Normal()
        y ~ to_submodel(observation(missing, mu))
        return (mu, y)
    end
    latent = decondition(generator(), @varname(y.counts))
    @test keys(rand(Xoshiro(1), latent)) == [@varname(mu), @varname(y.counts)]
    @test returned(
        latent, VarNamedTuple((@varname(mu) => 1.0, @varname(y.counts) => 2.0))
    ) == (1.0, 2.0)
    fixed_child = nested_outer(fix(decondition(observed_child()); x=5.0))
    @test returned(unfix(fixed_child, @varname(a.x)), (a=(x=2.0,),)) == 2.0
    @test keys(rand(Xoshiro(1), unfix(fixed_child, @varname(a.x)))) == [@varname(a.x)]
    @test decondition(fixed_child, @varname(a.x))() == 5.0
    @test returned(unfix(fixed_child), (a=(x=2.0,),)) == 2.0
    @test isempty(
        conditioned(decondition(nested_outer(condition(observed_child(missing); x=2.0))))
    )
    @test keys(
        rand(
            Xoshiro(1), decondition(nested_outer(condition(observed_child(missing); x=2.0)))
        ),
    ) == [@varname(a.x)]
    removed = decondition(
        nested_outer(condition(observed_child(missing); x=2.0)), @varname(a.x)
    )
    @test keys(rand(Xoshiro(1), decondition(removed, @varname(a.x)))) == [@varname(a.x)]
    @test condition(removed, @varname(a.x) => 3.0)() == 3.0
    @test condition(outer_model(removed), @varname(b.a.x) => 4.0)() == 4.0
    @test keys(rand(Xoshiro(1), prefix(removed, @varname(p)))) == [@varname(p.a.x)]
    @test keys(rand(Xoshiro(1), outer_model(removed))) == [@varname(b.a.x)]
    @test unfix(observed_child(), @varname(x))(Xoshiro(1)) == 2.0
    @test unfix(nested_outer(condition(observed_child(missing); x=2.0)), @varname(a.x))(
        Xoshiro(1)
    ) == 2.0
    @test keys(
        rand(
            Xoshiro(1),
            decondition(nested_outer(decondition(observed_child())), @varname(a.x)),
        ),
    ) == [@varname(a.x)]
    @test fixed(unfix(fixed_child)) == VarNamedTuple()
    @model function runtime()
        mu ~ Normal()
        a ~ to_submodel(condition(observed_child(); x=2mu))
        return a
    end
    @test keys(rand(Xoshiro(2), decondition(runtime(), @varname(a.x)))) ==
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
    @model function removal_parent()
        m ~ Normal()
        a ~ to_submodel(namespace_leaf())
        return m
    end
    @model function removal_fields()
        m = (x=zeros(1),)
        m.x[1] ~ Normal()
        a ~ to_submodel(namespace_leaf())
        return m
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        for (model, name, templates) in (
            (removal_parent(), @varname(m), ()),
            (removal_fields(), @varname(m.x[1]), (@of(m = @of(x = of(Array, 1))),)),
        )
            @test name in keys(rand(Xoshiro(1), remove(model, name)))
            prefixed_name = DynamicPPL.maybe_prefix(name, @varname(p))
            @test prefixed_name in
                keys(rand(Xoshiro(1), remove(prefix(model, @varname(p)), prefixed_name)))
            deferred = remove(
                removal_wrapper(model), DynamicPPL.maybe_prefix(name, @varname(b))
            )
            @test DynamicPPL.maybe_prefix(name, @varname(b)) in
                keys(rand(Xoshiro(1), deferred))
            if !isempty(templates)
                @test_throws "without storage" bind(model, name => 2.0)
            end
            matched = remove(bind(model, name => 2.0, templates...), name)
            @test name in keys(rand(Xoshiro(1), matched))
        end
    end
end

@testset "recursive removals overlap whole LHS variables" begin
    @model function removal_whole(t, run=true)
        run && (t ~ product_distribution((a=Normal(),)))
        return nothing
    end
    @model removal_whole_local() = t ~ product_distribution((a=Normal(),))
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = decondition(removal_whole((a=2.0,)))
        for (model, name, whole_name) in (
            (child, @varname(t.a), @varname(t)),
            (removal_whole_local(), @varname(t.a), @varname(t)),
            (prefix(child, @varname(p)), @varname(p.t.a), @varname(p.t)),
            (removal_runtime(removal_runtime(child)), @varname(a.a.t.a), @varname(a.a.t)),
        )
            @test whole_name in keys(rand(Xoshiro(1), remove(model, name)))
            matched = remove(bind(model, whole_name => (a=2.0,)), name)
            @test whole_name in keys(rand(Xoshiro(1), matched))
        end
        skipped = decondition(removal_whole((a=2.0,), false))
        for (model, name, whole_name) in (
            (skipped, @varname(t.a), @varname(t)),
            (removal_runtime(skipped), @varname(a.t.a), @varname(a.t)),
        )
            @test remove(model, name)(Xoshiro(1)) === nothing
            @test remove(bind(model, whole_name => (a=2.0,)), name)(Xoshiro(1)) === nothing
        end
        # A child's removal cannot clear an enclosing binding.
        removed_child = remove(removal_runtime(child), @varname(a.t.a))
        enclosing = bind(removal_runtime(removed_child), @varname(a.a.t) => (a=2.0,))
        @test enclosing(Xoshiro(1)) === nothing
        @test isempty(keys(rand(Xoshiro(1), enclosing)))
        @test logjoint(enclosing, (;)) ≈ (bind === condition ? logpdf(Normal(), 2.0) : 0.0)
    end
end

@testset "argument submodel tildes reject sibling edits only when reached" begin
    @model function removal_siblings(a, run)
        run && (a.child ~ to_submodel(condition(observed_child(missing); x=2.0)))
        a.obs ~ Normal()
        return a
    end
    for (bind, remove) in ((condition, decondition), (fix, unfix)),
        wrap in (identity, m -> prefix(m, @varname(p)), removal_wrapper)

        reached = condition(
            removal_siblings((child=0.0, obs=1.0), true); a=(child=0.0, obs=1.0)
        )
        bind === fix && (reached = decondition(reached, @varname(a.obs)))
        reached = remove(bind(reached, @varname(a.obs) => 3.0), @varname(a.obs))
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" wrap(
            reached
        )(
            Xoshiro(1)
        )
        skipped = condition(
            removal_siblings((child=0.0, obs=1.0), false); a=(child=0.0, obs=1.0)
        )
        bind === fix && (skipped = decondition(skipped, @varname(a.obs)))
        skipped = remove(bind(skipped, @varname(a.obs) => 3.0), @varname(a.obs))
        @test wrap(skipped)(Xoshiro(1)).obs isa Real
    end
end

@testset "recursive removals through computed and branched submodels" begin
    @model removal_child() = t ~ Normal()
    @model function removal_mixed(child, c)
        if c
            a ~ Normal()
        else
            a ~ to_submodel(child)
        end
        return a
    end
    my_sub(child) = to_submodel(child)
    @model removal_call(child) = a ~ my_sub(child)
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = bind(removal_child(); t=2.0)
        for parent in (m -> removal_mixed(m, false), removal_runtime, removal_call)
            model = parent(child)
            @test model(Xoshiro(1)) == 2.0
            removed = remove(model, @varname(a.t))
            @test keys(rand(Xoshiro(1), removed)) == [@varname(a.t)]
            @test logjoint(removed, (; a=(; t=0.5))) ≈ logpdf(Normal(), 0.5)
            @test keys(rand(Xoshiro(1), remove(removed, @varname(a.t)))) == [@varname(a.t)]
        end
        @test keys(rand(Xoshiro(1), remove(removal_mixed(child, true), @varname(a.t)))) ==
            [@varname(a)]
    end
end

@testset "matched recursive removal at evaluation" begin
    m = decondition(
        condition(
            nested_outer(condition(depth_leaf(missing); x=2.0)), @varname(a.x) => 3.0
        ),
        @varname(a.x)
    )
    @test logjoint(m, (; a=(; x=0.5))) ≈ logpdf(Normal(), 0.5)
end

@testset "recursive removal storage and scope" begin
    @test_throws r"x.*decondition.*unfix" condition(indexed(zeros(2)); x=ones(3))
    child = condition(indexed(zeros(3)); x=ones(3))
    @test length(decondition(nested_outer(child), @varname(a.x))(Xoshiro(1))) == 3
    @test returned(
        unfix(nested_outer(fix(decondition(indexed(zeros(3))); x=ones(3))), @varname(a.x)),
        (a=(x=zeros(3),),),
    ) == zeros(3)
    partial = decondition(nested_outer(child), @varname(a.x[2]))
    @test returned(partial, VarNamedTuple((@varname(a.x[2]) => 5.0,))) == [1.0, 5.0, 1.0]
    @test condition(decondition(nested_outer(child)), @varname(a.x) => [4.0, 5.0])() ==
        [4.0, 5.0]
    @test condition(decondition(nested_outer(child)), @varname(a.x) => [4.0, 5.0, 6.0])() ==
        [4.0, 5.0, 6.0]
    @test returned(
        condition(
            decondition(nested_outer(condition(indexed(zeros(2)); x=zeros(2)))),
            @varname(a.x[1]) => 4.0,
            @of(a = @of(x = of(Array, 2)))
        ),
        VarNamedTuple((@varname(a.x[2]) => 5.0,)),
    ) == [4.0, 5.0]
    @model function shared(x)
        x ~ Normal()
        a ~ to_submodel(condition(depth_leaf(missing); x=2.0))
        return (x, a)
    end
    both = decondition(shared(1.0), @varname(x))
    @test both(Xoshiro(1)) == (rand(Xoshiro(1), Normal()), 2.0)
    @test condition(both; x=3.0)(Xoshiro(1)) == (3.0, 2.0)
    @test isempty(conditioned(decondition(decondition(nested_outer(child)))))
    once = decondition(nested_outer(child), @varname(a.x))
    twice = decondition(once, @varname(a.x))
    @test typeof(twice) == typeof(once)
    @test_logs DynamicPPL.DebugUtils.check_model(Xoshiro(1), twice)
    @model branch(run) = (
        run && (a ~ to_submodel(condition(depth_leaf(missing); x=1.0))); nothing
    )
    @test decondition(branch(false), @varname(a.x))() === nothing
    @model function repeated()
        for i in 1:2
            unused ~ to_submodel(
                if i == 1
                    condition(depth_leaf(missing); x=1.0)
                else
                    decondition(depth_leaf(1.0))
                end,
            )
        end
    end
    @test decondition(repeated(), @varname(unused.x))(Xoshiro(1)) === nothing
end

@testset "recursive removal namespace boundaries" begin
    @model parent(a) = a ~ to_submodel(depth_leaf(missing))
    @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" decondition(
        parent(missing), @varname(a.x)
    )(
        Xoshiro(1)
    )
    @model local_parent() = a ~ to_submodel(decondition(depth_leaf(missing)))
    @test local_parent()(Xoshiro(1)) isa Real
    @model function sharearr(x)
        x[1] ~ Normal()
        x[2] ~ Normal()
        a ~ to_submodel(decondition(arr([2.0, 2.0])))
        return (x, a)
    end
    @model function arr(x)
        x[1] ~ Normal()
        x[2] ~ Normal()
        return x
    end
    m = condition(sharearr([0.0, 0.0]); x=[4.0, 4.0], a=(x=[4.0, 4.0],))
    @test condition(m, @varname(x[1]) => 3.0)() == ([3.0, 4.0], [4.0, 4.0])
    @model ownprefix() = a ~ to_submodel(condition(depth_leaf(missing); x=1.0))
    @test_throws ArgumentError decondition(prefix(ownprefix(), @varname(p)), @varname(q.x))
end

@testset "recursive unfix releases whole shape owners" begin
    @model function owner_grows(x)
        x = vcat(x, 0.0)
        for i in eachindex(x)
            x[i] ~ Normal()
        end
        return x
    end
    old = fix(decondition(owner_grows(zeros(2))); x=ones(2))
    rebound = fix(unfix(old), @varname(x[1]) => 2.0)
    @test_throws r"storage axes" rebound(Xoshiro(1))
    for names in ((), (@varname(a.x),))
        model = fix(
            unfix(nested_outer(old), names...),
            @varname(a.x[1]) => 2.0,
            @of(a = @of(x = of(Array, 2)))
        )
        @test_throws r"storage axes" model(Xoshiro(1))
    end
end

@testset "recursive removal resolves indexed prefixes" begin
    @test keys(rand(Xoshiro(1), decondition(indexed([1.0, 2.0]), @varname(x[end])))) ==
        [@varname(x[2])]
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = prefix(
            bind(bind === fix ? decondition(observed_child()) : observed_child(); x=3.0),
            @varname(p[end]);
            template=zeros(2),
        )
        removed = remove(child, @varname(p[end].x))
        @test isempty(bind === condition ? conditioned(removed) : fixed(removed))
        @test remove(removed, @varname(p[2].x))(Xoshiro(1)) == removed(Xoshiro(1))
        @test keys(rand(Xoshiro(1), removed)) == [@varname(p[2].x)]
        @test returned(removed, VarNamedTuple((@varname(p[2].x) => 2.0,))) == 2.0
    end
    model = prefix(
        nested_outer(condition(observed_child(missing); x=2.0)),
        @varname(p[end]);
        template=zeros(2),
    )
    removed = decondition(model, @varname(p[end].a.x))
    @test keys(rand(Xoshiro(1), removed)) == [@varname(p[2].a.x)]
end

@testset "recursive removal classifies the executed RHS" begin
    @model function removal_branch(a, observed)
        if observed
            a ~ Normal()
        else
            a ~ to_submodel(depth_leaf(missing))
        end
        return a
    end
    @model function removal_dynamic(a)
        child = to_submodel(depth_leaf(missing))
        a ~ child
        return a
    end
    m = decondition(removal_branch(1.0, true), @varname(a))
    @test keys(rand(Xoshiro(1), m)) == [@varname(a)]
    @test keys(
        rand(
            Xoshiro(1),
            unfix(
                decondition(fix(decondition(removal_branch(1.0, true)); a=2.0), @varname(a))
            ),
        ),
    ) == [@varname(a)]
    m = decondition(
        outer_model(condition(removal_branch(missing, true); a=1.0)), @varname(b.a)
    )
    @test keys(rand(Xoshiro(1), m)) == [@varname(b.a)]
    @test keys(
        rand(Xoshiro(1), decondition(removal_branch(missing, true), @varname(a.x)))
    ) == [@varname(a)]
    for child in (removal_dynamic(missing), removal_branch(missing, false))
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" decondition(
            child, @varname(a.x)
        )(
            Xoshiro(1)
        )
        @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" decondition(
            outer_model(child), @varname(b.a.x)
        )(
            Xoshiro(1)
        )
    end
end

@testset "recursive partial edits preserve replacement shape" begin
    @model function shaped_child(y, mu)
        for i in eachindex(y)
            y[i] ~ Normal(mu)
        end
        return y
    end
    @model function shaped_parent(n)
        mu ~ Normal()
        a ~ to_submodel(condition(shaped_child(zeros(n), mu); y=zeros(n)))
        return a
    end
    for (n, replacement) in ((3, [1.0, 2.0]), (2, [1.0, 2.0, 3.0]))
        for (bind, remove) in ((condition, decondition), (fix, unfix))
            @test_throws ArgumentError returned(
                bind(shaped_parent(n), @varname(a.y) => replacement), (mu=0.3,)
            )
            whole = bind(
                decondition(shaped_parent(length(replacement)), @varname(a.y)),
                @varname(a.y) => replacement,
            )
            edited = bind(whole, @varname(a.y[1]) => 4.0)
            expected = [4.0; replacement[2:end]]
            @test returned(edited, (mu=0.3,)) == expected
            partial = remove(whole, @varname(a.y[1]))
            @test returned(
                partial, VarNamedTuple((@varname(mu) => 0.3, @varname(a.y[1]) => 4.0))
            ) == expected
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

@model depth_parent(x, child) = (x ~ Normal(); a ~ to_submodel(child); a)
@testset "bindings replace recursive removals at their address" begin
    m = condition(
        decondition(depth_parent(2.0, condition(depth_leaf(missing); x=3.0)));
        x=7.0,
        a=(x=7.0,),
    )
    @test loglikelihood(m, (;)) ≈ 2 * logpdf(Normal(), 7.0)
    m = fix(
        unfix(decondition(depth_parent(2.0, fix(decondition(depth_leaf(3.0)); x=9.0))));
        x=7.0,
        a=(x=7.0,),
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
        m = remove(skipped_removal_parent(false), @varname(a.x))
        @test m(Xoshiro(1)) == :ok
        @test_logs (:warn, r"removal") match_mode = :any DynamicPPL.DebugUtils.check_model(
            m
        )
        m = remove(skipped_removal_parent(true), @varname(a.x))
        @test m(Xoshiro(1)) == :ok
    end
end

@testset "unused nested removal addresses" begin
    for remove in (decondition, unfix), name in (@varname(a.a.x), @varname(a.a.absent))
        m = remove(nested_outer(nested_outer(namespace_leaf())), name)
        @test keys(rand(Xoshiro(1), m)) == [@varname(a.a.x)]
        @test_logs (:warn, r"Recursive removal.*unused") DynamicPPL.DebugUtils.check_model(
            Xoshiro(1), m
        )
    end
end

@testset "partial runtime return arguments are rejected" begin
    @model runtime_return_leaf(x=3.0) = x ~ Normal()
    @model runtime_return(a, rhs) = a ~ rhs
    for arg in ([1.0, 2.0], (x=1.0, y=2.0))
        vn = arg isa NamedTuple ? @varname(a.x) : @varname(a[1])
        for edit in (
            m -> decondition(m, vn),
            m -> condition(m, vn => 4.0),
            m -> fix(decondition(m, vn), vn => 4.0),
        )
            m = edit(
                condition(
                    runtime_return(
                        arg, to_submodel(condition(runtime_return_leaf(missing); x=3.0))
                    );
                    a=arg,
                ),
            )
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" m(
                Xoshiro(1)
            )
            @test_throws r"ArgumentError: Submodel tilde .*model argument `a`.*local LHS" outer_model(
                m
            )(
                Xoshiro(1)
            )
            @test outer_model(condition(runtime_return_leaf(missing); x=3.0))(Xoshiro(1)) ==
                3.0
        end
    end
end

@testset "produced bindings round-trip through submodels" begin
    @model function prefixed_parent()
        z ~ Normal()
        a ~ to_submodel(namespace_leaf())
        return (z, a)
    end
    @model function unprefixed_parent()
        z ~ Normal()
        a ~ to_submodel(namespace_leaf(), true)
        return (z, a)
    end
    for (model, child_address) in
        ((prefixed_parent(), @varname(a.x)), (unprefixed_parent(), @varname(a.x)))
        draw = rand(Xoshiro(42), model)
        for bind in (condition, fix)
            bound = bind(model, draw)
            @test bound(Xoshiro(1)) == (draw[@varname(z)], draw[child_address])
            @test isempty(rand(Xoshiro(1), bound))
        end
    end
    for child_bind in (condition, fix), parent_bind in (condition, fix)
        child = child_bind(namespace_leaf(); x=2.0)
        @test_throws ArgumentError parent_bind(removal_wrapper(child), @varname(b.x) => 3.0)(
            Xoshiro(1)
        )
        child_remove = child_bind === condition ? decondition : unfix
        middle = parent_bind(
            child_remove(removal_wrapper(child), @varname(b.x)), @varname(b.x) => 3.0
        )
        late_bound = condition(removal_wrapper(middle), @varname(b.b.x) => 1.0)
        @test_throws ArgumentError late_bound(Xoshiro(1))
        parent_remove = parent_bind === condition ? decondition : unfix
        model = condition(
            parent_remove(removal_wrapper(middle), @varname(b.b.x)), @varname(b.b.x) => 1.0
        )
        @test model(Xoshiro(1)) == 1.0
        @test loglikelihood(model, (;)) ≈ logpdf(Normal(), 1.0)
    end
end

@testset "prefixes require scalar integer indices" begin
    @model prefix_leaf() = x ~ Dirac(3.0)
    @model prefix_nested(child) = outer ~ to_submodel(child)
    @model function indexed_return(index, auto_prefix=true)
        a = zeros(2, 2)
        a[index] ~ to_submodel(prefix_leaf(), auto_prefix)
        return a
    end
    @model vector_return() = (x ~ Dirac(3.0); [x, x])
    @model function sliced_return(index, auto_prefix=true)
        a = zeros(2)
        a[index] ~ to_submodel(vector_return(), auto_prefix)
        return a
    end
    @model function dynamic_slice_return(auto_prefix=true)
        a = zeros(2)
        a[begin:end] ~ to_submodel(vector_return(), auto_prefix)
        return a
    end
    for index in (1:2, :, [true, false], [1, 2], true, 1.0, :field),
        template in (NoTemplate(), zeros(2)),
        wrap in (identity, prefix_nested)

        @test_throws r"ArgumentError: Prefix index .* is not a scalar integer" prefix(
            wrap(prefix_leaf()), @varname(p[index]); template
        )
        @test prefix(wrap(prefix_leaf()), @varname(p[1]); template)(Xoshiro(1)) == 3.0
        @test_throws r"ArgumentError: Prefix index .* is not a scalar integer" wrap(
            indexed_return(index)
        )(
            Xoshiro(1)
        )
        @test wrap(indexed_return(1))(Xoshiro(1))[1] == 3.0
    end
    for index in (1:2, :, [true, true]), wrap in (identity, prefix_nested)
        @test_throws r"ArgumentError: Prefix index .* is not a scalar integer" wrap(
            sliced_return(index)
        )(
            Xoshiro(1)
        )
        @test_throws r"Unprefixed submodels" wrap(sliced_return(index, false))(Xoshiro(1))
    end
    @test_throws r"ArgumentError: Prefix index .* is not a scalar integer" dynamic_slice_return()(
        Xoshiro(1)
    )
    @test_throws r"Unprefixed submodels" dynamic_slice_return(false)(Xoshiro(1))
    @test_throws r"ArgumentError: `begin` and `end` in a prefix need a template" prefix(
        prefix_leaf(), @varname(p[end])
    )
    @test prefix(prefix_leaf(), @varname(p[end]); template=zeros(2))(Xoshiro(1)) == 3.0
    @test_throws r"ArgumentError: Prefix index .* is not a scalar integer" prefix(
        prefix_leaf(), @varname(p[begin:end]); template=zeros(2)
    )
    for name in (@varname(p[1].q[:]), @varname(p[1:2].q[end]))
        @test_throws r"ArgumentError: Prefix index .* is not a scalar integer" prefix(
            prefix_leaf(), name; template=[(q=zeros(2),), (q=zeros(2),)]
        )
        model = prefix(
            prefix_leaf(), @varname(p[2].q[end]); template=[(q=zeros(2),), (q=zeros(2),)]
        )
        @test DynamicPPL._model_prefix(model) == @varname(p[2].q[2])
        @test model(Xoshiro(1)) == 3.0
    end
    for (name, canonical) in (
        (@varname(p[begin]), @varname(p[1])),
        (@varname(p[end]), @varname(p[4])),
        (@varname(p[end, end]), @varname(p[2, 2])),
        (@varname(p[CartesianIndex(1, 2)]), @varname(p[1, 2])),
        (@varname(p[CartesianIndex(1), 2]), @varname(p[1, 2])),
    )
        for leaf in (prefix_leaf(), condition(prefix_leaf(); x=3.0))
            model = prefix(leaf, name; template=zeros(2, 2))
            @test DynamicPPL._model_prefix(model) == canonical
            @test model(Xoshiro(1)) == 3.0
            @test prefix_nested(model)(Xoshiro(1)) == 3.0
        end
    end
    cartesian = indexed_return(CartesianIndex(1, 2))
    @test keys(rand(Xoshiro(1), cartesian)) == [@varname(a[1, 2].x)]
    @test keys(rand(Xoshiro(1), prefix_nested(cartesian))) == [@varname(outer.a[1, 2].x)]
    canonicalize(vn) = DynamicPPL._concretize_prefix(vn, zeros(2, 2); prefix=Val(true))
    @test (@inferred canonicalize(@varname(p[CartesianIndex(1, 2)].q))) ==
        @varname(p[1, 2].q)
    # Bindings and removals accept either spelling of the prefix, but not a linear index.
    model = prefix(prefix_leaf(), @varname(p[CartesianIndex(1, 2)]))
    for name in (@varname(p[CartesianIndex(1, 2)].x), @varname(p[1, 2].x))
        bound = conditioned(condition(model, name => 3.0))
        @test keys(bound) == [@varname(p[1, 2].x)]
        @test bound[@varname(p[1, 2].x)] == 3.0
        @test conditioned(condition(model, bound)) == bound
        @test isempty(fixed(unfix(fix(model, @varname(p[1, 2].x) => 3.0), name)))
    end
    @test_throws r"outside this model's prefix" condition(model, @varname(p[3].x) => 3.0)
    # `Int32` indices address the same storage as `Int` indices.
    model = prefix(prefix_leaf(), @varname(p[Int32(2)]))
    bound = conditioned(condition(model, @varname(p[2].x) => 3.0))
    @test keys(bound) == [@varname(p[2].x)]
    @test bound[@varname(p[2].x)] == 3.0
    model = condition(prefix(prefix_leaf(), @varname(p[2])), @varname(p[2].x) => 3.0)
    @test isempty(conditioned(decondition(model, @varname(p[Int32(2)].x))))
    model = condition(
        indexed_return(Int32(2)), @varname(a[2].x) => 3.0, @of(a = of(Array, 2, 2))
    )
    @test model(Xoshiro(1))[2] == 3.0
end

@testset "submodels reject argument observations" begin
    @model inner(y, x) = y ~ Normal(x)
    @model direct(y) = a ~ to_submodel(inner(y, 0.3))
    @model function indexed(child)
        a = Vector{Any}(undef, 1)
        a[1] ~ to_submodel(child)
        return a
    end
    @model function chosen(flag)
        child = flag ? inner(1.0, 0.3) : inner(missing, 0.3)
        return a ~ to_submodel(child)
    end
    @model multiple(y, z, x) = (y ~ Normal(x); z ~ Normal(x))
    @model keyword(; y=1.0, x=0.3) = y ~ Normal(x)

    message = "Submodel `a` observes `a.y` through its argument `y`; submodels cannot take observations as arguments. Pass `missing` for `y` and condition the submodel on the value instead, e.g. `a ~ to_submodel(condition(inner(missing, x), (y=y,)))`. Use `condition` rather than `fix`: `condition` keeps the likelihood term for `y`, while `fix` sets `y` without one."
    @test_throws ArgumentError(message) direct(1.0)(Xoshiro(1))
    for model in (
        argument_geometry_parent(inner(1.0, 0.3)),
        argument_geometry_parent(prefix(inner(1.0, 0.3), @varname(p))),
        argument_geometry_parent(removal_wrapper(inner(1.0, 0.3))),
        indexed(inner(1.0, 0.3)),
        chosen(true),
        argument_geometry_parent(keyword()),
    )
        @test_throws r"submodels cannot take observations as arguments" model(Xoshiro(1))
    end
    @test_throws r"observes `a.y`, `a.z`.*multiple\(missing, missing, x\).*y=y, z=z" argument_geometry_parent(
        multiple(1.0, 2.0, 0.3)
    )(
        Xoshiro(1)
    )
    for placeholder in (missing, nothing)
        model = argument_geometry_parent(inner(placeholder, 0.3))
        @test logjoint(model, Dict(@varname(a.y) => 1.0)) ≈ logpdf(Normal(0.3), 1.0)
    end
    @test logjoint(chosen(false), Dict(@varname(a.y) => 1.0)) ≈ logpdf(Normal(0.3), 1.0)
    @test logjoint(inner(1.0, 0.3), (;)) ≈ logpdf(Normal(0.3), 1.0)
    @test loglikelihood(inner(1.0, 0.3), (;)) ≈ logpdf(Normal(0.3), 1.0)
    @test logjoint(argument_geometry_parent(condition(inner(missing, 0.3); y=1.0)), (;)) ≈
        logpdf(Normal(0.3), 1.0)
    @test logjoint(
        argument_geometry_parent(decondition(inner(1.0, 0.3))), Dict(@varname(a.y) => 1.0)
    ) ≈ logpdf(Normal(0.3), 1.0)
    @model gated(run, child) = (run && (a ~ to_submodel(child)); nothing)
    @test gated(false, inner(1.0, 0.3))(Xoshiro(1)) === nothing
    @test_throws ArgumentError gated(true, inner(1.0, 0.3))(Xoshiro(1))
    @model partial(y) = (y[1] ~ Normal(); y[2] ~ Normal())
    @test_throws r"submodels cannot take observations as arguments" argument_geometry_parent(
        decondition(partial([1.0, 2.0]), @varname(y[1]))
    )(
        Xoshiro(1)
    )
    for names in ((), (@varname(a.y),))
        @test_throws r"submodels cannot take observations as arguments" decondition(
            argument_geometry_parent(inner(1.0, 0.3)), names...
        )(
            Xoshiro(1)
        )
    end
    @test_throws r"keyword\(; y=missing, x=x\)" argument_geometry_parent(keyword())(
        Xoshiro(1)
    )

    @model splatted(y...) = y[1] ~ Normal()
    @model keyword_splatted(; y...) = y[:x] ~ Normal()
    @test_throws r"splatted\(map\(_ -> missing, y\)\.\.\.\)" argument_geometry_parent(
        splatted(1.0)
    )()
    @test_throws r"keyword_splatted\(; map\(_ -> missing, NamedTuple\(y\)\)\.\.\.\)" argument_geometry_parent(
        keyword_splatted(; x=1.0)
    )()
    y = (1.0,)
    @test loglikelihood(
        argument_geometry_parent(condition(splatted(map(_ -> missing, y)...); y)), (;)
    ) ≈ logpdf(Normal(), 1.0)
    y = (; x=1.0)
    @test loglikelihood(
        argument_geometry_parent(
            condition(keyword_splatted(; map(_ -> missing, NamedTuple(y))...); y)
        ),
        (;),
    ) ≈ logpdf(Normal(), 1.0)
end

@testset "explicit submodel observations preserve demo densities" begin
    model = DynamicPPL.TestUtils.demo_dot_assume_observe_submodel()
    params = (; s=[1.0, 2.0], m=[0.3, 0.4])
    @test logprior(model, params) ≈ DynamicPPL.TestUtils.logprior_true(model, params...)
    @test loglikelihood(model, params) ≈
        DynamicPPL.TestUtils.loglikelihood_true(model, params...)
    @test logjoint(model, params) ≈ DynamicPPL.TestUtils.logjoint_true(model, params...)
    @test returned(model, params) == (; params..., x=[1.5, 2.0])
    @test Set(keys(rand(Xoshiro(1), model))) == Set(DynamicPPL.TestUtils.varnames(model))
end

@testset "removals recurse by default" begin
    @model body(bind) = a ~ to_submodel(bind(namespace_leaf(); x=2.0))
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        child = bind(namespace_leaf(); x=2.0)
        for (model, address) in (
            (argument_geometry_parent(child), @varname(a.x)),
            (body(bind), @varname(a.x)),
            (removal_wrapper(argument_geometry_parent(child)), @varname(b.a.x)),
            (prefix(argument_geometry_parent(child), @varname(p)), @varname(p.a.x)),
            (argument_geometry_parent(prefix(child, @varname(p))), @varname(a.p.x)),
        )
            for names in ((), (address,))
                removed = remove(model, names...)
                @test keys(rand(Xoshiro(1), removed)) == [address]
                @test logprior(removed, Dict(address => 0.3)) ≈ logpdf(Normal(), 0.3)
                @test loglikelihood(removed, Dict(address => 0.3)) == 0.0
                @test typeof(remove(removed, names...)) === typeof(removed)
                replaced = bind(removed, address => 3.0)
                @test replaced(Xoshiro(1)) == 3.0
                @test logjoint(replaced, (;)) ≈
                    (bind === condition ? logpdf(Normal(), 3.0) : 0.0)
            end
        end
        @test_logs @test check_model(Xoshiro(1), remove(namespace_leaf()))
        @test_logs @test check_model(
            Xoshiro(1), prefix(remove(namespace_leaf()), @varname(p))
        )
        conflicting = bind(argument_geometry_parent(child), @varname(a.x) => 3.0)
        @test_throws r"remove it with `decondition`/`unfix` first" conflicting()
    end
    @model observation(y) = y ~ Normal()
    for names in ((), (@varname(y),))
        @test keys(rand(Xoshiro(1), decondition(observation(2.0), names...))) ==
            [@varname(y)]
    end
    @model dormant(run) = (run && (y ~ Normal()); x ~ Normal())
    @test_logs (:warn, r"removal.*y.*unused") check_model(
        Xoshiro(1), decondition(dormant(false), @varname(y))
    )
end

@testset "placeholder binding removals infer concretely" begin
    @model observation(y) = y ~ Normal()
    observed = condition(observation(missing); y=0.0)
    fixed_model = fix(observation(missing); y=0.0)
    @test @inferred(decondition(observed)) isa Model
    @test @inferred(decondition(observed, @varname(y))) isa Model
    @test @inferred(unfix(fixed_model)) isa Model
    @test @inferred(unfix(fixed_model, @varname(y))) isa Model
end

@testset "dynamic removal addresses" begin
    @model fields(p) = (p.a[1] ~ Normal(); p.a[2] ~ Normal(); p.a)
    @model parent() = a ~ to_submodel(deferred_array_leaves())
    for (bind, remove) in ((condition, decondition), (fix, unfix))
        original = bind(decondition(fields((a=[1.0, 2.0],))); p=(a=[1.0, 2.0],))
        changed = remove(original, @varname(p.a[end]))
        @test bind(changed, @varname(p.a[2]) => 3.0)() == [1.0, 3.0]
        model = bind(parent(); a=(x=[1.0, 2.0],))
        removed = remove(model, @varname(a.x[end]))
        @test returned(removed, Dict(@varname(a.x[2]) => 3.0)) == [1.0, 3.0]
    end
end

end
