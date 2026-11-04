module DynamicPPLPrefixTests

using AbstractPPL: AbstractPPL
using Dates: now
using Distributions
using DynamicPPL
using Test
using Random: Xoshiro

@info "Testing $(@__FILE__)..."
__now__ = now()

@testset "model prefixes" begin
    @model function demo_prefix()
        x ~ Normal()
        y = Vector{Float64}(undef, 2)
        y[1] ~ Normal()
        y[2] ~ Normal()
        return (; x, y)
    end

    model = demo_prefix()
    prefixed = @inferred DynamicPPL.prefix(model, @varname(a))
    twice_prefixed = @inferred DynamicPPL.prefix(prefixed, @varname(b))

    @test Set(keys(rand(prefixed))) ==
        Set([@varname(a.x), @varname(a.y[1]), @varname(a.y[2])])
    @test Set(keys(rand(twice_prefixed))) ==
        Set([@varname(b.a.x), @varname(b.a.y[1]), @varname(b.a.y[2])])

    @testset "compound prefixes" begin
        for prefix_vn in (@varname(a.b), @varname(a[1]))
            transformed = DynamicPPL.prefix(model, prefix_vn)
            @test Set(keys(rand(transformed))) == Set([
                AbstractPPL.prefix(@varname(x), prefix_vn),
                AbstractPPL.prefix(@varname(y[1]), prefix_vn),
                AbstractPPL.prefix(@varname(y[2]), prefix_vn),
            ])
        end
    end

    @testset "bindings before and after prefixing" begin
        for (bind, bound, remove, before, after) in
            ((condition, conditioned, decondition, 1.0, 2.0), (fix, fixed, unfix, 3.0, 4.0))
            bound_first = DynamicPPL.prefix(bind(model; x=before), @varname(a))
            @test bound(bound_first)[@varname(a.x)] == before
            @test bound_first().x == before

            prefixed_first = bind(prefixed, @varname(a.x) => after)
            @test bound(prefixed_first)[@varname(a.x)] == after
            @test prefixed_first().x == after
            @test isempty(keys(bound(remove(bound_first, @varname(a)))))
        end
    end

    @testset "explicit bindings use prefixed names" begin
        @model argument_lhs(y) = y ~ Normal()
        for op in (condition, fix)
            argument = argument_lhs(1.0)
            op === fix && (argument = decondition(argument, @varname(y)))
            prefixed_argument = DynamicPPL.prefix(argument, @varname(p))
            message = "Cannot bind `y`: it is outside this model's prefix `p`."
            for bindings in ((; y=0.0), @varname(y) => 0.0, VarNamedTuple(; y=0.0))
                @test_throws message op(prefixed_argument, bindings)
            end
            @test op(prefixed_argument, @varname(p.y) => 2.0)(Xoshiro(1)) == 2.0
            @test op(prefixed_argument; p=(y=3.0,))(Xoshiro(1)) == 3.0
            @test_throws "not an LHS top symbol" op(
                prefixed_argument, @varname(p.unused) => 0.0
            )

            nested = DynamicPPL.prefix(prefixed_argument, @varname(q))
            @test_throws ArgumentError op(nested, @varname(q.other.y) => 0.0)
            @test op(nested, @varname(q.p.y) => 4.0)(Xoshiro(1)) == 4.0

            indexed = DynamicPPL.prefix(argument, @varname(p[1]))
            @test_throws ArgumentError op(indexed, @varname(p[2].y) => 0.0)
            @test op(indexed, @varname(p[1].y) => 5.0)(Xoshiro(1)) == 5.0

            @model parent_binding(child) = result ~ to_submodel(child)
            @model parent_binding_unprefixed(child) = result ~ to_submodel(child, false)
            child = op(nested, @varname(q.p.y) => 6.0)
            @test_throws ArgumentError parent_binding_unprefixed(child)(Xoshiro(1))
            parent = parent_binding(child)
            @test parent(Xoshiro(1)) == 6.0
            address = @varname(result.q.p.y)
            remove = op === condition ? decondition : unfix
            parent = remove(parent, address)
            @test op(parent, address => 7.0)(Xoshiro(1)) == 7.0
        end
    end

    @testset "dynamic prefixes use the supplied template" begin
        for op in (condition, fix)
            prefixed = @inferred DynamicPPL.prefix(
                op(model; x=2.0), @varname(a[end, end]); template=zeros(2, 3)
            )
            @test prefixed().x == 2.0
            values = op === condition ? conditioned(prefixed) : fixed(prefixed)
            @test size(values.data.a.data) == (2, 3)
            @test values[@varname(a[2, 3].x)] == 2.0
        end
    end
end

@info "Completed $(@__FILE__) in $(now() - __now__)."

end
