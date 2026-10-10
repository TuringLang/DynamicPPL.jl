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

    @testset "stored values follow operation order" begin
        conditioned_first = DynamicPPL.prefix(condition(model; x=1.0), @varname(a))
        @test conditioned(conditioned_first)[@varname(a.x)] == 1.0
        @test conditioned_first().x == 1.0

        prefixed_first = condition(
            DynamicPPL.prefix(model, @varname(a)), @varname(a.x) => 2.0
        )
        @test conditioned(prefixed_first)[@varname(a.x)] == 2.0
        @test prefixed_first().x == 2.0

        fixed_first = DynamicPPL.prefix(fix(model; x=3.0), @varname(a))
        @test fixed(fixed_first)[@varname(a.x)] == 3.0
        @test fixed_first().x == 3.0

        prefixed_then_fixed = fix(
            DynamicPPL.prefix(model, @varname(a)), @varname(a.x) => 4.0
        )
        @test fixed(prefixed_then_fixed)[@varname(a.x)] == 4.0
        @test prefixed_then_fixed().x == 4.0

        @test isempty(keys(conditioned(decondition(conditioned_first, @varname(a)))))
        @test isempty(keys(fixed(unfix(fixed_first, @varname(a)))))
    end

    @testset "explicit bindings stay under the model prefix" begin
        @model argument_lhs(y) = y ~ Normal()
        for op in (condition, fix)
            prefixed_argument = DynamicPPL.prefix(argument_lhs(1.0), @varname(p))
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
            @test_throws ArgumentError op(nested, @varname(p.y) => 0.0)
            @test_throws ArgumentError op(nested, @varname(q.other.y) => 0.0)
            @test op(nested, @varname(q.p.y) => 4.0)(Xoshiro(1)) == 4.0

            indexed = DynamicPPL.prefix(argument_lhs(1.0), @varname(p[1]))
            @test_throws ArgumentError op(indexed, @varname(p[2].y) => 0.0)
            @test op(indexed, @varname(p[1].y) => 5.0)(Xoshiro(1)) == 5.0

            @model function parent_binding(child)
                result ~ to_submodel(child)
                return result
            end
            @model function parent_binding_unprefixed(child)
                result ~ to_submodel(child, false)
                return result
            end
            for auto_prefix in (false, true)
                child = op(nested, @varname(q.p.y) => 6.0)
                parent =
                    auto_prefix ? parent_binding(child) : parent_binding_unprefixed(child)
                @test parent(Xoshiro(1)) == 6.0
                address = auto_prefix ? @varname(result.q.p.y) : @varname(q.p.y)
                @test op(parent, address => 7.0)(Xoshiro(1)) == 7.0
            end
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
