module DynamicPPLPrefixTests

using AbstractPPL: AbstractPPL
using Dates: now
using Distributions
using DynamicPPL
using Test

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
