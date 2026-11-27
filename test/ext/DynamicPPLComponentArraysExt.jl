module DynamicPPLComponentArraysTests

using DynamicPPL, Distributions, Random, Test
using DynamicPPL: @vnt
using ComponentArrays: ComponentVector, getaxes

@testset "ComponentArray bindings" begin
    @model function local_components()
        x = ComponentVector(; a=[0.0, 0.0], b=0.0)
        x[:a][1] ~ Normal()
        x[:a][2] ~ Normal()
        x[:b] ~ Normal()
        return x
    end
    @model function positional_components()
        x = ComponentVector(; a=[0.0, 0.0], b=0.0)
        for i in 1:3
            x[i] ~ Normal()
        end
        return x
    end
    @model function field_components()
        x = (; c=ComponentVector(; a=[0.0, 0.0], b=0.0))
        for i in 1:3
            x.c[i] ~ Normal()
        end
        return x.c
    end
    @model function plain_components()
        x = zeros(3)
        for i in 1:3
            x[i] ~ Normal()
        end
        return x
    end
    @model function parent_components(components=positional_components())
        child ~ to_submodel(components)
        return child
    end
    @model function argument(x)
        x[:a][1] ~ Normal()
        return x
    end
    data = ComponentVector{Union{Missing,Float64}}(; a=[missing, 1.0], b=0.0)
    @test collect(argument(data)(Xoshiro(1))) == [rand(Xoshiro(1), Normal()), 1.0, 0.0]
    observed = ComponentVector(; a=[1.0, 2.0], b=0.0)
    @test collect(decondition(argument(observed), @varname(x[:a][1]))(Xoshiro(1))) ==
        [rand(Xoshiro(1), Normal()), 2.0, 0.0]
    rng = Xoshiro(1)
    draws = [rand(rng, Normal()) for _ in 1:2]
    for (op, remove) in ((condition, decondition), (fix, unfix))
        value = ComponentVector(; a=[5.0, 6.0], b=7.0)
        result = op(local_components(); x=value)(Xoshiro(1))
        @test result[:a] == [5.0, 6.0]
        @test result[:b] == 7.0
        @test getaxes(result) == getaxes(value)
        result = remove(op(local_components(); x=value), @varname(x[:a][1]))(Xoshiro(1))
        @test collect(result) == [rand(Xoshiro(1), Normal()), 6.0, 7.0]
        @test getaxes(result) == getaxes(value)
        @test collect(op(argument(data), @varname(x[:a][1]) => 9.0)(Xoshiro(1))) ==
            [9.0, 1.0, 0.0]
        values = @vnt begin
            @template x = value
            x[:a][1] := 9.0
        end
        @test collect(op(decondition(argument(value)), values)(Xoshiro(1))) ==
            [9.0, 6.0, 7.0]
        partial = op(local_components(), values)
        @test collect(partial(Xoshiro(1))) == [9.0, draws...]
        repeated = op(partial, values)
        @test repeated == partial
        @test hash(repeated) == hash(partial)
        matching = @vnt begin
            @template x = ComponentVector(; a=Float32[5, 6], b=7.0f0)
            x[:b] := 0.1
        end
        prior = op(positional_components(); x=[5.0, 6.0, 7.0])
        @test collect(op(prior, matching)(Xoshiro(1))) == [5.0, 6.0, 0.1]
        field_edit = @vnt begin
            @template x = (; c=value)
            x.c[:b] := 9.0
        end
        held_field = op(field_components(), @varname(x.c) => [5.0, 6.0, 7.0])
        @test collect(op(held_field, field_edit)(Xoshiro(1))) == [5.0, 6.0, 9.0]
        child_edit = @vnt begin
            @template child = (; x=value)
            child.x[:b] := 9.0
        end
        for child in (positional_components(), plain_components())
            held_child = op(parent_components(child), @varname(child.x) => [5.0, 6.0, 7.0])
            @test collect(op(held_child, child_edit)(Xoshiro(1))) == [5.0, 6.0, 9.0]
        end
        @test_throws "`child.x[1]`: local storage shape changed" op(
            parent_components(), @varname(child.x) => [5.0, 6.0]
        )(
            Xoshiro(1)
        )
    end
end

end
