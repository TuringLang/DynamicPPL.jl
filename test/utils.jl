@testset "utils.jl" begin
    @testset "addlogprob!" begin
        @model function testmodel()
            global lp_before = getlogp(__varinfo__)
            @addlogprob!(42)
            return global lp_after = getlogp(__varinfo__)
        end

        model = testmodel()
        varinfo = VarInfo(model)
        @test iszero(lp_before)
        @test getlogp(varinfo) == lp_after == 42
    end

    @testset "getargs_dottilde" begin
        # Some things that are not expressions.
        @test getargs_dottilde(:x) === nothing
        @test getargs_dottilde(1.0) === nothing
        @test getargs_dottilde([1.0, 2.0, 4.0]) === nothing

        # Some expressions.
        @test getargs_dottilde(:(x ~ Normal(μ, σ))) === nothing
        @test getargs_dottilde(:((.~)(x, Normal(μ, σ)))) == (:x, :(Normal(μ, σ)))
        @test getargs_dottilde(:((~).(x, Normal(μ, σ)))) == (:x, :(Normal(μ, σ)))
        @test getargs_dottilde(:(x .~ Normal(μ, σ))) == (:x, :(Normal(μ, σ)))
        @test getargs_dottilde(:(@. x ~ Normal(μ, σ))) === nothing
        @test getargs_dottilde(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) === nothing
        @test getargs_dottilde(:(@~ Normal.(μ, σ))) === nothing
    end

    @testset "getargs_tilde" begin
        # Some things that are not expressions.
        @test getargs_tilde(:x) === nothing
        @test getargs_tilde(1.0) === nothing
        @test getargs_tilde([1.0, 2.0, 4.0]) === nothing

        # Some expressions.
        @test getargs_tilde(:(x ~ Normal(μ, σ))) == (:x, :(Normal(μ, σ)))
        @test getargs_tilde(:((.~)(x, Normal(μ, σ)))) === nothing
        @test getargs_tilde(:((~).(x, Normal(μ, σ)))) === nothing
        @test getargs_tilde(:(@. x ~ Normal(μ, σ))) === nothing
        @test getargs_tilde(:(@. x ~ Normal(μ, $(Expr(:$, :(sqrt(v))))))) === nothing
        @test getargs_tilde(:(@~ Normal.(μ, σ))) === nothing
    end

    @testset "vectorize" begin
        dist = LKJCholesky(2, 1)
        x = rand(dist)
        @test vectorize(dist, x) == vec(x.UL)
    end

    @testset "BangBang.possible" begin
        # Some utility methods for testing `setindex!`.
        test_linear_index_only(::Tuple, ::AbstractArray) = false
        test_linear_index_only(inds::NTuple{1}, ::AbstractArray) = true
        test_linear_index_only(inds::NTuple{1}, ::AbstractVector) = false

        function replace_colon_with_axis(inds::Tuple, x)
            ntuple(length(inds)) do i
                inds[i] isa Colon ? axes(x, i) : inds[i]
            end
        end
        function replace_colon_with_vector(inds::Tuple, x)
            ntuple(length(inds)) do i
                inds[i] isa Colon ? collect(axes(x, i)) : inds[i]
            end
        end
        function replace_colon_with_range(inds::Tuple, x)
            ntuple(length(inds)) do i
                inds[i] isa Colon ? (1:size(x, i)) : inds[i]
            end
        end
        function replace_colon_with_booleans(inds::Tuple, x)
            ntuple(length(inds)) do i
                inds[i] isa Colon ? trues(size(x, i)) : inds[i]
            end
        end

        function replace_colon_with_range_linear(inds::NTuple{1}, x::AbstractArray)
            return inds[1] isa Colon ? (1:length(x),) : inds
        end

        @testset begin
            @test setindex!!((1, 2, 3), :two, 2) === (1, :two, 3)
            @test setindex!!((a=1, b=2, c=3), :two, :b) === (a=1, b=:two, c=3)
            @test setindex!!([1, 2, 3], :two, 2) == [1, :two, 3]
            @test setindex!!(Dict{Symbol,Int}(:a => 1, :b => 2), 10, :a) ==
                Dict(:a => 10, :b => 2)
            @test setindex!!(Dict{Symbol,Int}(:a => 1, :b => 2), 3, "c") ==
                Dict(:a => 1, :b => 2, "c" => 3)
        end

        @testset "mutation" begin
            @testset "without type expansion" begin
                for args in [([1, 2, 3], 20, 2), (Dict(:a => 1, :b => 2), 10, :a)]
                    @test setindex!!(args...) === args[1]
                end
            end

            @testset "with type expansion" begin
                @test setindex!!([1, 2, 3], [4, 5], 1) == [[4, 5], 2, 3]
                @test setindex!!([1, 2, 3], [4, 5, 6], :, 1) == [4, 5, 6]
            end
        end

        @testset "slices" begin
            @testset "$(typeof(x)) with $(src_idx)" for (x, src_idx) in [
                # Vector.
                (randn(2), (:,)),
                (randn(2), (1:2,)),
                # Matrix.
                (randn(2, 3), (:,)),
                (randn(2, 3), (:, 1)),
                (randn(2, 3), (:, 1:3)),
                # 3D array.
                (randn(2, 3, 4), (:, 1, :)),
                (randn(2, 3, 4), (:, 1:3, :)),
                (randn(2, 3, 4), (1, 1:3, :)),
            ]
                # Base case.
                @test @inferred(setindex!!(x, x[src_idx...], src_idx...)) === x

                # If we have `Colon` in the index, we replace this with other equivalent indices.
                if any(Base.Fix2(isa, Colon), src_idx)
                    if test_linear_index_only(src_idx, x)
                        # With range instead of `Colon`.
                        @test @inferred(
                            setindex!!(
                                x,
                                x[src_idx...],
                                replace_colon_with_range_linear(src_idx, x)...,
                            )
                        ) === x
                    else
                        # With axis instead of `Colon`.
                        @test @inferred(
                            setindex!!(
                                x, x[src_idx...], replace_colon_with_axis(src_idx, x)...
                            )
                        ) === x
                        # With range instead of `Colon`.
                        @test @inferred(
                            setindex!!(
                                x, x[src_idx...], replace_colon_with_range(src_idx, x)...
                            )
                        ) === x
                        # With vectors instead of `Colon`.
                        @test @inferred(
                            setindex!!(
                                x, x[src_idx...], replace_colon_with_vector(src_idx, x)...
                            )
                        ) === x
                        # With boolean index instead of `Colon`.
                        @test @inferred(
                            setindex!!(
                                x, x[src_idx...], replace_colon_with_booleans(src_idx, x)...
                            )
                        ) === x
                    end
                end
            end
        end
    end
end
