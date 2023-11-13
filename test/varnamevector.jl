replace_sym(vn::VarName, sym_new::Symbol) = VarName{sym_new}(vn.lens)

@testset "VarNameVector" begin
    # Need to test element-related operations:
    # - `getindex`
    # - `setindex!`
    # - `push!`
    # - `update!`
    #
    # And these should all be tested for different types of values:
    # - scalar
    # - vector
    # - matrix

    # Need to test operations on `VarNameVector`:
    # - `empty!`
    # - `iterate`
    # - `convert` to
    #   - `AbstractDict`

    test_pairs = OrderedDict(
        @varname(x[1]) => rand(),
        @varname(x[2]) => rand(2),
        @varname(x[3]) => rand(2, 3),
        @varname(x[4]) => rand(2, 3, 4)
    )

    @testset "constructor" begin
        @testset "no args" begin
            # Empty.
            vnv = VarNameVector()
            @test isempty(vnv)
            @test eltype(vnv) == Real

            # Empty with types.
            vnv = VarNameVector{VarName,Float64}()
            @test isempty(vnv)
            @test eltype(vnv) == Float64
        end

        # Should be able to handle different types of values.
        @testset "$(vn_left) and $(vn_right)" for (vn_left, vn_right) in Iterators.product(
            keys(test_pairs), keys(test_pairs)
        )
            val_left = test_pairs[vn_left]
            val_right = test_pairs[vn_right]
            vnv = VarNameVector([vn_left, vn_right], [val_left, val_right])
            @test length(vnv) == length(val_left) + length(val_right)
            @test eltype(vnv) == promote_type(eltype(val_left), eltype(val_right))
        end

        # Should also work when mixing varnames with different symbols.
        @testset "$(vn_left) and $(replace_sym(vn_right, :y))" for (vn_left, vn_right) in Iterators.product(
            keys(test_pairs), keys(test_pairs)
        )
            val_left = test_pairs[vn_left]
            val_right = test_pairs[vn_right]
            vnv = VarNameVector([vn_left, replace_sym(vn_right, :y)], [val_left, val_right])
            @test length(vnv) == length(val_left) + length(val_right)
            @test eltype(vnv) == promote_type(eltype(val_left), eltype(val_right))
        end
    end

    @testset "basics" begin
        vns = [@varname(x[1]), @varname(x[2]), @varname(x[3])]
        vals = [1, 2:3, reshape(4:9, 2, 3)]
        vnv = VarNameVector(vns, vals)

        # `getindex`
        for (vn, val) in zip(vns, vals)
            @test vnv[vn] == val
        end

        # `setindex!`
        for (vn, val) in zip(vns, vals)
            vnv[vn] = val .+ 100
        end
        for (vn, val) in zip(vns, vals)
            @test vnv[vn] == val .+ 100
        end

        # `push!`
        vn = @varname(x[4])
        val = 10:12
        push!(vnv, vn, val)
        @test vnv[vn] == val

        # `push!` existing varname is not allowed
        @test_throws ArgumentError push!(vnv, vn, val)

        # `update!` works with both existing and new varname
        # existing
        val = 20:22
        DynamicPPL.update!(vnv, vn, val)
        @test vnv[vn] == val

        # new
        vn = @varname(x[5])
        val = 30:32
        DynamicPPL.update!(vnv, vn, val)
        @test vnv[vn] == val
    end
end
