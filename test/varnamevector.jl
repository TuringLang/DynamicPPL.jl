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

    test_pairs_x = OrderedDict(
        @varname(x[1]) => rand(),
        @varname(x[2]) => rand(2),
        @varname(x[3]) => rand(2, 3),
        @varname(x[4]) => rand(2, 3, 4),
    )
    test_pairs_y = OrderedDict(
        @varname(y[1]) => rand(),
        @varname(y[2]) => rand(2),
        @varname(y[3]) => rand(2, 3),
        @varname(y[4]) => rand(2, 3, 4),
    )
    test_pairs = merge(test_pairs_x, test_pairs_y)

    @testset "constructor: no args" begin
        # Empty.
        vnv = VarNameVector()
        @test isempty(vnv)
        @test eltype(vnv) == Real

        # Empty with types.
        vnv = VarNameVector{VarName,Float64}()
        @test isempty(vnv)
        @test eltype(vnv) == Float64
    end

    test_varnames_iter = combinations(collect(keys(test_pairs)), 2)
    @info "Testing varnames" collect(
        map(Base.Fix1(convert, Vector{VarName}), test_varnames_iter)
    )
    @testset "$(vn_left) and $(vn_right)" for (vn_left, vn_right) in test_varnames_iter
        val_left = test_pairs[vn_left]
        val_right = test_pairs[vn_right]
        vnv = VarNameVector([vn_left, vn_right], [val_left, val_right])

        # Compare to alternative constructors.
        vnv_from_dict = VarNameVector(
            OrderedDict(vn_left => val_left, vn_right => val_right)
        )
        @test vnv == vnv_from_dict

        # We want the types of fields such as `varnames` and `transforms` to specialize
        # whenever possible + some functionality, e.g. `push!`, is only sensible
        # if the underlying containers can support it.
        # Expected behavior
        should_have_restricted_varname_type = typeof(vn_left) == typeof(vn_right)
        should_have_restricted_transform_type = size(val_left) == size(val_right)
        # Actual behavior
        has_restricted_transform_type = isconcretetype(eltype(vnv.transforms))
        has_restricted_varname_type = isconcretetype(eltype(vnv.varnames))

        @testset "type specialization" begin
            @test !should_have_restricted_varname_type || has_restricted_varname_type
            @test !should_have_restricted_transform_type ||
                  has_restricted_transform_type
        end

        # `eltype`
        @test eltype(vnv) == promote_type(eltype(val_left), eltype(val_right))
        # `length`
        @test length(vnv) == length(val_left) + length(val_right)

        # `getindex`
        @testset "getindex" begin
            # `getindex`
            @test vnv[vn_left] == val_left
            @test vnv[vn_right] == val_right
        end
        # `setindex!`
        @testset "setindex!" begin
            vnv[vn_left] = val_left .+ 100
            @test vnv[vn_left] == val_left .+ 100
            vnv[vn_right] = val_right .+ 100
            @test vnv[vn_right] == val_right .+ 100
        end

        # `push!` & `update!`
        # These are only allowed for all the varnames if both the varname and
        # the transform types used in the underlying containers are not concrete.
        push_test_varnames = filter(keys(test_pairs)) do vn
            val = test_pairs[vn]
            transform_is_okay =
                !has_restricted_transform_type ||
                size(val) == size(val_left) ||
                size(val) == size(val_right)
            varname_is_okay =
                !has_restricted_varname_type ||
                typeof(vn) == typeof(vn_left) ||
                typeof(vn) == typeof(vn_right)
            return transform_is_okay && varname_is_okay
        end
        @testset "push! ($(vn))" for vn in push_test_varnames
            val = test_pairs[vn]
            if vn == vn_left || vn == vn_right
                # Should not be possible to `push!` existing varname.
                @test_throws ArgumentError push!(vnv, vn, val)
            else
                push!(vnv, vn, val)
                @test vnv[vn] == val
            end
        end
        @testset "update! ($(vn))" for vn in push_test_varnames
            val = test_pairs[vn]
            # Perturb `val` a bit so we can also check that the existing `vn_left` and `vn_right`
            # are also updated correctly.
            DynamicPPL.update!(vnv, vn, val .+ 1)
            @test vnv[vn] == val .+ 1
        end
    end
end
