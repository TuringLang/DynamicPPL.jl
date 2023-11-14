replace_sym(vn::VarName, sym_new::Symbol) = VarName{sym_new}(vn.lens)

change_size_for_test(x::Real) = [x]
change_size_for_test(x::AbstractArray) = repeat(x, 2)

function need_varnames_relaxation(vnv::VarNameVector, vn::VarName, val)
    if isconcretetype(eltype(vnv.varnames))
        # If the container is concrete, we need to make sure that the varname types match.
        # E.g. if `vnv.varnames` has `eltype` `VarName{:x, IndexLens{Tuple{Int64}}}` then
        # we need `vn` to also be of this type.
        # => If the varname types don't match, we need to relax the container type.
        return any(keys(vnv)) do vn_present
            typeof(vn_present) !== typeof(val)
        end
    end

    return false
end

function need_values_relaxation(vnv::VarNameVector, vn::VarName, val)
    if isconcretetype(eltype(vnv.vals))
        return promote_type(eltype(vnv.vals), eltype(val)) != eltype(vnv.vals)
    end

    return false
end

function need_transforms_relaxation(vnv::VarNameVector, vn::VarName, val)
    if isconcretetype(eltype(vnv.transforms))
        # If the container is concrete, we need to make sure that the sizes match.
        # => If the sizes don't match, we need to relax the container type.
        return any(keys(vnv)) do vn_present
            size(vnv[vn_present]) != size(val)
        end
    end

    return false
end

function relax_container_types(vnv::VarNameVector, vn::VarName, val)
    return relax_container_types(vnv, [vn], [val])
end
function relax_container_types(vnv::VarNameVector, vns, vals)
    if any(need_varnames_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
        varname_to_index_new = convert(OrderedDict{VarName,Int}, vnv.varname_to_index)
        varnames_new = convert(Vector{VarName}, vnv.varnames)
    else
        varname_to_index_new = vnv.varname_to_index
        varnames_new = vnv.varnames
    end

    transforms_new =
        if any(need_transforms_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
            convert(Vector{Any}, vnv.transforms)
        else
            vnv.transforms
        end

    vals_new = if any(need_values_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
        convert(Vector{Any}, vnv.vals)
    else
        vnv.vals
    end

    return VarNameVector(
        varname_to_index_new,
        varnames_new,
        vnv.ranges,
        vals_new,
        transforms_new,
        vnv.inactive_ranges,
        vnv.metadata,
    )
end

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
        @varname(y[1]) => rand(),
        @varname(y[2]) => rand(2),
        @varname(y[3]) => rand(2, 3),
        @varname(z[1]) => rand(1:10),
        @varname(z[2]) => rand(1:10, 2),
        @varname(z[3]) => rand(1:10, 2, 3),
    )
    test_vns = collect(keys(test_pairs))
    test_vals = collect(test_vals)

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

    test_varnames_iter = combinations(test_vns, 2)
    @testset "$(vn_left) and $(vn_right)" for (vn_left, vn_right) in test_varnames_iter
        val_left = test_pairs[vn_left]
        val_right = test_pairs[vn_right]
        vnv_base = VarNameVector([vn_left, vn_right], [val_left, val_right])

        # We'll need the transformations later.
        # TODO: Should we test other transformations than just `FromVec`?
        from_vec_left = DynamicPPL.FromVec(val_left)
        from_vec_right = DynamicPPL.FromVec(val_right)
        to_vec_left = inverse(from_vec_left)
        to_vec_right = inverse(from_vec_right)

        # Compare to alternative constructors.
        vnv_from_dict = VarNameVector(
            OrderedDict(vn_left => val_left, vn_right => val_right)
        )
        @test vnv_base == vnv_from_dict

        # We want the types of fields such as `varnames` and `transforms` to specialize
        # whenever possible + some functionality, e.g. `push!`, is only sensible
        # if the underlying containers can support it.
        # Expected behavior
        should_have_restricted_varname_type = typeof(vn_left) == typeof(vn_right)
        should_have_restricted_transform_type = size(val_left) == size(val_right)
        # Actual behavior
        has_restricted_transform_type = isconcretetype(eltype(vnv_base.transforms))
        has_restricted_varname_type = isconcretetype(eltype(vnv_base.varnames))

        @testset "type specialization" begin
            @test !should_have_restricted_varname_type || has_restricted_varname_type
            @test !should_have_restricted_transform_type || has_restricted_transform_type
        end

        # `eltype`
        @test eltype(vnv_base) == promote_type(eltype(val_left), eltype(val_right))
        # `length`
        @test length(vnv_base) == length(val_left) + length(val_right)

        # `isempty`
        @test !isempty(vnv_base)

        # `empty!`
        @testset "empty!" begin
            vnv = deepcopy(vnv_base)
            empty!(vnv)
            @test isempty(vnv)
        end

        # `similar`
        @testset "similar" begin
            vnv = similar(vnv_base)
            @test isempty(vnv)
            @test typeof(vnv) == typeof(vnv_base)
        end

        # `getindex`
        @testset "getindex" begin
            # `getindex`
            @test vnv_base[vn_left] == val_left
            @test vnv_base[vn_right] == val_right
        end

        # `setindex!`
        @testset "setindex!" begin
            vnv = deepcopy(vnv_base)
            vnv[vn_left] = val_left .+ 100
            @test vnv[vn_left] == val_left .+ 100
            vnv[vn_right] = val_right .+ 100
            @test vnv[vn_right] == val_right .+ 100
        end

        # `getindex_raw`
        @testset "getindex_raw" begin
            @test DynamicPPL.getindex_raw(vnv_base, vn_left) == to_vec_left(val_left)
            @test DynamicPPL.getindex_raw(vnv_base, vn_right) == to_vec_right(val_right)
        end

        # `setindex_raw!`
        @testset "setindex_raw!" begin
            vnv = deepcopy(vnv_base)
            DynamicPPL.setindex_raw!(vnv, to_vec_left(val_left .+ 100), vn_left)
            @test vnv[vn_left] == val_left .+ 100
            DynamicPPL.setindex_raw!(vnv, to_vec_right(val_right .+ 100), vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        # `push!` & `update!`
        @testset "push!" begin
            vnv = relax_container_types(
                deepcopy(vnv_base), test_vns, test_vals
            )
            @testset "$vn" for vn in test_vns
                val = test_pairs[vn]
                if vn == vn_left || vn == vn_right
                    # Should not be possible to `push!` existing varname.
                    @test_throws ArgumentError push!(vnv, vn, val)
                else
                    push!(vnv, vn, val)
                    @test vnv[vn] == val
                end
            end
        end
        @testset "update!" begin
            vnv = relax_container_types(
                deepcopy(vnv_base), test_vns, test_vals
            )
            @testset "$vn" for vn in test_vns
                val = test_pairs[vn]
                expected_length = if haskey(vnv, vn)
                    # If it's already present, the resulting length will be unchanged.
                    length(vnv)
                else
                    length(vnv) + length(val)
                end

                DynamicPPL.update!(vnv, vn, val .+ 1)
                @test vnv[vn] == val .+ 1
                @test length(vnv) == expected_length
                @test length(vnv[:]) == length(vnv)

                # There should be no redundant values in the underlying vector.
                @test !DynamicPPL.has_inactive_ranges(vnv)
            end

            # Need to recompute valid varnames for the changing of the sizes; before
            # we required either a) the underlying `transforms` to be non-concrete,
            # or b) the sizes of the values to match. But now the sizes of the values
            # will change, so we can only test the former.
            vnv = relax_container_types(
                deepcopy(vnv_base), test_vns, test_vals
            )
            @testset "$vn (different size)" for vn in test_vns
                val_original = test_pairs[vn]
                val = change_size_for_test(val_original)
                vn_already_present = haskey(vnv, vn)
                expected_length = if vn_already_present
                    # If it's already present, the resulting length will be altered.
                    length(vnv) + length(val) - length(val_original)
                else
                    length(vnv) + length(val)
                end

                DynamicPPL.update!(vnv, vn, val .+ 1)
                @test vnv[vn] == val .+ 1
                @test length(vnv) == expected_length
                @test length(vnv[:]) == length(vnv)
            end
        end
    end
end
