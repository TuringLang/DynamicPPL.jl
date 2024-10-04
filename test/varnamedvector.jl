replace_sym(vn::VarName, sym_new::Symbol) = VarName{sym_new}(vn.lens)

increase_size_for_test(x::Real) = [x]
increase_size_for_test(x::AbstractArray) = repeat(x, 2)

decrease_size_for_test(x::Real) = x
decrease_size_for_test(x::AbstractVector) = first(x)
decrease_size_for_test(x::AbstractArray) = first(eachslice(x; dims=1))

function need_varnames_relaxation(vnv::VarNamedVector, vn::VarName, val)
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
function need_varnames_relaxation(vnv::VarNamedVector, vns, vals)
    return any(need_varnames_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
end

function need_values_relaxation(vnv::VarNamedVector, vn::VarName, val)
    if isconcretetype(eltype(vnv.vals))
        return promote_type(eltype(vnv.vals), eltype(val)) != eltype(vnv.vals)
    end

    return false
end
function need_values_relaxation(vnv::VarNamedVector, vns, vals)
    return any(need_values_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
end

function need_transforms_relaxation(vnv::VarNamedVector, vn::VarName, val)
    return if isconcretetype(eltype(vnv.transforms))
        # If the container is concrete, we need to make sure that the sizes match.
        # => If the sizes don't match, we need to relax the container type.
        any(keys(vnv)) do vn_present
            size(vnv[vn_present]) != size(val)
        end
    elseif eltype(vnv.transforms) !== Any
        # If it's not concrete AND it's not `Any`, then we should just make it `Any`.
        true
    else
        # Otherwise, it's `Any`, so we don't need to relax the container type.
        false
    end
end
function need_transforms_relaxation(vnv::VarNamedVector, vns, vals)
    return any(need_transforms_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
end

"""
    relax_container_types(vnv::VarNamedVector, vn::VarName, val)
    relax_container_types(vnv::VarNamedVector, vns, val)

Relax the container types of `vnv` if necessary to accommodate `vn` and `val`.

This attempts to avoid unnecessary container type relaxations by checking whether
the container types of `vnv` are already compatible with `vn` and `val`.

# Notes
For example, if `vn` is not compatible with the current keys in `vnv`, then
the underlying types will be changed to `VarName` to accommodate `vn`.

Similarly:
- If `val` is not compatible with the current values in `vnv`, then
  the underlying value type will be changed to `Real`.
- If `val` requires a transformation that is not compatible with the current
  transformations type in `vnv`, then the underlying transformation type will
  be changed to `Any`.
"""
function relax_container_types(vnv::VarNamedVector, vn::VarName, val)
    return relax_container_types(vnv, [vn], [val])
end
function relax_container_types(vnv::VarNamedVector, vns, vals)
    if need_varnames_relaxation(vnv, vns, vals)
        varname_to_index_new = convert(OrderedDict{VarName,Int}, vnv.varname_to_index)
        varnames_new = convert(Vector{VarName}, vnv.varnames)
    else
        varname_to_index_new = vnv.varname_to_index
        varnames_new = vnv.varnames
    end

    transforms_new = if need_transforms_relaxation(vnv, vns, vals)
        convert(Vector{Any}, vnv.transforms)
    else
        vnv.transforms
    end

    vals_new = if need_values_relaxation(vnv, vns, vals)
        convert(Vector{Real}, vnv.vals)
    else
        vnv.vals
    end

    return VarNamedVector(
        varname_to_index_new,
        varnames_new,
        vnv.ranges,
        vals_new,
        transforms_new,
        vnv.is_unconstrained,
        vnv.num_inactive,
    )
end

@testset "VarNamedVector" begin
    # Test element-related operations:
    # - `getindex`
    # - `setindex!`
    # - `push!`
    # - `update!`
    # - `insert!`
    # - `reset!`
    # - `_internal!` versions of the above
    # - !! versions of the above
    #
    # And these are all be tested for different types of values:
    # - scalar
    # - vector
    # - matrix

    # Test operations on `VarNamedVector`:
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
    test_vals = collect(values(test_pairs))

    @testset "constructor: no args" begin
        # Empty.
        vnv = VarNamedVector()
        @test isempty(vnv)
        @test eltype(vnv) == Real

        # Empty with types.
        vnv = VarNamedVector{VarName,Float64}()
        @test isempty(vnv)
        @test eltype(vnv) == Float64
    end

    test_varnames_iter = combinations(test_vns, 2)
    @testset "$(vn_left) and $(vn_right)" for (vn_left, vn_right) in test_varnames_iter
        val_left = test_pairs[vn_left]
        val_right = test_pairs[vn_right]
        vnv_base = VarNamedVector([vn_left, vn_right], [val_left, val_right])

        # We'll need the transformations later.
        # TODO: Should we test other transformations than just `ReshapeTransform`?
        from_vec_left = DynamicPPL.from_vec_transform(val_left)
        from_vec_right = DynamicPPL.from_vec_transform(val_right)
        to_vec_left = inverse(from_vec_left)
        to_vec_right = inverse(from_vec_right)

        # Compare to alternative constructors.
        vnv_from_dict = VarNamedVector(
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

        @test eltype(vnv_base) == promote_type(eltype(val_left), eltype(val_right))
        @test length_internal(vnv_base) == length(val_left) + length(val_right)
        @test length(vnv_base) == 2

        @test !isempty(vnv_base)

        @testset "empty!" begin
            vnv = deepcopy(vnv_base)
            empty!(vnv)
            @test isempty(vnv)
        end

        @testset "similar" begin
            vnv = similar(vnv_base)
            @test isempty(vnv)
            @test typeof(vnv) == typeof(vnv_base)
        end

        @testset "getindex" begin
            # With `VarName` index.
            @test vnv_base[vn_left] == val_left
            @test vnv_base[vn_right] == val_right
        end

        @testset "getindex_internal" begin
            @test DynamicPPL.getindex_internal(vnv_base, vn_left) == to_vec_left(val_left)
            @test DynamicPPL.getindex_internal(vnv_base, vn_right) ==
                to_vec_right(val_right)
        end

        @testset "getindex_internal with Ints" begin
            for (i, val) in enumerate(to_vec_left(val_left))
                @test DynamicPPL.getindex_internal(vnv_base, i) == val
            end
            offset = length(to_vec_left(val_left))
            for (i, val) in enumerate(to_vec_right(val_right))
                @test DynamicPPL.getindex_internal(vnv_base, offset + i) == val
            end
        end

        @testset "update!" begin
            vnv = deepcopy(vnv_base)
            update!(vnv, val_left .+ 100, vn_left)
            @test vnv[vn_left] == val_left .+ 100
            update!(vnv, val_right .+ 100, vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "update!!" begin
            vnv = deepcopy(vnv_base)
            vnv = update!!(vnv, val_left .+ 100, vn_left)
            @test vnv[vn_left] == val_left .+ 100
            vnv = update!!(vnv, val_right .+ 100, vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "update_internal!" begin
            vnv = deepcopy(vnv_base)
            update_internal!(vnv, to_vec_left(val_left .+ 100), vn_left)
            @test vnv[vn_left] == val_left .+ 100
            update_internal!(vnv, to_vec_right(val_right .+ 100), vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "update_internal!!" begin
            vnv = deepcopy(vnv_base)
            vnv = update_internal!!(vnv, to_vec_left(val_left .+ 100), vn_left)
            @test vnv[vn_left] == val_left .+ 100
            vnv = update_internal!!(vnv, to_vec_right(val_right .+ 100), vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "delete!" begin
            vnv = deepcopy(vnv_base)
            delete!(vnv, vn_left)
            @test !haskey(vnv, vn_left)
            @test haskey(vnv, vn_right)
            delete!(vnv, vn_right)
            @test !haskey(vnv, vn_right)
        end

        @testset "insert!" begin
            vnv = deepcopy(vnv_base)
            delete!(vnv, vn_left)
            delete!(vnv, vn_right)
            insert!(vnv, val_left .+ 100, vn_left)
            @test vnv[vn_left] == val_left .+ 100
            insert!(vnv, val_right .+ 100, vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "insert!!" begin
            vnv = deepcopy(vnv_base)
            delete!(vnv, vn_left)
            delete!(vnv, vn_right)
            vnv = insert!!(vnv, val_left .+ 100, vn_left)
            @test vnv[vn_left] == val_left .+ 100
            vnv = insert!!(vnv, val_right .+ 100, vn_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "insert_internal!" begin
            vnv = deepcopy(vnv_base)
            delete!(vnv, vn_left)
            delete!(vnv, vn_right)
            insert_internal!(vnv, to_vec_left(val_left .+ 100), vn_left, from_vec_left)
            @test vnv[vn_left] == val_left .+ 100
            insert_internal!(vnv, to_vec_right(val_right .+ 100), vn_right, from_vec_right)
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "insert_internal!!" begin
            vnv = deepcopy(vnv_base)
            delete!(vnv, vn_left)
            delete!(vnv, vn_right)
            vnv = insert_internal!!(
                vnv, to_vec_left(val_left .+ 100), vn_left, from_vec_left
            )
            @test vnv[vn_left] == val_left .+ 100
            vnv = insert_internal!!(
                vnv, to_vec_right(val_right .+ 100), vn_right, from_vec_right
            )
            @test vnv[vn_right] == val_right .+ 100
        end

        @testset "merge" begin
            # When there are no inactive entries, `merge` on itself result in the same.
            @test merge(vnv_base, vnv_base) == vnv_base

            # Merging with empty should result in the same.
            @test merge(vnv_base, similar(vnv_base)) == vnv_base
            @test merge(similar(vnv_base), vnv_base) == vnv_base

            # With differences.
            vnv_left_only = deepcopy(vnv_base)
            delete!(vnv_left_only, vn_right)
            vnv_right_only = deepcopy(vnv_base)
            delete!(vnv_right_only, vn_left)

            # `(x,)` and `(x, y)` should be `(x, y)`.
            @test merge(vnv_left_only, vnv_base) == vnv_base
            # `(x, y)` and `(x,)` should be `(x, y)`.
            @test merge(vnv_base, vnv_left_only) == vnv_base
            # `(x, y)` and `(y,)` should be `(x, y)`.
            @test merge(vnv_base, vnv_right_only) == vnv_base
            # `(y,)` and `(x, y)` should be `(y, x)`.
            vnv_merged = merge(vnv_right_only, vnv_base)
            @test vnv_merged != vnv_base
            @test collect(keys(vnv_merged)) == [vn_right, vn_left]
        end

        @testset "push!" begin
            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn" for vn in test_vns
                val = test_pairs[vn]
                vnv_copy = deepcopy(vnv)
                push!(vnv, (vn => val))
                @test vnv[vn] == val
            end
        end

        @testset "setindex_internal!" begin
            # Not setting the transformation.
            vnv = deepcopy(vnv_base)
            setindex_internal!(vnv, to_vec_left(val_left .+ 100), vn_left)
            @test vnv[vn_left] == val_left .+ 100
            setindex_internal!(vnv, to_vec_right(val_right .+ 100), vn_right)
            @test vnv[vn_right] == val_right .+ 100

            # Explicitly setting the transformation.
            increment(x) = x .+ 10
            vnv = deepcopy(vnv_base)
            vnv = loosen_types!!(vnv, typeof(vn_left), typeof(increment))
            setindex_internal!(vnv, to_vec_left(val_left .+ 100), vn_left, increment)
            @test vnv[vn_left] == to_vec_left(val_left .+ 110)

            vnv = loosen_types!!(vnv, typeof(vn_right), typeof(increment))
            setindex_internal!(vnv, to_vec_right(val_right .+ 100), vn_right, increment)
            @test vnv[vn_right] == to_vec_right(val_right .+ 110)

            # Adding new values.
            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn" for vn in test_vns
                val = test_pairs[vn]
                from_vec_vn = DynamicPPL.from_vec_transform(val)
                to_vec_vn = inverse(from_vec_vn)
                setindex_internal!(vnv, to_vec_vn(val), vn, from_vec_vn)
                @test vnv[vn] == val
            end
        end

        @testset "setindex_internal! with Ints" begin
            vnv = deepcopy(vnv_base)
            for i in 1:length_internal(vnv_base)
                setindex_internal!(vnv, i, i)
            end
            for i in 1:length_internal(vnv_base)
                @test getindex_internal(vnv, i) == i
            end
        end

        @testset "setindex_internal!!" begin
            # Not setting the transformation.
            vnv = deepcopy(vnv_base)
            vnv = setindex_internal!!(vnv, to_vec_left(val_left .+ 100), vn_left)
            @test vnv[vn_left] == val_left .+ 100
            vnv = setindex_internal!!(vnv, to_vec_right(val_right .+ 100), vn_right)
            @test vnv[vn_right] == val_right .+ 100

            # Explicitly setting the transformation.
            # Note that unlike with setindex_internal!, we don't need loosen_types!! here.
            increment(x) = x .+ 10
            vnv = deepcopy(vnv_base)
            vnv = setindex_internal!!(vnv, to_vec_left(val_left .+ 100), vn_left, increment)
            @test vnv[vn_left] == to_vec_left(val_left .+ 110)

            vnv = setindex_internal!!(
                vnv, to_vec_right(val_right .+ 100), vn_right, increment
            )
            @test vnv[vn_right] == to_vec_right(val_right .+ 110)

            # Adding new values.
            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn" for vn in test_vns
                val = test_pairs[vn]
                from_vec_vn = DynamicPPL.from_vec_transform(val)
                to_vec_vn = inverse(from_vec_vn)
                vnv = setindex_internal!!(vnv, to_vec_vn(val), vn, from_vec_vn)
                @test vnv[vn] == val
            end
        end

        @testset "setindex! and reset!" begin
            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn" for vn in test_vns
                val = test_pairs[vn]
                expected_length = if haskey(vnv, vn)
                    # If it's already present, the resulting length will be unchanged.
                    length_internal(vnv)
                else
                    length_internal(vnv) + length(val)
                end

                vnv[vn] = val .+ 1
                x = getindex_internal(vnv, :)
                @test vnv[vn] == val .+ 1
                @test length_internal(vnv) == expected_length
                @test length(x) == length_internal(vnv)
                @test all(getindex_internal(vnv, i) == x[i] for i in eachindex(x))

                # There should be no redundant values in the underlying vector.
                @test !DynamicPPL.has_inactive(vnv)
            end

            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn (increased size)" for vn in test_vns
                val_original = test_pairs[vn]
                val = increase_size_for_test(val_original)
                vn_already_present = haskey(vnv, vn)
                expected_length = if vn_already_present
                    # If it's already present, the resulting length will be altered.
                    length_internal(vnv) + length(val) - length(val_original)
                else
                    length_internal(vnv) + length(val)
                end

                # Have to use reset!, because setindex! doesn't support decreasing size.
                reset!(vnv, val .+ 1, vn)
                x = getindex_internal(vnv, :)
                @test vnv[vn] == val .+ 1
                @test length_internal(vnv) == expected_length
                @test length(x) == length_internal(vnv)
                @test all(getindex_internal(vnv, i) == x[i] for i in eachindex(x))
            end

            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn (decreased size)" for vn in test_vns
                val_original = test_pairs[vn]
                val = decrease_size_for_test(val_original)
                vn_already_present = haskey(vnv, vn)
                expected_length = if vn_already_present
                    # If it's already present, the resulting length will be altered.
                    length_internal(vnv) + length(val) - length(val_original)
                else
                    length_internal(vnv) + length(val)
                end

                # Have to use reset!, because setindex! doesn't support decreasing size.
                reset!(vnv, val .+ 1, vn)
                x = getindex_internal(vnv, :)
                @test vnv[vn] == val .+ 1
                @test length_internal(vnv) == expected_length
                @test length(x) == length_internal(vnv)
                @test all(getindex_internal(vnv, i) == x[i] for i in eachindex(x))
            end
        end
    end

    @testset "growing and shrinking" begin
        @testset "deterministic" begin
            n = 5
            vn = @varname(x)
            vnv = VarNamedVector(OrderedDict(vn => [true]))
            @test !DynamicPPL.has_inactive(vnv)
            # Growing should not create inactive ranges.
            for i in 1:n
                x = fill(true, i)
                update_internal!(vnv, x, vn, identity)
                @test !DynamicPPL.has_inactive(vnv)
            end

            # Same size should not create inactive ranges.
            x = fill(true, n)
            update_internal!(vnv, x, vn, identity)
            @test !DynamicPPL.has_inactive(vnv)

            # Shrinking should create inactive ranges.
            for i in (n - 1):-1:1
                x = fill(true, i)
                update_internal!(vnv, x, vn, identity)
                @test DynamicPPL.has_inactive(vnv)
                @test DynamicPPL.num_inactive(vnv, vn) == n - i
            end
        end

        @testset "random" begin
            n = 5
            vn = @varname(x)
            vnv = VarNamedVector(OrderedDict(vn => [true]))
            @test !DynamicPPL.has_inactive(vnv)

            # Insert a bunch of random-length vectors.
            for i in 1:100
                x = fill(true, rand(1:n))
                update!(vnv, x, vn)
            end
            # Should never be allocating more than `n` elements.
            @test DynamicPPL.num_allocated(vnv, vn) ≤ n

            # If we compaticfy, then it should always be the same size as just inserted.
            for i in 1:10
                x = fill(true, rand(1:n))
                update!(vnv, x, vn)
                DynamicPPL.contiguify!(vnv)
                @test DynamicPPL.num_allocated(vnv, vn) == length(x)
            end
        end
    end

    @testset "subset" begin
        vnv = VarNamedVector(test_pairs)
        @test subset(vnv, test_vns) == vnv
        @test subset(vnv, VarName[]) == VarNamedVector()
        @test merge(subset(vnv, test_vns[1:3]), subset(vnv, test_vns[4:end])) == vnv

        # Test that subset preserves transformations and unconstrainedness.
        vn = @varname(t[1])
        vns = vcat(test_vns, [vn])
        vnv = setindex_internal!!(vnv, [2.0], vn, x -> x .^ 2)
        DynamicPPL.settrans!(vnv, true, @varname(t[1]))
        @test vnv[@varname(t[1])] == [4.0]
        @test istrans(vnv, @varname(t[1]))
        @test subset(vnv, vns) == vnv
    end
end

@testset "VarInfo + VarNamedVector" begin
    models = DynamicPPL.TestUtils.DEMO_MODELS
    @testset "$(model.f)" for model in models
        # NOTE: Need to set random seed explicitly to avoid using the same seed
        # for initialization as for sampling in the inner testset below.
        Random.seed!(42)
        value_true = DynamicPPL.TestUtils.rand_prior_true(model)
        vns = DynamicPPL.TestUtils.varnames(model)
        varnames = DynamicPPL.TestUtils.varnames(model)
        varinfos = DynamicPPL.TestUtils.setup_varinfos(
            model, value_true, varnames; include_threadsafe=false
        )
        # Filter out those which are not based on `VarNamedVector`.
        varinfos = filter(DynamicPPL.has_varnamedvector, varinfos)
        # Get the true log joint.
        logp_true = DynamicPPL.TestUtils.logjoint_true(model, value_true...)

        @testset "$(short_varinfo_name(varinfo))" for varinfo in varinfos
            # Need to make sure we're using a different random seed from the
            # one used in the above call to `rand_prior_true`.
            Random.seed!(43)

            # Are values correct?
            DynamicPPL.TestUtils.test_values(varinfo, value_true, vns)

            # Is evaluation correct?
            varinfo_eval = last(
                DynamicPPL.evaluate!!(model, deepcopy(varinfo), DefaultContext())
            )
            # Log density should be the same.
            @test getlogp(varinfo_eval) ≈ logp_true
            # Values should be the same.
            DynamicPPL.TestUtils.test_values(varinfo_eval, value_true, vns)

            # Is sampling correct?
            varinfo_sample = last(
                DynamicPPL.evaluate!!(model, deepcopy(varinfo), SamplingContext())
            )
            # Log density should be different.
            @test getlogp(varinfo_sample) != getlogp(varinfo)
            # Values should be different.
            DynamicPPL.TestUtils.test_values(
                varinfo_sample, value_true, vns; compare=!isequal
            )
        end
    end
end
