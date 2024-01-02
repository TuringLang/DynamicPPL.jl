replace_sym(vn::VarName, sym_new::Symbol) = VarName{sym_new}(vn.lens)

increase_size_for_test(x::Real) = [x]
increase_size_for_test(x::AbstractArray) = repeat(x, 2)

decrease_size_for_test(x::Real) = x
decrease_size_for_test(x::AbstractVector) = first(x)
decrease_size_for_test(x::AbstractArray) = first(eachslice(x; dims=1))

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
function need_varnames_relaxation(vnv::VarNameVector, vns, vals)
    return any(need_varnames_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
end

function need_values_relaxation(vnv::VarNameVector, vn::VarName, val)
    if isconcretetype(eltype(vnv.vals))
        return promote_type(eltype(vnv.vals), eltype(val)) != eltype(vnv.vals)
    end

    return false
end
function need_values_relaxation(vnv::VarNameVector, vns, vals)
    return any(need_values_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
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
function need_transforms_relaxation(vnv::VarNameVector, vns, vals)
    return any(need_transforms_relaxation(vnv, vn, val) for (vn, val) in zip(vns, vals))
end

"""
    relax_container_types(vnv::VarNameVector, vn::VarName, val)
    relax_container_types(vnv::VarNameVector, vns, val)

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
function relax_container_types(vnv::VarNameVector, vn::VarName, val)
    return relax_container_types(vnv, [vn], [val])
end
function relax_container_types(vnv::VarNameVector, vns, vals)
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

    return VarNameVector(
        varname_to_index_new,
        varnames_new,
        vnv.ranges,
        vals_new,
        transforms_new,
        vnv.num_inactive,
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
    test_vals = collect(values(test_pairs))

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

        # `delete!`
        @testset "delete!" begin
            vnv = deepcopy(vnv_base)
            delete!(vnv, vn_left)
            @test !haskey(vnv, vn_left)
            @test haskey(vnv, vn_right)
            delete!(vnv, vn_right)
            @test !haskey(vnv, vn_right)
        end

        # `push!` & `update!`
        @testset "push!" begin
            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
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
            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
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
                @test !DynamicPPL.has_inactive(vnv)
            end

            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn (increased size)" for vn in test_vns
                val_original = test_pairs[vn]
                val = increase_size_for_test(val_original)
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

            vnv = relax_container_types(deepcopy(vnv_base), test_vns, test_vals)
            @testset "$vn (decreased size)" for vn in test_vns
                val_original = test_pairs[vn]
                val = decrease_size_for_test(val_original)
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

    @testset "growing and shrinking" begin
        @testset "deterministic" begin
            n = 5
            vn = @varname(x)
            vnv = VarNameVector(OrderedDict(vn => [true]))
            @test !DynamicPPL.has_inactive(vnv)
            # Growing should not create inactive ranges.
            for i in 1:n
                x = fill(true, i)
                DynamicPPL.update!(vnv, vn, x)
                @test !DynamicPPL.has_inactive(vnv)
            end

            # Same size should not create inactive ranges.
            x = fill(true, n)
            DynamicPPL.update!(vnv, vn, x)
            @test !DynamicPPL.has_inactive(vnv)

            # Shrinking should create inactive ranges.
            for i in (n - 1):-1:1
                x = fill(true, i)
                DynamicPPL.update!(vnv, vn, x)
                @test DynamicPPL.has_inactive(vnv)
                @test DynamicPPL.num_inactive(vnv, vn) == n - i
            end
        end

        @testset "random" begin
            n = 5
            vn = @varname(x)
            vnv = VarNameVector(OrderedDict(vn => [true]))
            @test !DynamicPPL.has_inactive(vnv)

            # Insert a bunch of random-length vectors.
            for i in 1:100
                x = fill(true, rand(1:n))
                DynamicPPL.update!(vnv, vn, x)
            end
            # Should never be allocating more than `n` elements.
            @test DynamicPPL.num_allocated(vnv, vn) ≤ n

            # If we compaticfy, then it should always be the same size as just inserted.
            for i in 1:10
                x = fill(true, rand(1:n))
                DynamicPPL.update!(vnv, vn, x)
                DynamicPPL.contiguify!(vnv)
                @test DynamicPPL.num_allocated(vnv, vn) == length(x)
            end
        end
    end
end

has_varnamevector(vi) = false
function has_varnamevector(vi::VarInfo)
    return vi.metadata isa VarNameVector ||
           (vi isa TypedVarInfo && first(values(vi.metadata)) isa VarNameVector)
end

@testset "VarInfo + VarNameVector" begin
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
        # Filter out those which are not based on `VarNameVector`.
        varinfos = filter(has_varnamevector, varinfos)
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
