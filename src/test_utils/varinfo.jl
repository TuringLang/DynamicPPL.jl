# varinfo.jl
# ----------
#
# Utilities for testing varinfos.

"""
    test_values(vi::AbstractVarInfo, vals::NamedTuple, vns)

Test that `vi[vn]` corresponds to the correct value in `vals` for every `vn` in `vns`.
"""
function test_values(vi::AbstractVarInfo, vals::NamedTuple, vns; compare=isequal, kwargs...)
    for vn in vns
        val = get(vals, vn)
        # TODO(mhauru) Workaround for https://github.com/JuliaLang/LinearAlgebra.jl/pull/1404
        # Remove once the fix is all Julia versions we support.
        if val isa Cholesky
            @test compare(vi[vn].L, val.L; kwargs...)
        else
            @test compare(vi[vn], val; kwargs...)
        end
    end
end

"""
    setup_varinfos(model::Model, example_values::NamedTuple, varnames; include_threadsafe::Bool=false)

Return a tuple of instances for different implementations of `AbstractVarInfo` with
each `vi`, supposedly, satisfying `vi[vn] == get(example_values, vn)` for `vn` in `varnames`.

If `include_threadsafe` is `true`, then the returned tuple will also include thread-safe versions
of the varinfo instances.
"""
function setup_varinfos(
    model::Model, example_values::NamedTuple, varnames; include_threadsafe::Bool=false
)
    vi = DynamicPPL.VarInfo(model)
    vi = update_values!!(vi, example_values, varnames)
    last(DynamicPPL.evaluate!!(model, vi))

    varinfos = if include_threadsafe
        (vi, DynamicPPL.ThreadSafeVarInfo(deepcopy(vi)))
    else
        (vi,)
    end
    return varinfos
end
