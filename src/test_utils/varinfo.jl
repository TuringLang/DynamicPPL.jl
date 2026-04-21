# varinfo.jl
# ----------
#
# Utilities for testing varinfos.

"""
    test_values(vnt::VarNamedTuple, vals::NamedTuple, vns)

Test that `vnt[vn]` corresponds to the correct value in `vals` for every `vn` in `vns`.
"""
function test_values(vnt::VarNamedTuple, vals::NamedTuple, vns; compare=isequal, kwargs...)
    for vn in vns
        val = AbstractPPL.getvalue(vals, vn)
        # TODO(mhauru) Workaround for https://github.com/JuliaLang/LinearAlgebra.jl/pull/1404
        # Remove once the fix is all Julia versions we support.
        if val isa Cholesky
            @test compare(vnt[vn].L, val.L; kwargs...)
        else
            @test compare(vnt[vn], val; kwargs...)
        end
    end
end

"""
    setup_varinfos(model::Model, example_values::NamedTuple; include_threadsafe::Bool=false)

Return a tuple of VarInfo (and ThreadSafeVarInfo, if `include_threadsafe=true`) where the
values in `vi` have been obtained by evaluating the model with `example_values`.
"""
function setup_varinfos(
    model::Model, example_values::NamedTuple; include_threadsafe::Bool=false
)
    _, vi = init!!(
        model, VarInfo(), InitFromParams(example_values, InitFromPrior()), UnlinkAll()
    )
    return if include_threadsafe
        (vi, DynamicPPL.ThreadSafeVarInfo(deepcopy(vi)))
    else
        (vi,)
    end
end
