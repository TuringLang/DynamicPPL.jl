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
        @test compare(vi[vn], get(vals, vn); kwargs...)
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
    # VarInfo
    vi_untyped_metadata = DynamicPPL.untyped_varinfo(model)
    vi_untyped_vnv = DynamicPPL.untyped_vector_varinfo(model)
    vi_typed_metadata = DynamicPPL.typed_varinfo(model)
    vi_typed_vnv = DynamicPPL.typed_vector_varinfo(model)

    # SimpleVarInfo
    svi_typed = SimpleVarInfo(example_values)
    svi_untyped = SimpleVarInfo(OrderedDict())
    svi_vnv = SimpleVarInfo(DynamicPPL.VarNamedVector())

    varinfos = map((
        vi_untyped_metadata,
        vi_untyped_vnv,
        vi_typed_metadata,
        vi_typed_vnv,
        svi_typed,
        svi_untyped,
        svi_vnv,
    )) do vi
        # Set them all to the same values and evaluate logp.
        vi = update_values!!(vi, example_values, varnames)
        last(DynamicPPL.evaluate!!(model, vi, DefaultContext()))
    end

    if include_threadsafe
        varinfos = (varinfos..., map(DynamicPPL.ThreadSafeVarInfo ∘ deepcopy, varinfos)...)
    end

    return varinfos
end
