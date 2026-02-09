"""
    OnlyAccsVarInfo(accs...)

`OnlyAccsVarInfo` is a wrapper around a tuple of accumulators.

Its name stems from the fact that it implements the minimal `AbstractVarInfo` interface to
work with the `tilde_assume!!` and `tilde_observe!!` functions for `InitContext`.

Note that this does not implement almost every other AbstractVarInfo interface function, and
so using this with a different leaf context such as `DefaultContext` will result in errors.

For more information about accumulators, please see the [DynamicPPL documentation on
accumulators](@ref accumulators-overview).
"""
struct OnlyAccsVarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
OnlyAccsVarInfo() = OnlyAccsVarInfo(default_accumulators())
function OnlyAccsVarInfo(accs::NTuple{N,AbstractAccumulator}) where {N}
    return OnlyAccsVarInfo(AccumulatorTuple(accs))
end
function OnlyAccsVarInfo(accs::Vararg{AbstractAccumulator})
    return OnlyAccsVarInfo(AccumulatorTuple(accs))
end

function Base.show(io::IO, ::MIME"text/plain", oavi::OnlyAccsVarInfo)
    printstyled(io, "OnlyAccsVarInfo"; bold=true)
    println(io)
    print(io, " └─ ")
    DynamicPPL.pretty_print(io, oavi.accs, "    ")
    return nothing
end

# Minimal AbstractVarInfo interface
DynamicPPL.getaccs(vi::OnlyAccsVarInfo) = vi.accs
DynamicPPL.setaccs!!(::OnlyAccsVarInfo, accs::AccumulatorTuple) = OnlyAccsVarInfo(accs)
function DynamicPPL.get_transform_strategy(::OnlyAccsVarInfo)
    # OAVI doesn't contain this info, we can't return a sensible value. Hopefully this
    # method doesn't ever get called though.
    return error(
        "get_transform_strategy cannot be implemented for OnlyAccsVarInfo; please specify a transform strategy manually in your call to `init!!`",
    )
end

# This allows us to make use of the main tilde_assume!!(::InitContext) method without
# having to duplicate the code here
@inline function DynamicPPL.setindex_with_dist!!(
    vi::OnlyAccsVarInfo, ::AbstractTransformedValue, ::Distribution, ::VarName, ::Any
)
    return vi
end

"""
    get_vector_values(accs::OnlyAccsVarInfo)

Get the vectorised values from `accs`. This will error if `accs` does not contain a
`VectorValueAccumulator`.

Note that this function is implemented for `OnlyAccsVarInfo`, but not `VarInfo` since that
could be ambiguous (VarInfo stores its own vectorised values!). If you want to extract the
vectorised values from `varinfo.values` where `varinfo isa VarInfo`, you should use
[`DynamicPPL.internal_values_as_vector(varinfo)`](@ref internal_values_as_vector).
"""
function get_vector_values(oavi::OnlyAccsVarInfo)
    return get_vector_values(getacc(oavi, Val(VECTORVAL_ACCNAME)).values)
end
