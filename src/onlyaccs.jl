"""
    OnlyAccsVarInfo(accs...)

`OnlyAccsVarInfo` is a wrapper around a tuple of accumulators.

It implements the minimal `AbstractVarInfo` interface needed for accumulation and
latent-value evaluation with an explicit context.

This does not store parameter values. Supply them through the evaluation context.

For more information about accumulators, please see the [DynamicPPL documentation on
accumulators](@ref accumulators-overview).
"""
struct OnlyAccsVarInfo{Accs<:AccumulatorTuple} <: AbstractVarInfo
    accs::Accs
end
Base.copy(vi::OnlyAccsVarInfo) = OnlyAccsVarInfo(copy(vi.accs))
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
    vi::OnlyAccsVarInfo, ::TransformedValue, ::Distribution, ::VarName, ::Any
)
    return vi
end

"""
    get_vector_values(accs::OnlyAccsVarInfo)

Get a `VarNamedTuple` containing vectorised values from `accs`. This will error if `accs`
does not contain a `VectorValueAccumulator`.

Note that this function is implemented for `OnlyAccsVarInfo`, but not `VarInfo` since that
could be ambiguous (VarInfo stores its own vectorised values!). If you want to extract the
vectorised values from `varinfo.values` where `varinfo isa VarInfo`, you should use
[`DynamicPPL.internal_values_as_vector(varinfo)`](@ref internal_values_as_vector).
"""
function get_vector_values(oavi::OnlyAccsVarInfo)
    return getacc(oavi, Val(VECTORVAL_ACCNAME)).values
end
