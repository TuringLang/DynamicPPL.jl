module DynamicPPLForwardDiffExt

using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems
using ForwardDiff
using SparseArrays: SparseArrays

#
# Model-input provenance tracking
#

# Consider this model:
#
#     @model function demo(y)
#         z = exp(y)
#         z ~ Normal()
#     end
#
# The tilde statement classifies `z` as latent and overwrites `exp(y)`. A user may instead
# have intended to observe the transformed input, so `check_model` should warn that the
# existing value depends on a model input.
#
# To detect this, the check treats all continuous model inputs as a vector `x` and runs the
# model once more with dual numbers carrying a single tangent. Under ordinary dual rules,
# a scalar or structured intermediate `z = f(x)` carries directional derivative data
# `dz = J_f(x) * v` with the same structure. Immediately before a latent tilde statement
# overwrites an existing `z`, a nonzero tangent in any numeric leaf indicates that `z` was
# computed from a model input and may have been intended as an observation. One direction
# suffices because the check needs only this yes-or-no signal, not a full Jacobian or the
# identity of each contributing input.
#
# A finite shared direction can miss dependencies when contributions cancel or a derivative
# is zero. This implementation instead uses NaN tangents as absorbing provenance markers:
# arithmetic cannot cancel them, and multiplication by a zero derivative retains the marker.
# This keeps the existing dual-number rules and the single additional model evaluation.
#
# A dedicated provenance number could instead carry `(value, depends)`. Model inputs would
# start as `(x, true)` and constants as `(c, false)`. Each numeric operation `g` would lift as
#
#     g((x1, d1), ..., (xn, dn)) = (g(x1, ..., xn), d1 || ... || dn)
#
# This still needs only one model pass and constant metadata per scalar, rather than one
# independent direction per input. The rules record data provenance rather than derivatives:
# they preserve a dependency when contributions cancel, or when `g` is locally flat or
# non-differentiable, provided `g` has a rule. The new number would therefore need its own
# numeric and package integrations.
#
# Both designs track only data carried by wrapped numeric values. They cannot retain
# dependencies expressed through control flow or input-derived indexing, and concrete
# conversions or foreign calls may erase the wrapper. Exact tracking for arbitrary Julia
# requires control- and data-flow interpretation.
#
# Libtask.jl's lowered-IR transformation and control-flow analysis provide a relevant
# precedent.

struct InputProvenanceTag end

struct InputProvenanceAccumulator <: DynamicPPL.AbstractAccumulator
    vns::Set{DynamicPPL.VarName}
end
InputProvenanceAccumulator() = InputProvenanceAccumulator(Set{DynamicPPL.VarName}())

function DynamicPPL.accumulator_name(::Type{InputProvenanceAccumulator})
    return DynamicPPL.INPUT_PROVENANCE_ACCNAME
end
function Base.copy(acc::InputProvenanceAccumulator)
    return InputProvenanceAccumulator(copy(acc.vns))
end
DynamicPPL.reset(::InputProvenanceAccumulator) = InputProvenanceAccumulator()
DynamicPPL.split(::InputProvenanceAccumulator) = InputProvenanceAccumulator()
function DynamicPPL.combine(
    acc1::InputProvenanceAccumulator, acc2::InputProvenanceAccumulator
)
    return InputProvenanceAccumulator(union(acc1.vns, acc2.vns))
end
function DynamicPPL.accumulate_assume!!(
    acc::InputProvenanceAccumulator, val, tval, logjac, vn, dist, template
)
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::InputProvenanceAccumulator, dist, val, vn, template
)
    return acc
end

function _dualize_input(x::T) where {T<:AbstractFloat}
    return ForwardDiff.Dual{InputProvenanceTag}(x, oftype(x, NaN))
end

# Laziness marks structural zeros without materializing them.
struct DualizedSparseArray{D,N,A<:SparseArrays.AbstractSparseArray} <: AbstractArray{D,N}
    parent::A
end
function DualizedSparseArray(
    x::A
) where {T,Ti,N,A<:SparseArrays.AbstractSparseArray{T,Ti,N}}
    D = Base.promote_op(_dualize_input, T)
    return DualizedSparseArray{D,N,A}(x)
end
Base.parent(x::DualizedSparseArray) = x.parent
Base.size(x::DualizedSparseArray) = size(parent(x))
Base.IndexStyle(::Type{<:DualizedSparseArray{D,N,A}}) where {D,N,A} = Base.IndexStyle(A)
Base.getindex(x::DualizedSparseArray, I...) = _dualize_input(parent(x)[I...])
SparseArrays.nnz(x::DualizedSparseArray) = SparseArrays.nnz(parent(x))

_dualize_input(x::SparseArrays.AbstractSparseArray) = DualizedSparseArray(x)
_dualize_input(x::AbstractArray) = map(_dualize_input, x)
_dualize_input(x::NamedTuple) = map(_dualize_input, x)
_dualize_input(x::Tuple) = map(_dualize_input, x)
_dualize_input(x) = x

function _has_input_provenance(x::ForwardDiff.Dual{InputProvenanceTag})
    return any(p -> !iszero(p), ForwardDiff.partials(x))
end
function _has_input_provenance(xs::AbstractArray)
    for i in eachindex(xs)
        if isassigned(xs, i) && _has_input_provenance(xs[i])
            return true
        end
    end
    return false
end
_has_input_provenance(xs::Union{Tuple,NamedTuple}) = any(_has_input_provenance, xs)
_has_input_provenance(::Any) = false

function DynamicPPL.check_input_provenance!!(
    vi::DynamicPPL.AbstractVarInfo,
    value::Union{ForwardDiff.Dual{InputProvenanceTag},AbstractArray,Tuple,NamedTuple},
    vn::DynamicPPL.VarName,
)
    accname = Val(DynamicPPL.INPUT_PROVENANCE_ACCNAME)
    _has_input_provenance(value) || return vi

    return DynamicPPL.map_accumulator!!(vi, accname) do acc
        vn in acc.vns && return acc
        vns = copy(acc.vns)
        push!(vns, vn)
        @warn (
            "Variable $(vn) has a value derived from a model input before its tilde " *
            "statement, but it is classified as latent and that value will be " *
            "overwritten. It might be intended as an observation."
        )
        return InputProvenanceAccumulator(vns)
    end
end

function check_input_provenance(rng, model, params)
    args = map(_dualize_input, model.args)
    defaults = map(_dualize_input, model.defaults)
    traced_model = DynamicPPL.Model{DynamicPPL.requires_threadsafe(model)}(
        model.f, args, defaults, model.context
    )
    vi = DynamicPPL.OnlyAccsVarInfo((InputProvenanceAccumulator(),))
    strategy = DynamicPPL.InitFromParams(params, nothing)

    try
        DynamicPPL.init!!(rng, traced_model, vi, strategy, DynamicPPL.UnlinkAll())
    catch err
        err isa InterruptException && rethrow()
        # This is a best-effort debug check. Valid models are not required to accept
        # ForwardDiff dual numbers, so an unsupported trace must not make `check_model` fail.
        return nothing
    end

    return nothing
end

# check if the AD type already has a tag
use_dynamicppl_tag(::ADTypes.AutoForwardDiff{<:Any,Nothing}) = true
use_dynamicppl_tag(::ADTypes.AutoForwardDiff) = false

function (f::DynamicPPL._StanDifferentiableFunction)(
    x::AbstractArray{<:ForwardDiff.Dual{T,V,N}}
) where {T,V,N}
    primal = ForwardDiff.value.(x)
    input_partials = ForwardDiff.partials.(x)
    input_tangents = ntuple(Val(N)) do direction
        map(partials -> partials[direction], input_partials)
    end
    output, output_tangents = DynamicPPL._stan_value_and_pushforward(
        f, primal, input_tangents
    )
    return _stan_dualize(T, output, output_tangents)
end

function _stan_dualize(::Type{T}, output::Real, tangents::NTuple{N}) where {T,N}
    return ForwardDiff.Dual{T}(output, ForwardDiff.Partials(tangents))
end

function _stan_dualize(::Type{T}, output::AbstractArray, tangents::NTuple{N}) where {T,N}
    return map(eachindex(output)) do index
        partials = ForwardDiff.Partials(ntuple(i -> tangents[i][index], Val(N)))
        ForwardDiff.Dual{T}(output[index], partials)
    end
end

function DynamicPPL.tweak_adtype(
    ad::ADTypes.AutoForwardDiff{chunk_size}, ::DynamicPPL.Model, params::AbstractVector
) where {chunk_size}
    # Use DynamicPPL tag to improve stack traces
    # https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
    # NOTE: DifferentiationInterface disables tag checking if the
    # tag inside the AutoForwardDiff type is not nothing. See
    # https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/1df562180bdcc3e91c885aa5f4162a0be2ced850/DifferentiationInterface/ext/DifferentiationInterfaceForwardDiffExt/onearg.jl#L338-L350.
    # So we don't currently need to override ForwardDiff.checktag as well.
    tag = if use_dynamicppl_tag(ad)
        ForwardDiff.Tag(DynamicPPL.DynamicPPLTag(), eltype(params))
    else
        ad.tag
    end

    # Optimise chunk size according to size of model
    chunk = if chunk_size == 0 || chunk_size === nothing
        ForwardDiff.Chunk(params)
    else
        ForwardDiff.Chunk(length(params), chunk_size)
    end

    return ADTypes.AutoForwardDiff(; chunksize=ForwardDiff.chunksize(chunk), tag=tag)
end

end # module
