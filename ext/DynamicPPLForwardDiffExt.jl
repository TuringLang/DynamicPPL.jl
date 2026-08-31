module DynamicPPLForwardDiffExt

using DynamicPPL: ADTypes, DynamicPPL, LogDensityProblems
using ForwardDiff

struct InputDependencyTag end

struct InputDependencyAccumulator <: DynamicPPL.AbstractAccumulator
    vns::Set{DynamicPPL.VarName}
end
InputDependencyAccumulator() = InputDependencyAccumulator(Set{DynamicPPL.VarName}())

function DynamicPPL.accumulator_name(::Type{InputDependencyAccumulator})
    return DynamicPPL.INPUT_DEPENDENCY_ACCNAME
end
function Base.copy(acc::InputDependencyAccumulator)
    return InputDependencyAccumulator(copy(acc.vns))
end
DynamicPPL.reset(::InputDependencyAccumulator) = InputDependencyAccumulator()
DynamicPPL.split(::InputDependencyAccumulator) = InputDependencyAccumulator()
function DynamicPPL.combine(
    acc1::InputDependencyAccumulator, acc2::InputDependencyAccumulator
)
    return InputDependencyAccumulator(union(acc1.vns, acc2.vns))
end
function DynamicPPL.accumulate_assume!!(
    acc::InputDependencyAccumulator, val, tval, logjac, vn, dist, template
)
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::InputDependencyAccumulator, dist, val, vn, template
)
    return acc
end

# We only need to know whether the model inputs affect a value, not which input does so.
# Seed every input in the same direction to keep this to one ForwardDiff pass.
function _dualize_input(x::T) where {T<:AbstractFloat}
    return ForwardDiff.Dual{InputDependencyTag}(x, one(x))
end
_dualize_input(x::AbstractArray) = map(_dualize_input, x)
_dualize_input(x::NamedTuple) = map(_dualize_input, x)
_dualize_input(x::Tuple) = map(_dualize_input, x)
_dualize_input(x) = x

function _has_input_dependency(x::ForwardDiff.Dual{InputDependencyTag})
    return any(p -> !iszero(p), ForwardDiff.partials(x))
end
function _has_input_dependency(xs::AbstractArray)
    for i in eachindex(xs)
        if isassigned(xs, i) && _has_input_dependency(xs[i])
            return true
        end
    end
    return false
end
_has_input_dependency(xs::Union{Tuple,NamedTuple}) = any(_has_input_dependency, xs)
_has_input_dependency(::Any) = false

function _check_input_dependency!!(vi, value, vn)
    accname = Val(DynamicPPL.INPUT_DEPENDENCY_ACCNAME)
    _has_input_dependency(value) || return vi

    return DynamicPPL.map_accumulator!!(vi, accname) do acc
        vn in acc.vns && return acc
        vns = copy(acc.vns)
        push!(vns, vn)
        return InputDependencyAccumulator(vns)
    end
end

function DynamicPPL.check_input_dependency!!(
    vi::DynamicPPL.AbstractVarInfo,
    value::Union{ForwardDiff.Dual{InputDependencyTag},AbstractArray,Tuple,NamedTuple},
    vn::DynamicPPL.VarName,
)
    return _check_input_dependency!!(vi, value, vn)
end

function check_input_dependencies(rng, model, params)
    args = map(_dualize_input, model.args)
    defaults = map(_dualize_input, model.defaults)
    traced_model = DynamicPPL.Model{DynamicPPL.requires_threadsafe(model)}(
        model.f, args, defaults, model.context
    )
    vi = DynamicPPL.OnlyAccsVarInfo((InputDependencyAccumulator(),))
    strategy = DynamicPPL.InitFromParams(params, nothing)

    try
        _, vi = DynamicPPL.init!!(rng, traced_model, vi, strategy, DynamicPPL.UnlinkAll())
    catch err
        err isa InterruptException && rethrow()
        # This is a best-effort debug check. Valid models are not required to accept
        # ForwardDiff dual numbers, so an unsupported trace must not make `check_model` fail.
        return nothing
    end

    acc = DynamicPPL.getacc(vi, Val(DynamicPPL.INPUT_DEPENDENCY_ACCNAME))
    for vn in sort!(collect(acc.vns); by=string)
        @warn (
            "Variable $(vn) has a value derived from a model input before its tilde " *
            "statement, but it is classified as latent and that value will be " *
            "overwritten. It might be intended as an observation."
        )
    end
    return nothing
end

# check if the AD type already has a tag
use_dynamicppl_tag(::ADTypes.AutoForwardDiff{<:Any,Nothing}) = true
use_dynamicppl_tag(::ADTypes.AutoForwardDiff) = false

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
