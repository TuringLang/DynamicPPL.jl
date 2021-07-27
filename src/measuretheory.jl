using MeasureTheory: MeasureTheory

# src/compiler.jl
# Allow `AbstractMeasure` on the RHS of `~`.
check_tilde_rhs(x::MeasureTheory.AbstractMeasure) = x

# src/utils.jl
# Linearization.
vectorize(d::MeasureTheory.AbstractMeasure, x::Real) = [x]
vectorize(d::MeasureTheory.AbstractMeasure, x::AbstractArray{<:Real}) = copy(vec(x))

function reconstruct(d::MeasureTheory.AbstractMeasure, x::AbstractVector{<:Real})
    return _reconstruct(d, x, MeasureTheory.sampletype(d))
end

# TODO: Higher dims. What to do? Do we have access to size, e.g. for `LKJ` we should have?
function _reconstruct(
    d::MeasureTheory.AbstractMeasure, x::AbstractVector{<:Real}, ::Type{<:Real}
)
    return x[1]
end
function _reconstruct(
    d::MeasureTheory.AbstractMeasure,
    x::AbstractVector{<:Real},
    ::Type{<:AbstractVector{<:Real}},
)
    return x
end

# src/context_implementations.jl
# assume
function assume(dist::MeasureTheory.AbstractMeasure, vn::VarName, vi)
    r = vi[vn]
    # TODO: Transformed variables.
    return r, MeasureTheory.logdensity(dist, r)
end

function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::MeasureTheory.AbstractMeasure,
    vn::VarName,
    vi,
)
    if haskey(vi, vn)
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if sampler isa SampleFromUniform || is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = init(rng, dist, sampler)
            vi[vn] = vectorize(dist, r)
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            r = vi[vn]
        end
    else
        r = init(rng, dist, sampler)
        push!(vi, vn, r, dist, sampler)
        settrans!(vi, false, vn)
    end

    # TODO: Transformed variables.
    return r, MeasureTheory.logdensity(dist, r)
end

# observe
function observe(right::MeasureTheory.AbstractMeasure, left, vi)
    increment_num_produce!(vi)
    return MeasureTheory.logdensity(right, left)
end
