struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end

"""
gradient_logp_reverse(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    backend::ADBackend,
)

Computes the value of the log joint of `θ` and its gradient for the model
specified by `(vi, sampler, model)` using reverse-mode AD from the specified `backend`, e.g. `TrackerAD()` which uses `Tracker.jl` or `ZygoteAD()` which uses `Zygote.jl`.
"""
function gradient_logp_reverse(
    backend::ZygoteAD,
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
)
    T = typeof(getlogp(vi))

    # Specify objective function.
    function f(θ)
        new_vi = VarInfo(vi, sampler, θ)
        return getlogp(runmodel!(model, new_vi, sampler))
    end

    # Compute forward and reverse passes.
    l, ȳ = Zygote.pullback(f, θ)
    ∂l∂θ = ȳ(1)[1]

    return l, ∂l∂θ
end
