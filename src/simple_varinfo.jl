struct SimpleVarInfo{NT, T} <: AbstractVarInfo
    θ::NT
    logp::Base.RefValue{T}
end

SimpleVarInfo{T}(θ) where {T<:Real} = SimpleVarInfo{typeof(θ), T}(θ, Ref(zero(T)))
SimpleVarInfo(θ) = SimpleVarInfo{Float64}(θ)

function setlogp!(vi::SimpleVarInfo, logp)
    vi.logp[] = logp
    return vi
end

function acclogp!(vi::SimpleVarInfo, logp)
    vi.logp[] += logp
    return vi
end

getindex(vi::SimpleVarInfo, spl::SampleFromPrior) = vi.θ
getindex(vi::SimpleVarInfo, spl::SampleFromUniform) = vi.θ
getindex(vi::SimpleVarInfo, spl::Sampler) = vi.θ
