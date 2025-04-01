@deprecate generated_quantities(model, params) returned(model, params)

Base.@deprecate VarInfo(
    rng::Random.AbstractRNG,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
) TypedVarInfo(rng, model, sampler, context)
Base.@deprecate VarInfo(
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
) TypedVarInfo(model, sampler, context)
Base.@deprecate VarInfo(model::Model, context::AbstractContext) TypedVarInfo(model, context)
