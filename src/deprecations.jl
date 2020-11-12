@deprecate getmissing(model) getmissings(model)

# `@deprecate` doesn't work with qualified function names,
# so we use the following hack
const _base_in = Base.in
@deprecate _base_in(vn::VarName, space::Tuple) inspace(vn, space)

@deprecate elementwise_loglikelihoods(
    model::Model, chain,
) pointwise_loglikelihoods(
    model, chain, String,
)
@deprecate elementwise_loglikelihoods(
    model::Model, chain, ::Type{T},
) where {T} pointwise_loglikelihoods(
    model, chain, T,
)
@deprecate elementwise_loglikelihoods(
    model::Model, varinfo::AbstractVarInfo,
) pointwise_loglikelihoods(
    model, varinfo,
)
