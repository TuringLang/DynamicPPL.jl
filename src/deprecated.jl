# https://invenia.github.io/blog/2022/06/17/deprecating-in-julia/

Base.@deprecate pointwise_loglikelihoods(model::Model, chain, keytype) pointwise_logdensities(
    model::Model, LikelihoodContext(), chain, keytype)

Base.@deprecate pointwise_loglikelihoods(
    model::Model, varinfo::AbstractVarInfo) pointwise_logdensities(
    model::Model, varinfo, LikelihoodContext())

