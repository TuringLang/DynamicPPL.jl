"""
    struct Model{F, Targs <: NamedTuple, Tmodelgen, Tmissings <: Val}
        f::F
        args::Targs
        modelgen::Tmodelgen
        missings::Tmissings
    end

A `Model` struct with arguments `args`, inner function `f`, model generator `modelgen` and
missing data `missings`. `missings` is a `Val` instance, e.g. `Val{(:a, :b)}()`. An
argument in `args` with a value `missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in
`missings` are treated as random variables rather than observations.
"""
struct Model{F, Targs <: NamedTuple, Tmodelgen, Tmissings <: Val} <: AbstractModel
    f::F
    args::Targs
    modelgen::Tmodelgen
    missings::Tmissings
end
Model(f, args::NamedTuple, modelgen) = Model(f, args, modelgen, getmissing(args))
(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())
(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)

getmissing(model::Model) = model.missings
@generated function getmissing(args::NamedTuple{names, ttuple}) where {names, ttuple}
    length(names) == 0 && return :(Val{()}())
    minds = filter(1:length(names)) do i
        ttuple.types[i] == Missing
    end
    mnames = names[minds]
    return :(Val{$mnames}())
end
