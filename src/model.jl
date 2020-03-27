"""
    struct Model{S, Targs<:NamedTuple, Tmissings <: Val}
        args::Targs
        missings::Tmissings
    end

A `Model` struct with arguments `args`, model generator `modelgen` and
missing data `missings`. `missings` is a `Val` instance, e.g. `Val{(:a, :b)}()`. An
argument in `args` with a value `missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in
`missings` are treated as random variables rather than observations.
"""
struct Model{S, Targs<:NamedTuple, Tmissings<:Val} <: AbstractModel
    args::Targs
    missings::Tmissings
end

(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())


"""
    getdefaults(model)

Get a named tuple of the default argument values defined in a `Model` type.
"""
getdefaults(model::Model) = getdefaults(typeof(model))


getargtype(::Type{<:Model{S, Targs}}) where {S, Targs} = Targs

getmissing(model::Model) = model.missings
@generated function getmissing(args::NamedTuple{names, ttuple}) where {names, ttuple}
    length(names) == 0 && return :(Val{()}())
    minds = filter(1:length(names)) do i
        ttuple.types[i] == Missing
    end
    mnames = names[minds]
    return :(Val{$mnames}())
end
