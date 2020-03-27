"""
    struct Model{S, Targs<:NamedTuple, Tdefaults<:NamedTuple, Tmissings <: Val}
        f::ModelFunction{S}
        args::Targs
        defaults::Tmodelgen
        missings::Tmissings
    end

A `Model` struct with arguments `args`, model generator `modelgen` and
missing data `missings`. `missings` is a `Val` instance, e.g. `Val{(:a, :b)}()`. An
argument in `args` with a value `missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in
`missings` are treated as random variables rather than observations.
"""
struct Model{S, Targs<:NamedTuple, Tdefaults<:NamedTuple, Tmissings<:Val} <: AbstractModel
    args::Targs
    defaults::Tdefaults
    missings::Tmissings
end

function Model{S}(args::NamedTuple, defaults::NamedTuple) where {S}
    missings = getmissing(args)
    Model{S, typeof(args), typeof(defaults), typeof(missings)}(args, defaults, missings)
end

(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())


"""
    getdefaults(::Type{<:Model})

Get a named tuple of the default argument values defined in a `Model` type.
"""
function getdefaults end


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
