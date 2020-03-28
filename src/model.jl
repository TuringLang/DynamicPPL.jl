"""
    struct Model{G, Targs<:NamedTuple, Tmissings <: Val}
        args::Targs
        missings::Tmissings
    end

A `Model` struct with arguments `args`, model generator type `G` and
missing data `missings`. `missings` is a `Val` instance, e.g. `Val{(:a, :b)}()`. An
argument in `args` with a value `missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in
`missings` are treated as random variables rather than observations.
"""
struct Model{G, Targs<:NamedTuple, Tmissings<:Val} <: AbstractModel
    args::Targs
    missings::Tmissings
end

Model{G}(args::Targs, missings::Tmissings) where {G, Targs, Tmissings} =
    Model{G, Targs, Tmissings}(args, missings)

(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())


"""
    getgenerator(model::Model)

Get the generator (the function defined by the `@model` macro) of a certain model instance.
"""
function getgenerator end

"""
    getmodeltype(::typeof(modelgen))

Get the associated model type for a model generating function.
"""
function getmodeltype end

"""
    getdefaults(::typeof(modelgen))

Get a named tuple of the default argument values defined for a model defined by a generating function.
"""
function getdefaults end

"""
    getargnames(::typeof(modelgen))

Get a tuple of the argument names of the model defined by a generating function.
"""
function getargnames end


getmissing(model::Model) = model.missings
@generated function getmissing(args::NamedTuple{names, ttuple}) where {names, ttuple}
    length(names) == 0 && return :(Val{()}())
    minds = filter(1:length(names)) do i
        ttuple.types[i] == Missing
    end
    mnames = names[minds]
    return :(Val{$mnames}())
end
