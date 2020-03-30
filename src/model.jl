"""
    struct Model{G, argnames, missings, Targs}
        args::NamedTuple{argnames, Targs}
    end

A `Model` struct with model generator type `G`, arguments names `argnames`, arguments types `Targs`,
and missing arguments `missings`. `argnames` and `missings` are tuples of symbols, e.g. `(:a,
:b)`.  An argument with a type of `Missing` will be in `missings` by default.  However, in
non-traditional use-cases `missings` can be defined differently.  All variables in `missings` are
treated as random variables rather than observations.

# Example

```julia
julia> Model{typeof(gdemo)}((x = 1.0, y = 2.0))
Model{typeof(gdemo),(),(:x, :y),Tuple{Float64,Float64}}((x = 1.0, y = 2.0))

julia> Model{typeof(gdemo), (:y,)}((x = 1.0, y = 2.0))
Model{typeof(gdemo),(:y,),(:x, :y),Tuple{Float64,Float64}}((x = 1.0, y = 2.0))
```
"""
struct Model{G, argnames, missings, Targs} <: AbstractModel
    args::NamedTuple{argnames, Targs}

    Model{G, missings}(args::NamedTuple{argnames, Targs}) where {G, argnames, missings, Targs} =
        new{G, argnames, missings, Targs}(args)
end

@generated function Model{G}(args::NamedTuple{argnames, Targs}) where {G, argnames, Targs}
    missings = Tuple(name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing)
    return :(Model{G, $missings}(args))
end


(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())


"""
    getargnames(model::Model)

Get a tuple of the argument names of the `model`.
"""
getargnames(model::Model) = getargnames(typeof(model))
getargnames(::Type{<:Model{_G, argnames} where {_G}}) where {argnames} = argnames

@generated function inargnames(::Val{s}, ::Model{_G, argnames}) where {s, _G, argnames}
    return s in argnames
end


"""
    getmissings(model::Model)

Get a tuple of the names of the missing arguments of the `model`.
"""
getmissings(model::Model{_G, _a, missings}) where {missings, _G, _a} = missings

getmissing(model::Model) = getmissings(model)
@deprecate getmissing(model) getmissings(model)

@generated function inmissings(::Val{s}, ::Model{_G, _a, missings}) where {s, missings, _G, _a}
    return s in missings
end


"""
    getgenerator(model::Model)

Get the generator (the function defined by the `@model` macro) of a certain model instance.
"""
getgenerator(model::Model) = getgenerator(typeof(model))


"""
    getdefaults(model::Model)

Get a named tuple of the default argument values defined for a model defined by a generating function.
"""
getdefaults(model::Model) = getdefaults(typeof(model))


"""
    getmodeltype(::typeof(modelgen))

Get the associated model type for a model generator (the function defined by the `@model` macro).
"""
getmodeltype(model::Model) = typeof(model)
