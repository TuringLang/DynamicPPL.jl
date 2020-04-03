"""
    struct ModelGen{G, defaultnames, Tdefaults}
        generator::G
        defaults::Tdefaults
    end

A `ModelGen` struct with model generator function of type `G`, and default arguments `defaultnames`
with values `Tdefaults`.
"""
struct ModelGen{G, argnames, defaultnames, Tdefaults}
    generator::G
    defaults::NamedTuple{defaultnames, Tdefaults}

    function ModelGen{argnames}(
        generator::G,
        defaults::NamedTuple{defaultnames, Tdefaults}
    ) where {G, argnames, defaultnames, Tdefaults}
        return new{G, argnames, defaultnames, Tdefaults}(generator, defaults)
    end
end

(m::ModelGen)(args...; kwargs...) = m.generator(args...; kwargs...)


"""
    getdefaults(modelgen::ModelGen)

Get a named tuple of the default argument values defined for a model defined by a generating function.
"""
getdefaults(modelgen::ModelGen) = modelgen.defaults

"""
    getargnames(modelgen::ModelGen)

Get a tuple of the argument names of the `modelgen`.
"""
getargnames(model::ModelGen{_G, argnames}) where {argnames, _G} = argnames



"""
    struct Model{F, argnames, Targs, missings}
        f::F
        args::NamedTuple{argnames, Targs}
        modelgen::Tgen
    end

A `Model` struct with model evaluation function of type `F`, arguments names `argnames`, arguments
types `Targs`, missing arguments `missings`, and corresponding model generator. `argnames` and
`missings` are tuples of symbols, e.g. `(:a, :b)`.  An argument with a type of `Missing` will be in
`missings` by default.  However, in non-traditional use-cases `missings` can be defined differently.
All variables in `missings` are treated as random variables rather than observations.

# Example

```julia
julia> Model(f, (x = 1.0, y = 2.0))
Model{typeof(f),(),(:x, :y),Tuple{Float64,Float64}}((x = 1.0, y = 2.0))

julia> Model{(:y,)}(f, (x = 1.0, y = 2.0))
Model{typeof(f),(:y,),(:x, :y),Tuple{Float64,Float64}}((x = 1.0, y = 2.0))
```
"""
struct Model{F, argnames, Targs, missings, Tgen} <: AbstractModel
    f::F
    args::NamedTuple{argnames, Targs}
    modelgen::Tgen

    """
        Model{missings}(f, args::NamedTuple, modelgen::ModelGen)

    Create a model with evalutation function `f` and missing arguments overwritten by `missings`.
    """
    function Model{missings}(
        f::F,
        args::NamedTuple{argnames, Targs},
        modelgen::Tgen
    ) where {missings, F, argnames, Targs, Tgen<:ModelGen}
        return new{F, argnames, Targs, missings, Tgen}(f, args, modelgen)
    end
end

"""
    Model(f, args::NamedTuple, modelgen::ModelGen)

    Create a model with evalutation function `f` and missing arguments deduced from `args`.
"""
@generated function Model(
    f::F,
    args::NamedTuple{argnames, Targs},
    modelgen::ModelGen{_G, argnames}
) where {F, argnames, Targs, _G}
    missings = Tuple(name for (name, typ) in zip(argnames, Targs.types) if typ <: Missing)
    return :(Model{$missings}(f, args, modelgen))
end


"""
    Model{missings}(modelgen::ModelGen, args::NamedTuple)

Create a copy of the model described by `modelgen(args...)`, with missing arguments 
overwritten by `missings`.
"""
function Model{missings}(
    modelgen::ModelGen,
    args::NamedTuple{argnames, Targs}
) where {missings, argnames, Targs}
    model = modelgen(args...)
    return Model{missings}(model.f, args, modelgen)
end


function (model::Model)(
    vi::AbstractVarInfo=VarInfo(),
    spl::AbstractSampler=SampleFromPrior(),
    ctx::AbstractContext=DefaultContext()
)
    return model.f(model, vi, spl, ctx)
end


"""
    runmodel!(model::Model, vi::AbstractVarInfo[, spl::AbstractSampler, ctx::AbstractContext])

Sample from `model` using the sampler `spl` storing the sample and log joint probability in `vi`.
Resets the `vi` and increases `spl`s `state.eval_num`.
"""
function runmodel!(
    model::Model,
    vi::AbstractVarInfo,
    spl::AbstractSampler=SampleFromPrior(),
    ctx::AbstractContext=DefaultContext()
)
    setlogp!(vi, 0)
    if has_eval_num(spl)
        spl.state.eval_num += 1
    end
    model(vi, spl, ctx)
    return vi
end


"""
    getargnames(model::Model)

Get a tuple of the argument names of the `model`.
"""
getargnames(model::Model{_F, argnames}) where {argnames, _F} = argnames


"""
    getmissings(model::Model)

Get a tuple of the names of the missing arguments of the `model`.
"""
getmissings(model::Model{_F, _a, _T, missings}) where {missings, _F, _a, _T} = missings

getmissing(model::Model) = getmissings(model)
@deprecate getmissing(model) getmissings(model)


"""
    getgenerator(model::Model)

Get the model generator associated with `model`.
"""
getgenerator(model::Model) = model.modelgen
