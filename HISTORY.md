# DynamicPPL Changelog

## 0.35.0

**Breaking**

### Remove indexing by samplers

This release removes the feature of `VarInfo` where it kept track of which variable was associated with which sampler. This means removing all user-facing methods where `VarInfo`s where being indexed with samplers. In particular,

  - `link` and `invlink`, and their `!!` versions, no longer accept a sampler as an argument to specify which variables to (inv)link. The `link(varinfo, model)` methods remain in place, and as a new addition one can give a `Tuple` of `VarName`s to (inv)link only select variables, as in `link(varinfo, varname_tuple, model)`.
  - `set_retained_vns_del_by_spl!` has been replaced by `set_retained_vns_del!` which applies to all variables.
  - `getindex`, `setindex!`, and `setindex!!` no longer accept samplers as arguments
  - `unflatten` no longer accepts a sampler as an argument
  - `eltype(::VarInfo)` no longer accepts a sampler as an argument
  - `keys(::VarInfo)` no longer accepts a sampler as an argument
  - `VarInfo(::VarInfo, ::Sampler, ::AbstactVector)` no longer accepts the sampler argument.

### Reverse prefixing order

  - For submodels constructed using `to_submodel`, the order in which nested prefixes are applied has been changed.
    Previously, the order was that outer prefixes were applied first, then inner ones.
    This version reverses that.
    To illustrate:
    
    ```julia
    using DynamicPPL, Distributions
    
    @model function subsubmodel()
        return x ~ Normal()
    end
    
    @model function submodel()
        x ~ to_submodel(prefix(subsubmodel(), :c), false)
        return x
    end
    
    @model function parentmodel()
        x1 ~ to_submodel(prefix(submodel(), :a), false)
        return x2 ~ to_submodel(prefix(submodel(), :b), false)
    end
    
    keys(VarInfo(parentmodel()))
    ```
    
    Previously, the final line would return the variable names `c.a.x` and `c.b.x`.
    With this version, it will return `a.c.x` and `b.c.x`, which is more intuitive.
    (Note that this change brings `to_submodel`'s behaviour in line with the now-deprecated `@submodel` macro.)
    
    This change also affects sampling in Turing.jl.

## 0.34.2

  - Fixed bugs in ValuesAsInModelContext as well as DebugContext where underlying PrefixContexts were not being applied.
    From a user-facing perspective, this means that for models which use manually prefixed submodels, e.g.
    
    ```julia
    using DynamicPPL, Distributions
    
    @model inner() = x ~ Normal()
    
    @model function outer()
        x1 ~ to_submodel(prefix(inner(), :a), false)
        return x2 ~ to_submodel(prefix(inner(), :b), false)
    end
    ```
    
    will: (1) no longer error when sampling due to `check_model_and_trace`; and (2) contain both submodel's variables in the resulting chain (the behaviour before this patch was that the second `x` would override the first `x`).

  - More broadly, implemented a general `prefix(ctx::AbstractContext, ::VarName)` which traverses the context tree in `ctx` to apply all necessary prefixes. This was a necessary step in fixing the above issues, but it also means that `prefix` is now capable of handling context trees with e.g. multiple prefixes at different levels of nesting.

## 0.34.1

  - Fix an issue that prevented merging two VarInfos if they had different dimensions for a variable.

  - Upper bound the compat version of KernelAbstractions to work around an issue in determining the right VarInfo type to use.

## 0.34.0

**Breaking**

  - `rng` argument removed from `values_as_in_model`, and `varinfo` made non-optional. This means that the only signatures allowed are
    
    ```
    values_as_in_model(::Model, ::Bool, ::AbstractVarInfo)
    values_as_in_model(::Model, ::Bool, ::AbstractVarInfo, ::AbstractContext)
    ```
    
    If you aren't using this function (it's probably only used in Turing.jl) then this won't affect you.

## 0.33.1

Reworked internals of `condition` and `decondition`.
There are no changes to the public-facing API, but internally you can no longer use `condition` and `decondition` on an `AbstractContext`, you can only use it on a `DynamicPPL.Model`. If you want to modify a context, use `ConditionContext` and `decondition_context`.

## 0.33.0

**Breaking**

  - `values_as_in_model()` now requires an extra boolean parameter, specifying whether variables on the lhs of `:=` statements are to be included in the resulting `OrderedDict` of values.
    The type signature is now `values_as_in_model([rng,] model, include_colon_eq::Bool [, varinfo, context])`

**Other**

  - Moved the implementation of `predict` from Turing.jl to DynamicPPL.jl; the user-facing behaviour is otherwise the same
  - Improved error message when a user tries to initialise a model with parameters that don't correspond strictly to the underlying VarInfo used
