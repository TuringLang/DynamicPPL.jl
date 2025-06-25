"""
    Submodel{M,AutoPrefix}

A wrapper around a model, plus a flag indicating whether it should be automatically
prefixed with the left-hand variable in a `~` statement.
"""
struct Submodel{M,AutoPrefix}
    model::M
end

"""
    to_submodel(model::Model[, auto_prefix::Bool])

Return a model wrapper indicating that it is a sampleable model over the return-values.

This is mainly meant to be used on the right-hand side of a `~` operator to indicate that
the model can be sampled from but not necessarily evaluated for its log density.

!!! warning
    Note that some other operations that one typically associate with expressions of the form
    `left ~ right` such as [`condition`](@ref), will also not work with `to_submodel`.

!!! warning
    To avoid variable names clashing between models, it is recommended to leave the argument `auto_prefix` equal to `true`.
    If one does not use automatic prefixing, then it's recommended to use [`prefix(::Model, input)`](@ref) explicitly, i.e. `to_submodel(prefix(model, @varname(my_prefix)))`

# Arguments
- `model::Model`: the model to wrap.
- `auto_prefix::Bool`: whether to automatically prefix the variables in the model using the left-hand
    side of the `~` statement. Default: `true`.

# Examples

## Simple example
```jldoctest submodel-to_submodel; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y)
            a ~ to_submodel(demo1(x))
            return y ~ Uniform(0, a)
       end;
```

When we sample from the model `demo2(missing, 0.4)` random variable `x` will be sampled:
```jldoctest submodel-to_submodel
julia> vi = VarInfo(demo2(missing, 0.4));

julia> @varname(a.x) in keys(vi)
true
```

The variable `a` is not tracked. However, it will be assigned the return value of `demo1`,
and can be used in subsequent lines of the model, as shown above.
```jldoctest submodel-to_submodel
julia> @varname(a) in keys(vi)
false
```

We can check that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-to_submodel
julia> x = vi[@varname(a.x)];

julia> getlogjoint(vi) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
true
```

## Without automatic prefixing
As mentioned earlier, by default, the `auto_prefix` argument specifies whether to automatically
prefix the variables in the submodel. If `auto_prefix=false`, then the variables in the submodel
will not be prefixed.
```jldoctest submodel-to_submodel-prefix; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2_no_prefix(x, z)
            a ~ to_submodel(demo1(x), false)
            return z ~ Uniform(-a, 1)
       end;

julia> vi = VarInfo(demo2_no_prefix(missing, 0.4));

julia> @varname(x) in keys(vi)  # here we just use `x` instead of `a.x`
true
```
However, not using prefixing is generally not recommended as it can lead to variable name clashes
unless one is careful. For example, if we're re-using the same model twice in a model, not using prefixing
will lead to variable name clashes: However, one can manually prefix using the [`prefix(::Model, input)`](@ref):
```jldoctest submodel-to_submodel-prefix
julia> @model function demo2(x, y, z)
            a ~ to_submodel(prefix(demo1(x), :sub1), false)
            b ~ to_submodel(prefix(demo1(y), :sub2), false)
            return z ~ Uniform(-a, b)
       end;

julia> vi = VarInfo(demo2(missing, missing, 0.4));

julia> @varname(sub1.x) in keys(vi)
true

julia> @varname(sub2.x) in keys(vi)
true
```

Variables `a` and `b` are not tracked, but are assigned the return values of the respective
calls to `demo1`:
```jldoctest submodel-to_submodel-prefix
julia> @varname(a) in keys(vi)
false

julia> @varname(b) in keys(vi)
false
```

We can check that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-to_submodel-prefix
julia> sub1_x = vi[@varname(sub1.x)];

julia> sub2_x = vi[@varname(sub2.x)];

julia> logprior = logpdf(Normal(), sub1_x) + logpdf(Normal(), sub2_x);

julia> loglikelihood = logpdf(Uniform(-1 - abs(sub1_x), 1 + abs(sub2_x)), 0.4);

julia> getlogjoint(vi) ≈ logprior + loglikelihood
true
```

## Usage as likelihood is illegal

Note that it is illegal to use a `to_submodel` model as a likelihood in another model:

```jldoctest submodel-to_submodel-illegal; setup=:(using Distributions)
julia> @model inner() = x ~ Normal()
inner (generic function with 2 methods)

julia> @model illegal_likelihood() = a ~ to_submodel(inner())
illegal_likelihood (generic function with 2 methods)

julia> model = illegal_likelihood() | (a = 1.0,);
julia> model()
ERROR: ArgumentError: `~` with a model on the right-hand side of an observe statement is not supported
[...]
```
"""
to_submodel(m::Model, auto_prefix::Bool=true) = Submodel{typeof(m),auto_prefix}(m)

# When automatic prefixing is used, the submodel itself doesn't carry the
# prefix, as the prefix is obtained from the LHS of `~` (whereas the submodel
# is on the RHS). The prefix can only be obtained in `tilde_assume!!`, and then
# passed into this function.
#
# `parent_context` here refers to the context of the model that contains the
# submodel.
function _evaluate!!(
    submodel::Submodel{M,AutoPrefix},
    vi::AbstractVarInfo,
    parent_context::AbstractContext,
    left_vn::VarName,
) where {M<:Model,AutoPrefix}
    # First, we construct the context to be used when evaluating the submodel. There
    # are several considerations here:
    # (1) We need to apply an appropriate PrefixContext when evaluating the submodel, but
    # _only_ if automatic prefixing is supposed to be applied.
    submodel_context_prefixed = if AutoPrefix
        PrefixContext(left_vn, submodel.model.context)
    else
        submodel.model.context
    end

    # (2) We need to respect the leaf-context of the parent model. This, unfortunately,
    # means disregarding the leaf-context of the submodel.
    submodel_context = setleafcontext(
        submodel_context_prefixed, leafcontext(parent_context)
    )

    # (3) We need to use the parent model's context to wrap the whole thing, so that
    # e.g. if the user conditions the parent model, the conditioned variables will be
    # correctly picked up when evaluating the submodel.
    eval_context = setleafcontext(parent_context, submodel_context)

    # (4) Finally, we need to store that context inside the submodel.
    model = contextualize(submodel.model, eval_context)

    # Once that's all set up nicely, we can just _evaluate!! the wrapped model. This
    # returns a tuple of submodel.model's return value and the new varinfo.
    return _evaluate!!(model, vi)
end
