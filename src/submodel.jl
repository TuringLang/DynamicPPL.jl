"""
    Submodel{M,AutoPrefix}

A wrapper around a model, plus a flag indicating whether it should be automatically
prefixed with the left-hand variable in a `~` statement.
"""
struct Submodel{M,AutoPrefix}
    model::M
end

# ----------------------
# Constructing submodels
# ----------------------

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

```jldoctest submodel-to_submodel
julia> using DynamicPPL, Distributions

julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y)
            a ~ to_submodel(demo1(x))
            return y ~ Uniform(0, a)
       end;
```

When we sample from the model `demo2(missing, 0.4)` the random variable `x` will be sampled, but
it will be prefixed with `a` (the left-hand side of the tilde):

```jldoctest submodel-to_submodel
julia> model = demo2(missing, 0.4);

julia> haskey(rand(model), @varname(a.x))
true
```

The variable `a` will be assigned the return value of `demo1`, and can be used in subsequent
lines of the model, e.g. in the definition of `y` above.

We can verify that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-to_submodel
julia> accs = setacc!!(OnlyAccsVarInfo(), RawValueAccumulator(false));

julia> _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll());

julia> x = get_raw_values(accs)[@varname(a.x)];

julia> getlogjoint(accs) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
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

julia> model = demo2_no_prefix(missing, 0.4);

julia> haskey(rand(model), @varname(x))  # here we just use `x` instead of `a.x`
true
```
However, not using prefixing is generally not recommended as it can lead to variable name
clashes unless one is careful. For example, if the same submodel is used multiple times in a
model, not using prefixing will lead to variable name clashes.

One can manually specify a prefix using [`prefix(::Model, prefix_varname)`](@ref):

```jldoctest submodel-to_submodel-prefix
julia> @model function demo2(x, y, z)
            a ~ to_submodel(prefix(demo1(x), @varname(sub1)), false)
            b ~ to_submodel(prefix(demo1(y), @varname(sub2)), false)
            return z ~ Uniform(-a, b)
       end;

julia> model = demo2(missing, missing, 0.4);

julia> haskey(rand(model), @varname(sub1.x))
true

julia> haskey(rand(model), @varname(sub2.x))
true
```
"""
to_submodel(m::Model, auto_prefix::Bool=true) = Submodel{typeof(m),auto_prefix}(m)

# ---------------------------
# Submodels in tilde-pipeline
# ---------------------------

"""
    DynamicPPL.tilde_assume!!(
        context::AbstractContext,
        right::DynamicPPL.Submodel,
        vn::VarName,
        ::Any,
        vi::AbstractVarInfo
    )

Evaluate the submodel with the given context.
"""
function tilde_assume!!(
    context::AbstractContext,
    right::DynamicPPL.Submodel,
    vn::VarName,
    ::Any,
    vi::AbstractVarInfo,
)
    return _evaluate!!(right, vi, context, vn)
end

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

function tilde_observe!!(
    context::AbstractContext,
    right::DynamicPPL.Submodel,
    left::Any,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    # TODO(penelopeysm) This is VERY BAD. See
    # https://github.com/TuringLang/DynamicPPL.jl/issues/1246.
    #
    # We need a much more principled way of dealing with this. The problem is that, if we
    # have
    #
    # @model inner() = a ~ Normal()
    # @model function f()
    #    x ~ to_submodel(inner())
    # end
    # model = f() | (@varname(x.a) => 2.0)
    #
    # and a user conditions the top-level model on `x.a` (for example), then when we check
    # whether `x` is conditioned, we will find that it indeed is (since the conditioned
    # values will have `values.data.x` pointing to a VNT). That sends us down the path of
    # tilde_observe!!, so we HAVE to deal with this by calling evaluate.
    #
    # What we actually want to forbid is conditioning on the RETURN VALUE. That is, we don't
    # want someone to think that they can do
    #
    # model = f() | (@varname(x) => 3.0)
    #
    # or indeed
    #
    # @model function f2(x)
    #     x ~ to_submodel(inner())
    # end
    # model2 = f2(3.0)
    # 
    # These are the cases that we want to ban. BUT WE HAVE NO WAY OF FIGURING OUT WHICH ONE
    # THE USER MEANT ---- BECAUSE WE LUMP THE RETURN VALUE AND LATENTS INTO ONE THING.
    # This is REALLY, really frustrating.
    #
    # What we do here is to just evaluate the submodel so that we handle the first case
    # above correctly. The other cases USED to error; however, now they will work (and the
    # submodel will be evaluated, but the value of `x` will be ignored). That is probably
    # not what the user wants, but hey, it'll make tests pass.
    return _evaluate!!(right, vi, context, vn)
end
function tilde_observe!!(
    ::AbstractContext, ::DynamicPPL.Submodel, left, ::Nothing, template, ::AbstractVarInfo
)
    throw(ArgumentError("`x ~ to_submodel(...)` is not supported when `x` is a literal"))
end
