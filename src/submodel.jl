"""
    Submodel{M}

Wrap a model whose variables are prefixed by its tilde address.
"""
struct Submodel{M}
    model::M
end

"""
    to_distribution(model)

Convert `model` to a distribution for use on the right-hand side of `~`.

In `variables ~ to_distribution(model)`, `variables` receives the latent variables
represented by the resulting distribution. The concrete representation and density depend
on the method for `typeof(model)`. This differs from [`to_submodel`](@ref), which assigns a
wrapped model's return value to the left-hand side and records its latent variables
separately.
"""
function to_distribution end

abstract type _StanDifferentiableFunction end
(f::_StanDifferentiableFunction)(x::AbstractArray{<:Real}) = _stan_value(f, x)

function _stan_value end
function _stan_value_and_pushforward end
function _stan_value_and_pullback end

# ----------------------
# Constructing submodels
# ----------------------

"""
    to_submodel(model::Model[, auto_prefix::Bool])

Wrap `model` for use on the right-hand side of `~`.

In `value ~ to_submodel(model)`, `model` is evaluated, its return value is assigned to
`value`, and its latent variables are recorded separately in the surrounding trace. Their
names are prefixed with the left-hand side: a latent variable `x` becomes `value.x`. This
differs from [`to_distribution`](@ref), which assigns the represented latent variables
themselves to the left-hand side.

Conceptually, `to_submodel(model)` is a `returned_value(model)` wrapper: its value is the
model's return value, not its latent variables.

`Submodel` is not a `Distribution`; it provides this tilde behavior but no standalone
`logpdf` method.

!!! warning
    Operations normally associated with `left ~ right`, such as [`condition`](@ref), do not
    necessarily work with `to_submodel`.

The optional second argument accepts `true` for compatibility. Passing `false` throws
`ArgumentError`; choose the child name on the left-hand side of `~` instead.

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

When sampling from `demo2(missing, 0.4)`, the latent variable `x` is prefixed with `a`, the
left-hand side of the tilde:

```jldoctest submodel-to_submodel
julia> model = demo2(missing, 0.4);

julia> haskey(rand(model), @varname(a.x))
true
```

The variable `a` receives the return value of `demo1` and can be used in subsequent lines,
as in the definition of `y` above.

We can verify that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-to_submodel
julia> accs = setacc!!(VarInfo(), RawValueAccumulator(false));

julia> _, accs = init!!(model, accs, InitFromPrior(), UnlinkAll());

julia> x = get_raw_values(accs)[@varname(a.x)];

julia> getlogjoint(accs) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
true
```

An explicit prefix composes with the tilde address: `a ~ to_submodel(prefix(child,
@varname(b)))` records a child variable `x` as `a.b.x`.
"""
to_submodel(m::Model) = Submodel(m)
function to_submodel(m::Model, auto_prefix::Bool)
    auto_prefix || throw(
        ArgumentError(
            "Unprefixed submodels are unsupported; use `name ~ to_submodel(model)` and update child variable addresses to `name.x`.",
        ),
    )
    return to_submodel(m)
end

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

# The tilde supplies the child address; parent contexts wrap the resulting namespace.
function _evaluate!!(
    submodel::Submodel{M},
    vi::AbstractVarInfo,
    parent_context::AbstractContext,
    left_vn::VarName,
) where {M<:Model}
    submodel_context_prefixed = PrefixContext(left_vn, submodel.model.context)

    # The parent controls evaluation; the child retains its enclosing contexts.
    submodel_context = setleafcontext(
        submodel_context_prefixed, leafcontext(parent_context)
    )

    eval_context = setleafcontext(parent_context, submodel_context)

    model = contextualize(submodel.model, eval_context)

    # Evaluate the wrapped model. These two lines are a verbatim copy of the body of
    # `_evaluate!!(model::Model, ::AbstractVarInfo)` (in `model.jl`), and the duplication is
    # deliberate: DO NOT replace them with `return _evaluate!!(model, vi)`. Each level of
    # submodel nesting grows the contextualised `Model`'s context type, and routing the
    # recursion through the shared `_evaluate!!(::Model, ...)` method trips Julia's recursion
    # limiter, which widens the `Model` argument to abstract and collapses the return type to
    # `Any`. Calling `model.f` directly avoids that. See
    # https://github.com/TuringLang/DynamicPPL.jl/pull/1427 and
    # https://github.com/TuringLang/Turing.jl/issues/2844 for the full explanation.
    args, kwargs = make_evaluate_args_and_kwargs(model, vi)
    return model.f(args...; kwargs...)
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
