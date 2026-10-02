# ------------
# Model values
# ------------

struct Condition end
struct ArgumentCondition end
struct Fix end

struct ModelValue{R<:Union{Condition,ArgumentCondition,Fix},T}
    value::T
    function ModelValue{R}(value::T) where {R<:Union{Condition,ArgumentCondition,Fix},T}
        (R === Condition || R === ArgumentCondition || R === Fix) ||
            throw(ArgumentError("A model value must have one concrete role"))
        return new{R,T}(value)
    end
end

struct NoModelBinding end

# Fixed bindings preserve the observation layer they shadow.
struct ModelBindingLayers{O<:VarNamedTuple,F<:VarNamedTuple,V<:VarNamedTuple,A<:Tuple}
    observations::O
    fixed::F
    values::V
    # Whole fixed owners survive expansion into partial storage.
    owners::A
end

# Child bindings selected from the parent's submodel namespace use the child's storage shape.
struct LocalModelValues{V<:Union{VarNamedTuple,ModelBindingLayers},A<:Tuple}
    values::V
    owners::A
end

# Default arguments need no enclosing namespace storage during evaluation.
struct UnprefixedArgumentValues{V<:VarNamedTuple}
    values::V
end

# Partial bindings retain the container from which their fields or tuple elements came.
struct ModelValueTree{T,V<:Union{VarNamedTuple,Tuple}}
    template::T
    values::V
    function ModelValueTree(template::T, values::V) where {T,V<:Union{VarNamedTuple,Tuple}}
        if values isa Tuple
            template isa Tuple && length(template) == length(values) ||
                throw(ArgumentError("Tuple bindings must preserve their template's length"))
        else
            all(name -> hasproperty(template, name), keys(values.data)) ||
                throw(ArgumentError("Field bindings must belong to their template"))
        end
        return new{T,V}(template, values)
    end
end

# ----------------
# Model definition
# ----------------

struct PrefixTemplate{V<:VarName,T,I}
    prefix::V
    template::T
    inner::I
end
_apply_prefix_template(::Nothing, template) = template
function _apply_prefix_template(prefix::VarName, template)
    return SkipTemplate{optic_skip_length(AbstractPPL.getoptic(prefix)) + 1}(template)
end
function _apply_prefix_template(prefix::PrefixTemplate, template)
    return VarNamedTuples.nested_template(
        AbstractPPL.getoptic(prefix.prefix),
        prefix.template,
        _apply_prefix_template(prefix.inner, template),
    )
end
_compose_prefix_templates(::Nothing, inner) = inner
function _compose_prefix_templates(prefix::VarName, inner)
    return PrefixTemplate(prefix, NoTemplate(), inner)
end
function _compose_prefix_templates(prefix::PrefixTemplate, inner)
    return PrefixTemplate(
        prefix.prefix, prefix.template, _compose_prefix_templates(prefix.inner, inner)
    )
end

is_splat_symbol(s::Symbol) = startswith(string(s), "#splat#")
function unsplat_symbol(s::Symbol)
    return is_splat_symbol(s) ? Symbol(chopprefix(string(s), "#splat#")) : s
end

# The existing argument metadata slot also carries macro-known LHS addresses.
struct ModelBindingMetadata{Arguments,LHS,Submodels,Types} end
_args_on_lhs(::ModelBindingMetadata{A}) where {A} = A
_args_on_lhs(names::Union{Tuple,Vector{Symbol}}) = Tuple(names)
_lhs_names(::ModelBindingMetadata{A,L}) where {A,L} = L
_lhs_names(::Tuple) = nothing
_may_have_submodels(::ModelBindingMetadata{A,L,S}) where {A,L,S} = !isempty(S)
_submodel_lhs_names(::ModelBindingMetadata{A,L,S}) where {A,L,S} = S
_may_have_submodels(::Tuple) = true

_declared_argument_type(::Tuple, name) = Any
function _declared_argument_type(::ModelBindingMetadata{A,L,S,T}, name) where {A,L,S,T}
    return fieldtype(T, name)
end

function _reconstruct_model end

"""
    Model{Threaded}(f, args::NamedTuple, defaults::NamedTuple, context=DefaultContext(); args_on_lhs=())

Store a model function, arguments, and context. Prefer [`@model`](@ref) for construction.
The names of arguments with LHS variables are stored as immutable type metadata.
Set `args_on_lhs` to the tuple of argument names that occur on the left-hand side of
`~`, for example `Model{false}(f, (; y=1.0), (;); args_on_lhs=(:y,))`.
These arguments record argument-supplied observations, just as with `@model`, and can
be bound with [`condition`](@ref) or [`fix`](@ref). Use [`decondition`](@ref) to remove
argument-supplied observations. Without `args_on_lhs`, direct construction records
no argument-supplied observations and its arguments cannot be bound.
Whole `missing` or `nothing` arguments throw `ArgumentError` when a tilde reads their
argument-supplied observations, even if the body replaces them. Use [`decondition`](@ref)
to make their LHS variables latent.
Partial bindings into whole `missing`/`nothing` arguments, even after deconditioning, throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.
At a submodel tilde, an argument LHS variable receives the submodel return value. The
argument supplies only its value before the tilde runs; its argument-supplied observation
is ignored at that tilde, so it needs no deconditioning. See [Binding rules](@ref).

Handwritten evaluators are responsible for the argument preparation and binding-aware tilde
protocol generated by `@model`. In particular, prepare argument LHS variables with
`prepare_model_argument` before the body, resolve their roles with `_get_argument_role`,
and validate observed or fixed values with `_check_tilde_value`. Calling `tilde_observe!!`
directly does not perform binding lookup or the whole-placeholder check. The constructor
records observations; it does not wrap a handwritten evaluator to enforce this protocol.
These helpers are internal; prefer reusing an evaluator generated by `@model`.

!!! note "Why keep `Model.f`?"
    Local models can capture variables from the enclosing function:

    ```julia
    function run_analysis(offset)
        @model function demo()
            x ~ Normal(offset, 1)
            return x
        end
        return demo()
    end
    ```

    `run_analysis(1)` and `run_analysis(2)` have the same evaluator type `F` but capture different
    offsets. The field `f` holds these values; the type alone cannot.

    Defining `(model::Model)(...)` or extending a global `get_evaluator` inside `run_analysis`
    requires a global method definition, which Julia rejects there. Using `eval` installs
    the method, but world age prevents the running `run_analysis` from calling it directly.
    We'd need a bridge such as `invokelatest` and separate storage for captures. Keeping `f`
    avoids both.
"""
struct Model{
    F,
    argnames,
    defaultnames,
    Targs,
    Tdefaults,
    C<:AbstractContext,
    Values<:Union{
        VarNamedTuple,ModelBindingLayers,LocalModelValues,UnprefixedArgumentValues
    },
    Threaded,
    ArgsOnLHS,
} <: AbstractProbabilisticProgram
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}
    context::C
    values::Values
    function Model{Threaded}(
        f::F,
        args::NamedTuple{A,Ta},
        defaults::NamedTuple{D,Td},
        context::C,
        values::V;
        args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol},ModelBindingMetadata}=(),
    ) where {F,A,Ta,D,Td,C,V,Threaded}
        mapreduce(
            pair -> pair.second isa ModelValue, &, _model_values(values); init=true
        ) || throw(ArgumentError("Model values must carry a condition or fix role"))
        argument_names = _args_on_lhs(args_on_lhs)
        metadata = args_on_lhs isa ModelBindingMetadata ? args_on_lhs : argument_names
        for name in argument_names
            name in (_argument_names(args)..., _argument_names(defaults)...) || throw(
                ArgumentError(
                    "`$name` in `args_on_lhs` is not an argument or default name"
                ),
            )
        end
        return new{F,A,D,Ta,Td,C,V,Threaded,metadata}(f, args, defaults, context, values)
    end
    # Internal reconstruction reuses already-validated bindings.
    function DynamicPPL._reconstruct_model(
        model::Model{F,A,D,Ta,Td}, context::C, values::V, ::Val{Threaded}
    ) where {F,A,D,Ta,Td,C,V,Threaded}
        return new{F,A,D,Ta,Td,C,V,Threaded,_binding_metadata(model)}(
            model.f, model.args, model.defaults, context, values
        )
    end
end

function _binding_metadata(
    ::Model{F,A,D,Ta,Td,C,V,Threaded,ArgsOnLHS}
) where {F,A,D,Ta,Td,C,V,Threaded,ArgsOnLHS}
    return ArgsOnLHS
end

_args_on_lhs(model::Model) = _args_on_lhs(_binding_metadata(model))

Base.@constprop :aggressive function Model{Threaded}(
    f,
    args::NamedTuple,
    defaults::NamedTuple,
    context::AbstractContext=DefaultContext();
    args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol},ModelBindingMetadata}=(),
) where {Threaded}
    values = _argument_defaults(merge(args, defaults), Val(_args_on_lhs(args_on_lhs)))
    return Model{Threaded}(f, args, defaults, context, values; args_on_lhs)
end

"""
    Model{Threaded}(f, args::NamedTuple; kwargs...)

Create a model with evaluation function `f` and arguments `args`.

Arguments are ordinary inputs; no argument-supplied observations or metadata about
argument LHS variables are recorded, and these arguments cannot be bound.
Use [`@model`](@ref) or the constructor accepting `defaults` and `args_on_lhs` to
construct a model with argument-supplied observations.

Keyword arguments `kwargs` are stored in the model's `defaults` field.
"""
function Model{Threaded}(f, args::NamedTuple; kwargs...) where {Threaded}
    return Model{Threaded}(f, args, NamedTuple(kwargs))
end

"""
    requires_threadsafe(model::Model)

Return whether `model` has been marked as needing threadsafe evaluation (using
`setthreadsafe`).
"""
requires_threadsafe(::Model{F,A,D,Ta,Td,C,V,Threaded}) where {F,A,D,Ta,Td,C,V,Threaded} =
    Threaded
function _reconstruct_model(model::Model; context=model.context, values=model.values)
    return _reconstruct_model(model, context, values, Val(requires_threadsafe(model)))
end

"""
    contextualize(model::Model, context::AbstractContext)

Return a model with its context replaced by `context`.
"""
function contextualize(model::Model, context::AbstractContext)
    if model.values isa UnprefixedArgumentValues && (
        last(extract_prefixes(context)) != _model_prefix(model) ||
        _prefix_template(context) !== _model_prefix_template(model)
    )
        model = _materialize_argument_values(model)
    end
    return _reconstruct_model(model; context)
end
"""Return a model with its leaf context replaced by `context`."""
function setleafcontext(model::Model, context::AbstractContext)
    return contextualize(model, setleafcontext(model.context, context))
end
_model_prefix(model::Model) = last(extract_prefixes(model.context))
_prefix_template(::AbstractContext) = nothing
_prefix_template(context::AbstractParentContext) = _prefix_template(childcontext(context))
_prefix_template(context::PrefixContext) = context.template
_model_prefix_template(model::Model) = _prefix_template(model.context)

"""
    setthreadsafe(model::Model, threadsafe::Bool)

Returns a new `Model` with its threadsafe flag set to `threadsafe`.

Threadsafe evaluation ensures correctness when executing model statements that mutate the
internal `VarInfo` object in parallel. For example, this is needed if tilde-statements are
nested inside `Threads.@threads` or similar constructs.

It is not needed for generic multithreaded operations that don't involve VarInfo. For
example, calculating a log-likelihood term in parallel and then calling `@addlogprob!`
outside of the parallel region is safe without needing to set `threadsafe=true`.

It is also not needed for multithreaded sampling with AbstractMCMC's `MCMCThreads()`.

Setting `threadsafe` to `true` increases the overhead in evaluating the model. Please see
[the Turing.jl docs](https://turinglang.org/docs/usage/threadsafe-evaluation/) for more
details.
"""
function setthreadsafe(model::Model, threadsafe::Bool)
    return if requires_threadsafe(model) == threadsafe
        model
    else
        _reconstruct_model(model, model.context, model.values, Val(threadsafe))
    end
end

function _prefix_values(values::ModelBindingLayers, vn::VarName, template)
    return ModelBindingLayers(
        _prefix_values(values.observations, vn, template),
        _prefix_values(values.fixed, vn, template),
        map(owner -> maybe_prefix(owner, vn), values.owners),
    )
end
function _prefix_values(values::VarNamedTuple, vn::VarName, template)
    isempty(values) && return values
    return templated_setindex!!(VarNamedTuple(), values, vn, template)
end

# Prefix templates can cross submodel boundaries without reading parent return values.
function _concretize_prefix(vn::VarName{S}, template) where {S}
    return VarName{S}(_concretize_prefix(AbstractPPL.getoptic(vn), template))
end
_concretize_prefix(optic::AbstractPPL.Iden, template) = optic
function _concretize_prefix(optic::AbstractPPL.Property{S}, template) where {S}
    AbstractPPL.is_dynamic(optic) || return optic
    child = _concretize_prefix(optic.child, VarNamedTuples.SharedGetProperty{S}()(template))
    return AbstractPPL.Property{S}(child)
end
function _concretize_prefix(optic::AbstractPPL.Index, template)
    AbstractPPL.is_dynamic(optic) || return optic
    optic = AbstractPPL.concretize_top_level(optic, VarNamedTuples.template_array(template))
    AbstractPPL.is_dynamic(optic.child) || return optic
    child = _concretize_prefix(optic.child, VarNamedTuples.index_template(template, optic))
    return AbstractPPL.Index(optic.ix, optic.kw, child)
end

maybe_prefix(vn::VarName, ::Nothing) = vn
maybe_prefix(::Nothing, ::Nothing) = nothing
maybe_prefix(::Nothing, prefix::VarName) = prefix
maybe_prefix(vn::VarName, prefix::VarName) = AbstractPPL.prefix(vn, prefix)

"""
    prefix(model::Model, x::VarName; template=NoTemplate())
    prefix(model::Model, x::Val{sym})
    prefix(model::Model, x::Any)

Return `model` but with all random variables prefixed by `x`, where `x` is either:
- a `VarName` (e.g. `@varname(a)`),
- a `Val{sym}` (e.g. `Val(:a)`), or
- for any other type, `x` is converted to a Symbol and then to a `VarName`. Note that
  this will introduce runtime overheads so is not recommended unless absolutely
  necessary.

For an indexed prefix, `template` supplies the enclosing container's shape and resolves
`begin` and `end` indices.

# Examples

```jldoctest
julia> using DynamicPPL: prefix

julia> @model demo() = x ~ Dirac(1)
demo (generic function with 2 methods)

julia> rand(prefix(demo(), @varname(my_prefix)))
VarNamedTuple
└─ my_prefix => VarNamedTuple
                └─ x => 1

julia> rand(prefix(demo(), Val(:my_prefix)))
VarNamedTuple
└─ my_prefix => VarNamedTuple
                └─ x => 1
```
"""
function prefix(model::Model, x::VarName; template=NoTemplate())
    template = VarNamedTuples.materialize_template(template)
    x = _concretize_prefix(x, template)
    model = _materialize_argument_values(model)
    values =
        if model.values isa VarNamedTuple &&
            _model_prefix(model) === nothing &&
            !isempty(model.values) &&
            mapreduce(
                pair -> pair.second isa ModelValue{ArgumentCondition},
                &,
                model.values;
                init=true,
            )
            UnprefixedArgumentValues(model.values)
        else
            _prefix_values(model.values, x, template)
        end
    return _prefix_model(model, x, template, values)
end
function _prefix_model(model::Model, x::VarName, template, values)
    model_prefix = maybe_prefix(_model_prefix(model), x)
    prefix_template =
        if template isa NoTemplate && _model_prefix_template(model) === nothing
            nothing
        else
            inner = if _model_prefix_template(model) === nothing
                _model_prefix(model)
            else
                _model_prefix_template(model)
            end
            PrefixTemplate(x, template, inner)
        end
    context = PrefixContext(
        model_prefix, first(extract_prefixes(model.context)), prefix_template
    )
    return _reconstruct_model(model; context, values)
end
function prefix(model::Model, ::Val{sym}) where {sym}
    return prefix(model, VarName{sym}())
end
function prefix(model::Model, x)
    return prefix(model, VarName{Symbol(x)}())
end

function _prefix_varname_and_template(vn::VarName, template::Any, model::Model)
    return _prefix_varname_and_template(
        vn, template, _model_prefix(model), _model_prefix_template(model)
    )
end
function _prefix_varname_and_template(vn::VarName, template, prefix, prefix_template)
    prefix === nothing && return vn, template
    pt = prefix_template === nothing ? prefix : prefix_template
    return AbstractPPL.prefix(vn, prefix), _apply_prefix_template(pt, template)
end

function tilde_assume!!(
    model::Model,
    context::AbstractContext,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return tilde_assume!!(context, right, vn, template, vi)
end

function _check_tilde_value(value, vn, role::Union{Condition,Fix})
    remove = role isa Fix ? "unfix" : "decondition"
    absent = if _contains_missing(value)
        "missing"
    elseif _contains_nothing(value)
        "nothing"
    else
        nothing
    end
    absent === nothing || throw(
        ArgumentError(
            "LHS variable `$vn` contains `$absent`; make it latent with `$remove`."
        ),
    )
    return value
end

"""
    tilde_observe!!(prefix, prefix_template, right::Distribution, left, vn, template, vi)

Accumulate an observation and return `(left, vi)` with the updated varinfo.

`left` is supplied by the model's conditioned values or by a literal expression. `vn` is
the variable name before prefixing, or `nothing` for a literal. `template` describes the
top-level variable's storage; literals use `NoTemplate()`.

Apply `prefix` (a `VarName` or `nothing`) and its storage template `prefix_template`,
then delegate to [`accumulate_observe!!`](@ref). The compiler passes this metadata directly
so observations do not box the model. Every observation calls this function, independently
of the evaluation context. Fixed LHS variables bypass it and do not contribute to the log probability.
"""
function tilde_observe!!(
    prefix, prefix_template, right::Distribution, left, vn, template, vi
)
    vn, template = if vn === nothing
        vn, NoTemplate()
    else
        _prefix_varname_and_template(vn, template, prefix, prefix_template)
    end
    left = _check_tilde_value(left, vn, Condition())
    vi = accumulate_observe!!(vi, right, left, vn, template)
    return left, vi
end

"""
    store_coloneq_value!!(model::Model, vn::VarName, right, template, vi)

Store a tracked assignment's value in the raw-value accumulator and return the updated `vi`.

Apply the model's prefix to `vn` and its storage `template`. The evaluator calls this
function only when tracked-value extraction is enabled; no context hook is involved.
"""
function store_coloneq_value!!(
    model::Model, vn::VarName, right::Any, template::Any, vi::AbstractVarInfo
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return map_accumulator!!(
        acc -> store_colon_eq!!(acc, vn, right, template), vi, Val(RAW_VALUE_ACCNAME)
    )
end

"""
    (model::Model)([rng, varinfo])

Sample from the prior of the `model` with random number generator `rng`.

Returns the model's return value.

Note that calling this with an existing `varinfo` object will mutate it.
"""
(model::Model)() = model(Random.default_rng(), VarInfo())
function (model::Model)(varinfo::AbstractVarInfo)
    return model(Random.default_rng(), varinfo)
end
# ^ Weird Documenter.jl bug means that we have to write the two above separately
# as it can only detect the `function`-less syntax.
function (model::Model)(rng::Random.AbstractRNG, varinfo::AbstractVarInfo=VarInfo(()))
    return first(init!!(rng, model, varinfo, InitFromPrior(), UnlinkAll()))
end

"""
    init!!(
        [rng::Random.AbstractRNG,]
        model::Model,
        varinfo::AbstractVarInfo,
        init_strategy::AbstractInitStrategy,
        [transform_strategy::AbstractTransformStrategy=UnlinkAll(),]
    )

Evaluate `model` with the given initialisation and transform strategies, resetting and
filling the accumulators in `varinfo`. Parameter values are recorded only if a value
accumulator is present.

`transform_strategy` controls the output representation and defaults to `UnlinkAll()`.

Returns a tuple of the model's return value, plus the updated `varinfo` object.
"""
function init!!(
    rng::Random.AbstractRNG,
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    ctx = InitContext(rng, init_strategy, transform_strategy)
    model = DynamicPPL.setleafcontext(model, ctx)
    return DynamicPPL.evaluate_nowarn!!(model, vi)
end
function init!!(
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    return init!!(Random.default_rng(), model, vi, init_strategy, transform_strategy)
end

"""
    evaluate!!(model::Model, varinfo)

Evaluate the `model` with the given `varinfo`, wrapping it in a `ThreadSafeVarInfo` if the
model is marked as needing threadsafe evaluation.

!!! warning
    The semantics of this method are complicated. We **strongly** recommend that users do
    *not* use this method unless absolutely necessary. In the future this method will be
    deprecated and removed. As far as possible (and it should **always** be possible --
    please open an issue if you do not know how to adapt your code!) you should use the
    five-argument `init!!([rng,] model, ::VarInfo, init_strategy,
    transform_strategy)` method, which has more explicit semantics and allows you to have
    more control over each part of the evaluation process.

The exact semantics depend on the `model`'s context. Fundamentally, this method executes the
model evaluation function (i.e., the function used to define the model) using the given
`varinfo` as an argument. At each tilde-statement, `tilde_assume!!` or `tilde_observe!!` is
called, whose behaviour depends on the model's context.

Broadly speaking, if the leaf context is an `InitContext`, then this function:

- uses the initialisation strategy inside the `InitContext`;
- uses the transform strategy inside the `InitContext`;
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- overwrites the values in `varinfo` with the new values obtained from the initialisation strategy.

If the leaf context is a `DefaultContext`, then this function:

- uses the values inside the `varinfo` as the initialisation strategy;
- derives a transform strategy from the `varinfo`'s stored variables (if a linked variable is
  stored, then the transform strategy will treat that variable as linked; likewise for
  unlinked)
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- records the values of executed LHS variables in the reset value accumulator, omitting LHS variables
  that are no longer executed.

The long-term plan for this method is to:

- Replace `DefaultContext` with `InitContext` by splitting up the functionality of `DefaultContext`
  into its constituent components
- Remove the `VarInfo` argument, and instead use only an `AccumulatorTuple`
- Separate the initialisation and transform strategies into separate arguments, instead of storing
  them inside the model's context.
"""
function AbstractPPL.evaluate!!(model::Model, varinfo::AbstractVarInfo)
    @warn (
        "Calling `evaluate!!(model, varinfo)` directly is not recommended and will be" *
        " deprecated in the future. Please switch to using `init!!([rng,] model," *
        " ::VarInfo, init_strategy, transform_strategy)` instead, which" *
        " has more explicit semantics and allows you to have more control over each" *
        " part of the evaluation process. Please see the DynamicPPL documentation" *
        " for more details: https://turinglang.org/DynamicPPL.jl/stable/evaluation"
    ) maxlog = 5
    return DynamicPPL.evaluate_nowarn!!(model, varinfo)
end

"""
    evaluate_nowarn!!(model::Model, varinfo)

This is the same as `evaluate!!(model, varinfo)` but without the deprecation warning.

!!! warning
    This is meant for internal use in DynamicPPL.jl only! If you rely on this method in your
    code, please note that it may break at any time.
"""
function evaluate_nowarn!!(model::Model, varinfo::AbstractVarInfo)
    if leafcontext(model.context) isa DefaultContext
        values, strategy = if hasacc(varinfo, Val(VECTORVAL_ACCNAME))
            copy(get_vector_values(varinfo)),
            infer_transform_strategy_from_values(get_vector_values(varinfo))
        else
            VarNamedTuple(), UnlinkAll()
        end
        ctx = InitContext(InitFromParams(values, nothing), strategy)
        model = setleafcontext(model, ctx)
    end
    return if requires_threadsafe(model)
        # Use of float_type_with_fallback(eltype(x)) is necessary to deal with cases where x is
        # a gradient type of some AD backend.
        param_eltype = DynamicPPL.get_param_eltype(varinfo, model.context)
        wrapper = ThreadSafeVarInfo(varinfo, param_eltype)
        result, wrapper_new = _evaluate!!(model, wrapper)
        # TODO(penelopeysm): If seems that if you pass a TSVI to this method, it
        # will return the underlying VI, which is a bit counterintuitive (because
        # calling TSVI(::TSVI) returns the original TSVI, instead of wrapping it
        # again).
        accs = map(getaccs(wrapper_new)) do acc
            if acc isa TSVNTAccumulator
                VNTAccumulator{accumulator_name(acc)}(acc.f, acc.values)
            else
                acc
            end
        end
        return result, setaccs!!(wrapper_new.varinfo, accs)
    else
        _evaluate!!(model, resetaccs!!(varinfo))
    end
end

"""
    _evaluate!!(model::Model, varinfo)

Evaluate the `model` with the given `varinfo`.

This function does not wrap the varinfo in a `ThreadSafeVarInfo`. It also does not
reset the log probability of the `varinfo` before running.
"""
function _evaluate!!(model::Model, varinfo::AbstractVarInfo)
    args, kwargs = make_evaluate_args_and_kwargs(model, varinfo)
    return model.f(args...; kwargs...)
end

"""
    make_evaluate_args_and_kwargs(model, varinfo)

Return the arguments and keyword arguments to be passed to the evaluator of the model, i.e. `model.f`e.
"""
@generated function make_evaluate_args_and_kwargs(
    model::Model{_F,argnames,defaultnames}, varinfo::AbstractVarInfo
) where {_F,argnames,defaultnames}
    unwrap_args = [
        if is_splat_symbol(var)
            :(
                $convert_model_argument(
                    $get_param_eltype(varinfo, model.context), model.args.$var
                )...
            )
        else
            :($convert_model_argument(
                $get_param_eltype(varinfo, model.context), model.args.$var
            ))
        end for var in argnames
    ]
    unwrap_kwargs = [
        is_splat_symbol(var) ? :(model.defaults.$var...) : :($var = model.defaults.$var) for
        var in defaultnames
    ]
    return quote
        args = (model, varinfo, $(unwrap_args...))
        kwargs = (; $(unwrap_kwargs...))
        return args, kwargs
    end
end

"""
    get_param_eltype(varinfo::AbstractVarInfo, context::AbstractContext)

Get the element type of the parameters being used to evaluate a model, using a `varinfo`
under the given `context`. For example, when evaluating a model with ForwardDiff AD, this
should return `ForwardDiff.Dual`.

For `InitContext`, query its initialisation strategy. For other leaf contexts, infer
this type from recorded vectorised values, or return `Union{}` if no value accumulator
is present. Parent contexts delegate to their child context.

See the docstring of `get_param_eltype(strategy::AbstractInitStrategy)` for the
strategy interface.
"""
function get_param_eltype(vi::AbstractVarInfo, ctx::AbstractParentContext)
    return get_param_eltype(vi, DynamicPPL.childcontext(ctx))
end
function get_param_eltype(vi::AbstractVarInfo, ::AbstractContext)
    hasacc(vi, Val(VECTORVAL_ACCNAME)) || return Union{}
    return get_param_eltype(InitFromParams(get_vector_values(vi), nothing))
end
function get_param_eltype(::AbstractVarInfo, ctx::InitContext)
    return get_param_eltype(ctx.strategy)
end

@generated function _argument_names(::NamedTuple{names}) where {names}
    return QuoteNode(map(unsplat_symbol, names))
end

"""
    getargnames(model::Model)

Get a tuple of the argument names of the `model`.
"""
getargnames(model::Model{_F,argnames}) where {argnames,_F} = argnames

"""
    nameof(model::Model)

Get the name of the `model` as `Symbol`.
"""
Base.nameof(model::Model) = Symbol(model.f)
Base.nameof(model::Model{<:Function}) = nameof(model.f)

"""
    rand([rng=Random.default_rng()], model::Model)

Sample a `VarNamedTuple` of raw values from the prior of `model`.
"""
function Base.rand(rng::Random.AbstractRNG, model::Model)
    vi = VarInfo((RawValueAccumulator(false),))
    vi = last(init!!(rng, model, vi, InitFromPrior(), UnlinkAll()))
    return get_raw_values(vi)
end
Base.rand(model::Model) = rand(Random.default_rng(), model)

"""
    logjoint(model::Model, params)
    logjoint(model::Model, varinfo::AbstractVarInfo)

Return the log joint probability of variables `params` for the probabilistic `model`, or the
log joint of the data in `varinfo` if provided.

Note that this probability always refers to the parameters in unlinked space, i.e., the
return value of `logjoint` does not depend on whether `VarInfo` has been linked or not.

See also [`logprior`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 3 methods)

julia> # Using a `NamedTuple`.
       logjoint(demo([1.0]), (m = 100.0, ))
-9902.33787706641

julia> # Using a `OrderedDict`.
       logjoint(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-9902.33787706641

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0) + logpdf(Normal(), 100.0)
-9902.33787706641
```
"""
function logjoint(model::Model, params)
    vi = VarInfo(AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getlogjoint(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function logjoint(model::Model, varinfo::AbstractVarInfo)
    return logjoint(model, get_values(varinfo))
end

"""
    logprior(model::Model, params)
    logprior(model::Model, varinfo::AbstractVarInfo)

Return the log prior probability of variables `params` for the probabilistic `model`, or the
log prior of the data in `varinfo` if provided.

Note that this probability always refers to the parameters in unlinked space, i.e., the
return value of `logprior` does not depend on whether `VarInfo` has been linked or not.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 3 methods)

julia> # Using a `NamedTuple`.
       logprior(demo([1.0]), (m = 100.0, ))
-5000.918938533205

julia> # Using a `OrderedDict`.
       logprior(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-5000.918938533205

julia> # Truth.
       logpdf(Normal(), 100.0)
-5000.918938533205
```
"""
function logprior(model::Model, params)
    vi = VarInfo(AccumulatorTuple(LogPriorAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getlogprior(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function logprior(model::Model, varinfo::AbstractVarInfo)
    return logprior(model, get_values(varinfo))
end

"""
    loglikelihood(model::Model, params)
    loglikelihood(model::Model, varinfo::AbstractVarInfo)

Return the log likelihood of variables `params` for the probabilistic `model`, or the log
likelihood of the data in `varinfo` if provided.

See also [`logjoint`](@ref) and [`logprior`](@ref).

# Examples
```jldoctest; setup=:(using Distributions)
julia> @model function demo(x)
           m ~ Normal()
           for i in eachindex(x)
               x[i] ~ Normal(m, 1.0)
           end
       end
demo (generic function with 3 methods)

julia> # Using a `NamedTuple`.
       loglikelihood(demo([1.0]), (m = 100.0, ))
-4901.418938533205

julia> # Using a `OrderedDict`.
       loglikelihood(demo([1.0]), OrderedDict(@varname(m) => 100.0))
-4901.418938533205

julia> # Truth.
       logpdf(Normal(100.0, 1.0), 1.0)
-4901.418938533205
"""
function Distributions.loglikelihood(model::Model, params)
    vi = VarInfo(AccumulatorTuple(LogLikelihoodAccumulator()))
    init_strategy = InitFromParams(params, nothing)
    return getloglikelihood(last(init!!(model, vi, init_strategy, UnlinkAll())))
end
function Distributions.loglikelihood(model::Model, varinfo::AbstractVarInfo)
    return loglikelihood(model, get_values(varinfo))
end

# Implemented & documented in DynamicPPLMCMCChainsExt
function predict end

"""
    returned(model::Model, parameters...)

Initialise a `model` using the given `parameters` and return the model's return value. The
parameters must be provided in a format that can be wrapped in an `InitFromParams`, i.e.,
`InitFromParams(parameters..., nothing)` must be a valid `AbstractInitStrategy` (where
`nothing` is the fallback strategy to use if parameters are not provided).

As far as DynamicPPL is concerned, `parameters` can be either a singular `NamedTuple` or an
`AbstractDict{<:VarName}`; however this method is left flexible to allow for other packages
that wish to extend `InitFromParams`.

# Example
```jldoctest
julia> using DynamicPPL, Distributions

julia> @model function demo()
           m ~ Normal()
           return (mp1 = m + 1,)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> returned(model, (; m = 1.0))
(mp1 = 2.0,)

julia> returned(model, Dict{VarName,Float64}(@varname(m) => 2.0))
(mp1 = 3.0,)
```
"""
function returned(model::Model, parameters...)
    # Note: we can't use `fix(model, parameters)` because
    # https://github.com/TuringLang/DynamicPPL.jl/issues/1097
    return first(
        init!!(
            model,
            DynamicPPL.VarInfo(DynamicPPL.AccumulatorTuple()),
            # Use `nothing` as the fallback to ensure that any missing parameters cause an
            # error
            InitFromParams(parameters..., nothing),
            UnlinkAll(),
        ),
    )
end

function AbstractPPL.evaluate!!(model::Model, context::AbstractContext, vi::AbstractVarInfo)
    return evaluate_nowarn!!(setleafcontext(model, context), vi)
end
