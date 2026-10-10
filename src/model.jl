# ------------
# Model values
# ------------

struct Condition end
struct ArgumentCondition end
struct Fix end

"""
    Recursive()

Remove bindings throughout a model and its submodels. Pass this marker
immediately after the model to `decondition` or `unfix`.
"""
struct Recursive end

struct ModelValue{R<:Union{Condition,ArgumentCondition,Fix},T}
    value::T
    function ModelValue{R}(value::T) where {R<:Union{Condition,ArgumentCondition,Fix},T}
        (R === Condition || R === ArgumentCondition || R === Fix) ||
            throw(ArgumentError("A model value must have one concrete role"))
        return new{R,T}(value)
    end
end

struct NoModelBinding end

# A removal has no value or storage. Exceptions are later bindings in this layer.
struct ModelRemoval{N,E<:Tuple}
    name::N
    exceptions::E
    matched::Bool # Used only by check_model to warn about unused removals.
    # Identity only: Ref{Nothing} carries no mutable evaluation data. Copies and
    # namespace selections deliberately share it; unlike gensym it is collectible.
    token::Base.RefValue{Nothing}
end
function ModelRemoval(name, exceptions, matched)
    return ModelRemoval(name, exceptions, matched, Ref(nothing))
end

struct ModelBindingTemplate{R,N,T}
    name::N
    storage::T
end
function ModelBindingTemplate{R}(name::N, storage::T) where {R,N,T}
    return ModelBindingTemplate{R,N,T}(name, storage)
end

# Fixed bindings preserve the observation layer they shadow.
struct ModelBindingLayers{
    O<:VarNamedTuple,
    F<:VarNamedTuple,
    V<:VarNamedTuple,
    A<:Tuple,
    OR<:Tuple,
    FR<:Tuple,
    BT<:Tuple,
}
    observations::O
    fixed::F
    values::V
    # Whole fixed owners survive expansion into partial storage.
    owners::A
    observation_removals::OR
    fixed_removals::FR
    templates::BT
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

# Partial bindings retain the NamedTuple from which their fields came.
# S records whether that whole owner reaches child namespaces.
struct ModelValueTree{T,V<:VarNamedTuple,S}
    template::T
    values::V
    function ModelValueTree(
        template::T, values::V, ::Val{S}=Val(false)
    ) where {T,V<:VarNamedTuple,S}
        all(name -> hasproperty(template, name), keys(values.data)) ||
            throw(ArgumentError("Field bindings must belong to their template"))
        return new{T,V,S}(template, values)
    end
end

# Copies and partial edits retain the whole owner's recursive scope.
function ModelValueTree(tree::ModelValueTree{T,V,S}, values::VarNamedTuple) where {T,V,S}
    return ModelValueTree(tree.template, values, Val(S))
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
function _compose_prefix_templates(prefix::VarName, inner::Union{Nothing,VarName})
    return maybe_prefix(inner, prefix)
end
function _compose_prefix_templates(prefix::VarName, inner)
    return PrefixTemplate(prefix, NoTemplate(), inner)
end
function _compose_prefix_templates(prefix::PrefixTemplate, inner)
    return PrefixTemplate(
        prefix.prefix, prefix.template, _compose_prefix_templates(prefix.inner, inner)
    )
end

_getprefix(prefix::Union{Nothing,VarName}) = prefix
_getprefix(prefix::PrefixTemplate) = maybe_prefix(_getprefix(prefix.inner), prefix.prefix)

is_splat_symbol(s::Symbol) = startswith(string(s), "#splat#")
function unsplat_symbol(s::Symbol)
    return is_splat_symbol(s) ? Symbol(chopprefix(string(s), "#splat#")) : s
end

# The existing argument metadata slot also carries macro-known LHS addresses.
struct ModelBindingMetadata{Arguments,LHS,Unprefixed,Types,Addresses,Templates} end
function ModelBindingMetadata{A,L,S,T,N}() where {A,L,S,T,N}
    return ModelBindingMetadata{A,L,S,T,N,()}()
end
_binding_template_names(::Tuple) = ()
_binding_template_names(::ModelBindingMetadata{A,L,S,T,N,B}) where {A,L,S,T,N,B} = B
function _with_binding_template_names(
    ::ModelBindingMetadata{A,L,S,T,N,B}, names
) where {A,L,S,T,N,B}
    return ModelBindingMetadata{A,L,S,T,N,_merge_binding_template_names(B, names)}()
end
_with_binding_template_names(metadata::Tuple, names) = metadata
_merge_binding_template_names(names::Tuple, ::Tuple{}) = names
function _merge_binding_template_names(names::Tuple, extra::Tuple)
    name = first(extra)
    merged = name in names ? names : (names..., name)
    return _merge_binding_template_names(merged, Base.tail(extra))
end
_args_on_lhs(::ModelBindingMetadata{A}) where {A} = A
_args_on_lhs(names::Union{Tuple,Vector{Symbol}}) = Tuple(names)
_lhs_names(::ModelBindingMetadata{A,L}) where {A,L} = L
_lhs_names(::Tuple) = nothing
_lhs_addresses(::ModelBindingMetadata{A,L,S,T,N}) where {A,L,S,T,N} = N
_lhs_addresses(::Tuple) = ()
_has_unprefixed_submodel(::ModelBindingMetadata{A,L,S}) where {A,L,S} = S

_declared_argument_type(::Tuple, name) = Any
function _declared_argument_type(::ModelBindingMetadata{A,L,S,T}, name) where {A,L,S,T}
    return fieldtype(T, name)
end

function _reconstruct_model end

"""
    Model{Threaded}(f, args::NamedTuple, defaults::NamedTuple; args_on_lhs=())

Store a model function, arguments, prefixes, and bindings. The evaluation context is passed
to [`evaluate!!`](@ref). Prefer [`@model`](@ref) for construction.
For direct construction, use [`condition`](@ref) or [`fix`](@ref) to supply bindings.
The names of arguments with LHS variables are stored as immutable type metadata.
Set `args_on_lhs` to the tuple of argument names that occur on the left-hand side of
`~`, for example `Model{false}(f, (; y=1.0), (;); args_on_lhs=(:y,))`.
These arguments record argument-supplied observations, just as with `@model`, and can
be bound with [`condition`](@ref) or [`fix`](@ref). Use [`decondition`](@ref) to remove
argument-supplied observations. Without `args_on_lhs`, direct construction records
no argument-supplied observations and its arguments cannot be bound.
A whole `missing` or `nothing` argument in `args_on_lhs`, or a `missing` element of such an
argument array, supplies no observation; see [Missing data](@ref).
Incomplete partial bindings into whole `missing`/`nothing` arguments throw
`ArgumentError` when bound; supply a concrete argument such as `f(zeros(n))` or a whole binding.
A submodel tilde cannot have an LHS rooted at a model argument; use a local LHS variable.
See [Binding rules](@ref).

Handwritten evaluators are responsible for the argument preparation and binding-aware tilde
protocol generated by `@model`. Their signature begins with `(model, context::Context,
varinfo::AbstractVarInfo)`, followed by the model arguments and keywords. In particular,
prepare argument LHS variables with
`prepare_model_argument` before the body, resolve their roles with `_get_argument_role`,
and validate observed or fixed values with `_check_tilde_value`. Calling `tilde_observe!!`
directly does not perform binding lookup. The constructor records observations; it does not
wrap a handwritten evaluator to enforce this protocol.
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
    Prefix<:Union{VarName,Nothing,PrefixTemplate},
    Values<:Union{
        VarNamedTuple,ModelBindingLayers,LocalModelValues,UnprefixedArgumentValues
    },
    Threaded,
    ArgsOnLHS,
} <: AbstractProbabilisticProgram
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}
    prefix::Prefix
    values::Values
    function Model{Threaded}(
        f::F,
        args::NamedTuple{A,Ta},
        defaults::NamedTuple{D,Td},
        prefix::P,
        values::V;
        args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol},ModelBindingMetadata}=(),
    ) where {F,A,Ta,D,Td,P,V,Threaded}
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
        return new{F,A,D,Ta,Td,P,V,Threaded,metadata}(f, args, defaults, prefix, values)
    end
    # Internal reconstruction reuses already-validated bindings.
    function DynamicPPL._reconstruct_model(
        model::Model{F,A,D,Ta,Td}, prefix::P, values::V, ::Val{Threaded}
    ) where {F,A,D,Ta,Td,P,V,Threaded}
        return new{F,A,D,Ta,Td,P,V,Threaded,_binding_metadata(model)}(
            model.f, model.args, model.defaults, prefix, values
        )
    end
end

"""
    getprefix(model::Model)

Return the combined prefix of the model's LHS variables, or `nothing` when absent.
Storage templates for nested submodel namespaces remain internal to the model.
"""
getprefix(model::Model) = _getprefix(model.prefix)

function _binding_metadata(
    ::Model{F,A,D,Ta,Td,P,V,Threaded,ArgsOnLHS}
) where {F,A,D,Ta,Td,P,V,Threaded,ArgsOnLHS}
    return ArgsOnLHS
end

_args_on_lhs(model::Model) = _args_on_lhs(_binding_metadata(model))

Base.@constprop :aggressive function Model{Threaded}(
    f,
    args::NamedTuple,
    defaults::NamedTuple;
    args_on_lhs::Union{Tuple{Vararg{Symbol}},Vector{Symbol},ModelBindingMetadata}=(),
) where {Threaded}
    values = _argument_defaults(merge(args, defaults), Val(_args_on_lhs(args_on_lhs)))
    model = Model{Threaded}(f, args, defaults, nothing, values; args_on_lhs)
    found = _placeholder_names(merge(args, defaults), Val(_args_on_lhs(args_on_lhs)))
    return all(isempty, found) ? model : _remove_placeholders(model, found)
end

# One removal per argument: whole placeholders as a tuple, which keeps the model type
# inferable, and array elements as a vector, since splatting it would compile per count.
_remove_placeholders(model, ::Tuple{}) = model
function _remove_placeholders(model, found::Tuple)
    names = first(found)
    model = isempty(names) ? model : _local_remove(Condition, model, names)
    return _remove_placeholders(model, Base.tail(found))
end

# Placeholders in arguments on the LHS supply no observation: a whole `missing` or `nothing`
# argument, and each `missing` element of a top-level argument array. The type gates the scan.
@generated function _placeholder_names(
    arguments::NamedTuple{names}, ::Val{args_on_lhs}
) where {names,args_on_lhs}
    found = map(names) do stored_name
        name = unsplat_symbol(stored_name)
        name in args_on_lhs || return :(())
        return :(_placeholder_names($(VarName{name}()), arguments.$stored_name))
    end
    return :(($(found...),))
end
_placeholder_names(vn::VarName, ::Union{Missing,Nothing}) = (vn,)
_placeholder_names(::VarName, value) = ()
function _placeholder_names(vn::VarName, value::AbstractArray)
    Missing <: eltype(value) || return ()
    names = [
        AbstractPPL.append_optic(vn, AbstractPPL.Index(Tuple(I), (;))) for
        I in CartesianIndices(value) if isassigned(value, I) && value[I] === missing
    ]
    isempty(names) ||
        _partial_binding_array(value) ||
        throw(
            ArgumentError(
                "Argument `$vn` has `missing` elements, which mark data unobserved only in " *
                "an `Array` or an `Array`-backed OffsetArray, ComponentArray or DimArray, " *
                "not in $(typeof(value)); pass `collect($vn)`.",
            ),
        )
    return names
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
requires_threadsafe(::Model{F,A,D,Ta,Td,P,V,Threaded}) where {F,A,D,Ta,Td,P,V,Threaded} =
    Threaded
function _reconstruct_model(model::Model; prefix=model.prefix, values=model.values)
    return _reconstruct_model(model, prefix, values, Val(requires_threadsafe(model)))
end

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
        _reconstruct_model(model, model.prefix, model.values, Val(threadsafe))
    end
end

function _prefix_values(values::ModelBindingLayers, vn::VarName, template)
    return ModelBindingLayers(
        _prefix_values(values.observations, vn, template),
        _prefix_values(values.fixed, vn, template),
        map(owner -> maybe_prefix(owner, vn), values.owners),
        map(r -> _prefix_removal(r, vn), values.observation_removals),
        map(r -> _prefix_removal(r, vn), values.fixed_removals),
        map(t -> _prefix_binding_template(t, vn), values.templates),
    )
end
function _prefix_values(values::VarNamedTuple, vn::VarName, template)
    isempty(values) && return values
    return templated_setindex!!(VarNamedTuple(), values, vn, template)
end

# Prefix templates can cross submodel boundaries without reading parent return values.
function _concretize_prefix(vn::VarName{S}, template; kwargs...) where {S}
    return VarName{S}(_concretize_prefix(AbstractPPL.getoptic(vn), template; kwargs...))
end
_concretize_prefix(optic::AbstractPPL.Iden, template; kwargs...) = optic
function _concretize_prefix(
    optic::AbstractPPL.Property{S}, template; depth=typemax(Int), prefix=Val(false)
) where {S}
    (prefix isa Val{false} && (depth == 0 || !AbstractPPL.is_dynamic(optic))) &&
        return optic
    child_template = if AbstractPPL.is_dynamic(optic.child)
        VarNamedTuples.SharedGetProperty{S}()(template)
    else
        NoTemplate()
    end
    child = _concretize_prefix(optic.child, child_template; depth=depth - 1, prefix)
    return AbstractPPL.Property{S}(child)
end
function _concretize_prefix(
    optic::AbstractPPL.Index, template; depth=typemax(Int), prefix=Val(false)
)
    (prefix isa Val{false} && (depth == 0 || !AbstractPPL.is_dynamic(optic))) &&
        return optic
    if any(i -> i isa AbstractPPL.DynamicIndex, optic.ix)
        template isa NoTemplate && throw(
            ArgumentError(
                "`begin` and `end` in a prefix need a template; pass `template=`."
            ),
        )
        optic = AbstractPPL.concretize_top_level(
            optic, VarNamedTuples.template_array(template)
        )
    end
    if prefix isa Val{true}
        indices = _expand_cartesian(optic.ix)
        all(i -> i isa Integer && !(i isa Bool), indices) && isempty(optic.kw) || throw(
            ArgumentError(
                "Prefix index [$(join(map(repr, optic.ix), ", "))] is not a scalar integer; prefixes require properties and integer indices. Use an integer-indexed prefix, or to_submodel(model, false) for a sliced return LHS.",
            ),
        )
        optic = AbstractPPL.Index(indices, optic.kw, optic.child)
    end
    prefix isa Val{false} && depth == 1 && return optic
    child_template = if AbstractPPL.is_dynamic(optic.child)
        VarNamedTuples.index_template(template, optic)
    else
        NoTemplate()
    end
    child = _concretize_prefix(optic.child, child_template; depth=depth - 1, prefix)
    return AbstractPPL.Index(optic.ix, optic.kw, child)
end

# `CartesianIndex` is one spelling of integer coordinates; addresses store the coordinates.
function _expand_cartesian(ix::Tuple)
    return mapreduce(
        i -> i isa CartesianIndex ? Tuple(i) : (i,), (a, b) -> (a..., b...), ix; init=()
    )
end
function _expand_cartesian(vn::VarName{S}) where {S}
    return VarName{S}(_expand_cartesian(AbstractPPL.getoptic(vn)))
end
_expand_cartesian(optic::AbstractPPL.Iden) = optic
function _expand_cartesian(optic::AbstractPPL.Property{S}) where {S}
    return AbstractPPL.Property{S}(_expand_cartesian(optic.child))
end
function _expand_cartesian(optic::AbstractPPL.Index)
    return AbstractPPL.Index(
        _expand_cartesian(optic.ix), optic.kw, _expand_cartesian(optic.child)
    )
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
`begin` and `end` indices. Prefixes accept properties and scalar integer indices
(excluding `Bool`); `CartesianIndex` is expanded into integer coordinates. Ranges,
colons and masks are rejected.

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
    AbstractPPL.is_dynamic(AbstractPPL.getoptic(x)) &&
        (template = VarNamedTuples.materialize_template(template))
    x = _concretize_prefix(x, template; prefix=Val(true))
    template = VarNamedTuples.materialize_template(template)
    model = _materialize_argument_values(model)
    values =
        if model.values isa VarNamedTuple &&
            getprefix(model) === nothing &&
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
    prefix = if template isa NoTemplate
        _compose_prefix_templates(x, model.prefix)
    else
        PrefixTemplate(x, template, model.prefix)
    end
    return _reconstruct_model(model; prefix, values)
end
function prefix(model::Model, ::Val{sym}) where {sym}
    return prefix(model, VarName{sym}())
end
function prefix(model::Model, x)
    return prefix(model, VarName{Symbol(x)}())
end

optic_skip_length(::AbstractPPL.Iden) = 0
optic_skip_length(optic::AbstractPPL.Index) = 1 + optic_skip_length(optic.child)
optic_skip_length(optic::AbstractPPL.Property) = 1 + optic_skip_length(optic.child)

function _prefix_varname_and_template(vn::VarName, template::Any, model::Model)
    return _prefix_varname_and_template(vn, template, model.prefix)
end
function _prefix_varname_and_template(vn::VarName, template, prefix)
    prefix === nothing && return vn, template
    return (
        AbstractPPL.prefix(vn, _getprefix(prefix)), _apply_prefix_template(prefix, template)
    )
end

function tilde_assume!!(
    model::Model,
    context::Context,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return tilde_assume!!(context, right, vn, template, vi)
end

function _check_tilde_value(value, vn, ::Union{Condition,Fix})
    placeholder = _classify_placeholder(value)
    absent = if placeholder === _MissingPlaceholder
        "missing"
    elseif placeholder === _NothingPlaceholder
        "nothing"
    else
        nothing
    end
    absent === nothing || throw(
        ArgumentError(
            "LHS variable `$vn` contains `$absent`. Only a whole `missing`/`nothing` " *
            "argument or a `missing` element `x[i]`/`x[i, j]` of a top-level array " *
            "argument marks data unobserved; otherwise use " *
            "`decondition(model, @varname($vn))`, or decondition the enclosing value " *
            "if it is a tuple or struct.",
        ),
    )
    return value
end

"""
    tilde_observe!!(prefix, right::Distribution, left, vn, template, vi)

Accumulate an observation and return `(left, vi)` with the updated varinfo.

`left` is supplied by the model's conditioned values or by a literal expression. `vn` is
the variable name before prefixing, or `nothing` for a literal. `template` describes the
top-level variable's storage; literals use `NoTemplate()`.

Apply `prefix`, the model's single prefix value (`nothing`, a `VarName`, or a
`PrefixTemplate` containing storage metadata), then delegate to [`accumulate_observe!!`](@ref).
The compiler passes this metadata directly so observations do not box the model.
Every observation calls this function, independently
of the evaluation context. Fixed LHS variables bypass it and do not contribute to the log probability.
"""
function tilde_observe!!(prefix, right::Distribution, left, vn, template, vi)
    vn, template = if vn === nothing
        vn, NoTemplate()
    else
        _prefix_varname_and_template(vn, template, prefix)
    end
    left = _check_tilde_value(left, vn, Condition())
    vi = accumulate_observe!!(vi, right, _nonmissing_eltype(left), vn, template)
    return left, vi
end

# Distributions take `AbstractArray{<:Real}`, so narrow a `Union{Missing,T}` array, which
# holds no `missing` after `_check_tilde_value`. `convert` returns other arrays uncopied.
function _nonmissing_eltype(x::AbstractArray{T}) where {T}
    return convert(AbstractArray{nonmissingtype(T)}, x)
end
_nonmissing_eltype(x) = x

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

Construct a `Context` and evaluate `model`, resetting and collecting the requested outputs.

The initialisation strategy supplies latent values. The transform strategy defaults to
`UnlinkAll()`, independently of the contents of `varinfo`. To reuse previous outputs,
explicitly pass `InitFromParams(get_vector_values(previous), nothing)` and the desired
transform strategy.

Returns a tuple of the model's return value, plus the updated `varinfo` object.
"""
function init!!(
    rng::Random.AbstractRNG,
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
)
    ctx = Context(rng, init_strategy, transform_strategy)
    return AbstractPPL.evaluate!!(model, ctx, vi)
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
    evaluate!!(model::Model, context::Context, varinfo::AbstractVarInfo)

Reset the accumulators and evaluate `model` using `context`, returning `(retval, varinfo)`.

The context belongs to this evaluation, not to the model. The same context is passed to
submodels and to [`tilde_assume!!`](@ref) for latent LHS variables. Observations and tracked values
go directly to accumulators, independently of the context. Models marked with
[`setthreadsafe`](@ref) use a `ThreadSafeVarInfo` during evaluation.

The [`Context`](@ref) supplies an RNG, initialisation strategy, and transform strategy.
The [`VarInfo`](@ref) contains only output accumulators. The convenience function
[`init!!`](@ref) constructs a `Context` and calls this method. Latent inputs are never
read from the output `varinfo`.

# Examples

```jldoctest
julia> using Random: Xoshiro

julia> @model example(y) = (x ~ Normal(); y ~ Normal(x); return x + y);

julia> ctx = Context(Xoshiro(1), InitFromParams((; x=1.0)), UnlinkAll());

julia> retval, vi = evaluate!!(example(2.0), ctx, VarInfo());

julia> retval
3.0
```
"""
function AbstractPPL.evaluate!!(model::Model, context::Context, varinfo::AbstractVarInfo)
    return if requires_threadsafe(model)
        # Thread-local accumulators must accept AD values before evaluation starts.
        param_eltype = DynamicPPL.get_param_eltype(context)
        wrapper = ThreadSafeVarInfo(varinfo, param_eltype)
        result, wrapper_new = _evaluate!!(model, context, wrapper)
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
        _evaluate!!(model, context, resetaccs!!(varinfo))
    end
end

"""
    _evaluate!!(model::Model, context::Context, varinfo)

Evaluate the `model` with the given `context` and `varinfo`.

This function does not wrap the varinfo in a `ThreadSafeVarInfo`. It also does not
reset the log probability of the `varinfo` before running.
"""
function _evaluate!!(model::Model, context::Context, varinfo::AbstractVarInfo)
    args, kwargs = make_evaluate_args_and_kwargs(model, context, varinfo)
    return model.f(args...; kwargs...)
end

"""
    make_evaluate_args_and_kwargs(model, context, varinfo)

Return the positional and keyword arguments for `model.f`, including the evaluation context.

The positional arguments begin with `(model, context, varinfo)`, followed by the model
arguments converted for the parameter element type. Pass the result to
`model.f(args...; kwargs...)` when a downstream evaluator, such as a taped task, controls
execution directly. This prepares arguments without executing the model, resetting
accumulators, or wrapping `varinfo` for thread safety; use [`evaluate!!`](@ref) otherwise.
"""
@generated function make_evaluate_args_and_kwargs(
    model::Model{_F,argnames,defaultnames}, context::Context, varinfo::AbstractVarInfo
) where {_F,argnames,defaultnames}
    unwrap_args = [
        if is_splat_symbol(var)
            :($convert_model_argument(varinfo, context, model.args.$var)...)
        else
            :($convert_model_argument(varinfo, context, model.args.$var))
        end for var in argnames
    ]
    unwrap_kwargs = [
        is_splat_symbol(var) ? :(model.defaults.$var...) : :($var = model.defaults.$var) for
        var in defaultnames
    ]
    return quote
        args = (model, context, varinfo, $(unwrap_args...))
        kwargs = (; $(unwrap_kwargs...))
        return args, kwargs
    end
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
