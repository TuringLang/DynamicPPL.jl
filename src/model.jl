#
# Model values
#

struct Condition end
struct Fix end

_contains_missing(::Any) = false
_contains_missing(::Missing) = true
_contains_missing(::AbstractArray{<:Number}) = false
function _contains_missing(values::Union{AbstractArray,Tuple,NamedTuple})
    return any(_contains_missing, values)
end

struct ModelValue{R<:Union{Condition,Fix},T}
    value::T
    function ModelValue{R}(value::T) where {R<:Union{Condition,Fix},T}
        (R === Condition || R === Fix) ||
            throw(ArgumentError("A model value must have one concrete role"))
        _contains_missing(value) && throw(
            ArgumentError(
                "`missing` no longer selects latent variables. Omit unobserved values from `condition` or `fix` instead.",
            ),
        )
        return new{R,T}(value)
    end
end

function VarNamedTuples._getindex_optic(
    value::ModelValue{R}, optic::AbstractPPL.AbstractOptic, vn
) where {R}
    return ModelValue{R}(VarNamedTuples._getindex_optic(value.value, optic, vn))
end
function VarNamedTuples._getindex_optic(
    value::ModelValue{R}, ::AbstractPPL.Iden, vn
) where {R}
    return value
end
function VarNamedTuples._haskey_optic(value::ModelValue, optic::AbstractPPL.AbstractOptic)
    return VarNamedTuples._haskey_optic(value.value, optic)
end
VarNamedTuples._haskey_optic(::ModelValue, ::AbstractPPL.Iden) = true

_model_value(value::ModelValue, ::VarName) = value
function _model_value(values::VarNamedTuples.PartialArray, vn::VarName)
    return _model_value(VarNamedTuples.unwrap_internal_array(values), vn)
end
function _model_value(values::AbstractArray, vn::VarName)
    isempty(values) && throw(ArgumentError("Cannot determine the role of empty `$vn`"))
    first_value = first(values)
    first_value isa ModelValue || throw(ArgumentError("Invalid model value for `$vn`"))
    return _model_value(values, vn, first_value)
end
function _model_value(values::AbstractArray, vn::VarName, ::ModelValue{R}) where {R}
    all(value -> value isa ModelValue{R}, values) || throw(
        ArgumentError(
            "Cannot condition and fix different parts of the same tilde variable `$vn`"
        ),
    )
    return ModelValue{R}(map(value -> value.value, values))
end
function _model_value(values::VarNamedTuple, vn::VarName)
    tagged = map(value -> _model_value(value, vn), values.data)
    isempty(tagged) && throw(ArgumentError("Cannot determine the role of empty `$vn`"))
    return _model_value(tagged, vn, first(tagged))
end
function _model_value(values::NamedTuple, vn::VarName, ::ModelValue{R}) where {R}
    all(value -> value isa ModelValue{R}, values) || throw(
        ArgumentError(
            "Cannot condition and fix different parts of the same tilde variable `$vn`"
        ),
    )
    return ModelValue{R}(map(value -> value.value, values))
end
function _model_value(values, vn::VarName)
    throw(ArgumentError("Invalid model value for `$vn`: $(typeof(values))"))
end

function _get_model_value(model, vn)
    return hasvalue(model.values, vn) ? _model_value(model.values[vn], vn) : nothing
end

function _tag_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return map_values!!(ModelValue{R}, copy(values))
end

function VarNamedTuples._merge(
    previous::ModelValue{R,<:NamedTuple}, updates::VarNamedTuple, recurse::Val{true}
) where {R}
    return VarNamedTuples._merge(
        _tag_model_values(R, VarNamedTuple(previous.value)), updates, recurse
    )
end
function VarNamedTuples._merge(
    previous::ModelValue{R}, updates::VarNamedTuple, recurse::Val{true}
) where {R}
    all(name -> hasproperty(previous.value, name), keys(updates.data)) || throw(
        ArgumentError(
            "Cannot override nonexistent properties of $(typeof(previous.value))"
        ),
    )
    names = propertynames(previous.value)
    fields = NamedTuple{names}(map(name -> getproperty(previous.value, name), names))
    return VarNamedTuples._merge(
        _tag_model_values(R, VarNamedTuple(fields)), updates, recurse
    )
end

function VarNamedTuples._merge(
    previous::ModelValue{R,<:AbstractArray},
    updates::VarNamedTuples.PartialArray,
    recurse::Val{true},
) where {R}
    data = map(ModelValue{R}, previous.value)
    complete = VarNamedTuples.PartialArray(data, fill!(similar(data, Bool), true))
    return _merge_model_array(complete, updates, previous.value, recurse)
end
function VarNamedTuples._merge(
    previous::ModelValue{R,<:AbstractArray}, updates::VarNamedTuple, recurse::Val{true}
) where {R}
    data = map(ModelValue{R}, previous.value)
    complete = VarNamedTuples.PartialArray(data, fill!(similar(data, Bool), true))
    return _merge_model_array(complete, updates, previous.value, recurse)
end
function VarNamedTuples._merge(
    previous::VarNamedTuples.PartialArray{<:ModelValue},
    updates::VarNamedTuple,
    recurse::Val{true},
)
    return _merge_model_array(previous, updates, previous.data, recurse)
end
function VarNamedTuples._merge(
    previous::VarNamedTuples.PartialArray{<:ModelValue},
    updates::VarNamedTuples.PartialArray,
    recurse::Val{true},
)
    return _merge_model_array(previous, updates, previous.data, recurse)
end
function _merge_model_array(previous, updates, template, recurse)
    values = mapfoldl(
        identity,
        (values, pair) ->
            merge(values, _template_model_value(pair.second, pair.first, template)),
        VarNamedTuple(; x=updates);
        init=VarNamedTuple(),
    )
    isempty(values) && return previous
    normalized = values.data.x
    normalized isa VarNamedTuples.PartialArray ||
        throw(ArgumentError("Cannot apply property updates to $(typeof(template))"))
    return invoke(
        VarNamedTuples._merge,
        Tuple{VarNamedTuples.PartialArray,VarNamedTuples.PartialArray,Val},
        previous,
        normalized,
        recurse,
    )
end
function _template_model_value(value::ModelValue{R}, vn, template) where {R}
    return _tag_model_values(
        R, templated_setindex!!(VarNamedTuple(), value.value, vn, template)
    )
end

_model_data(value::ModelValue) = value.value
_model_data(values::AbstractArray) = map(_model_data, values)
_model_data(values::VarNamedTuple) = map(_model_data, values.data)
function _model_data(values::VarNamedTuples.PartialArray)
    return _model_data(VarNamedTuples.unwrap_internal_array(values))
end

_model_argument_value(::Nothing, template) = deepcopy(template)
_model_argument_value(value::ModelValue, template) = value.value
_model_argument_value(values::AbstractArray, template) = _model_data(values)
function _model_argument_value(values::VarNamedTuples.PartialArray, template)
    # Growable arrays describe supplied indices, not the extent of the argument.
    return if !(values.data isa VarNamedTuples.GrowableArray) &&
        VarNamedTuples._haskey_optic(values, AbstractPPL.Iden())
        _model_data(values)
    else
        deepcopy(template)
    end
end
@generated function _model_argument_value(
    values::VarNamedTuple{names}, template
) where {names}
    updates = map(names) do name
        :(
            result = Accessors.set(
                result,
                AbstractPPL.with_mutation(AbstractPPL.Property{$(QuoteNode(name))}()),
                _model_argument_value(
                    values.data.$name,
                    VarNamedTuples.SharedGetProperty{$(QuoteNode(name))}()(result),
                ),
            )
        )
    end
    return quote
        template isa NoTemplate && return _model_data(values)
        result = deepcopy(template)
        $(updates...)
        return result
    end
end

function _select_model_values(::Type{R}, values::VarNamedTuple) where {R}
    return mapfoldl(
        identity,
        function (selected, pair)
            vn, value = pair
            return if value isa ModelValue{R}
                templated_setindex!!(
                    selected, value.value, vn, values.data[AbstractPPL.getsym(vn)]
                )
            else
                selected
            end
        end,
        values;
        init=VarNamedTuple(),
    )
end

#
# Model definition
#

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

"""
    struct Model{
        F,
        argnames,
        defaultnames,
        Targs,
        Tdefaults,
        Ctx<:AbstractContext,
        Prefix<:Union{VarName,Nothing},
        PrefixTemplate,
        Values<:VarNamedTuple,
        Threaded,
    }
        f::F
        args::NamedTuple{argnames,Targs}
        defaults::NamedTuple{defaultnames,Tdefaults}
        context::Ctx=DefaultContext()
        prefix::Prefix=nothing
        prefix_template::PrefixTemplate=nothing
        values::Values
    end

A `Model` struct with model evaluation function of type `F`, arguments of names `argnames`
types `Targs`, default arguments of names `defaultnames` with types `Tdefaults`,
and evaluation context of type `Ctx`. Conditioned and fixed values
share one store, with each value carrying its role.

Here `argnames` and `defaultnames` are tuples of symbols, e.g. `(:a, :b)`.
`context` is by default `DefaultContext()`.

Model arguments supply default conditioned values. `condition` replaces these observations,
and `decondition` removes them, making the corresponding stochastic sites latent.
Arguments not used at stochastic sites remain ordinary Julia data.

The `Threaded` type parameter indicates whether the model requires threadsafe evaluation
(i.e., whether the model contains statements which modify the internal VarInfo that are
executed in parallel). By default, this is set to `false`.

The default arguments are used internally when constructing instances of the same model with
different arguments.

# Examples

```julia
julia> Model(f, (x = 1.0, y = 2.0)).args
(x = 1.0, y = 2.0)

julia> Model(f, (x = 1.0, y = 2.0), (x = 42,)).defaults
(x = 42,)

```
"""
struct Model{
    F,
    argnames,
    defaultnames,
    Targs,
    Tdefaults,
    Ctx<:AbstractContext,
    Prefix<:Union{VarName,Nothing},
    PT,
    Values<:VarNamedTuple,
    Threaded,
} <: AbstractProbabilisticProgram
    f::F
    args::NamedTuple{argnames,Targs}
    defaults::NamedTuple{defaultnames,Tdefaults}
    context::Ctx
    prefix::Prefix
    prefix_template::PT
    values::Values

    function Model{Threaded}(
        f::F,
        args::NamedTuple{argnames,Targs},
        defaults::NamedTuple{defaultnames,Tdefaults},
        context::Ctx=DefaultContext(),
        prefix::Prefix=nothing,
        values::Values=_tag_model_values(Condition, VarNamedTuple(merge(args, defaults))),
        prefix_template::PT=nothing,
    ) where {F,argnames,Targs,defaultnames,Tdefaults,Ctx,Prefix,PT,Values,Threaded}
        mapreduce(pair -> pair.second isa ModelValue, &, values; init=true) ||
            throw(ArgumentError("Model values must carry a condition or fix role"))
        return new{F,argnames,defaultnames,Targs,Tdefaults,Ctx,Prefix,PT,Values,Threaded}(
            f, args, defaults, context, prefix, prefix_template, values
        )
    end
end

"""
    Model(f, args::NamedTuple[, defaults::NamedTuple = ()])

Create a model with evaluation function `f` and arguments `args`.

Arguments supply default conditioned values. Use [`decondition`](@ref) to make an
argument-backed stochastic site latent, or [`condition`](@ref) to replace its observation.

Default arguments `defaults` are used internally when constructing instances of the same
model with different arguments.
"""
function Model{Threaded}(
    f, args::NamedTuple, context::AbstractContext=DefaultContext(); kwargs...
) where {Threaded}
    return Model{Threaded}(f, args, NamedTuple(kwargs), context)
end

"""
    requires_threadsafe(model::Model)

Return whether `model` has been marked as needing threadsafe evaluation (using
`setthreadsafe`).
"""
function requires_threadsafe(
    ::Model{F,A,D,Ta,Td,Ctx,P,PT,V,Threaded}
) where {F,A,D,Ta,Td,Ctx,P,PT,V,Threaded}
    return Threaded
end

function _reconstruct_model(
    model::Model{F,A,D,Ta,Td,Ctx,P,PT,V,Threaded};
    context::AbstractContext=model.context,
    prefix::Union{VarName,Nothing}=model.prefix,
    values::VarNamedTuple=model.values,
    prefix_template=model.prefix_template,
) where {F,A,D,Ta,Td,Ctx,P,PT,V,Threaded}
    return Model{Threaded}(
        model.f, model.args, model.defaults, context, prefix, values, prefix_template
    )
end

"""
    contextualize(model::Model, context::AbstractContext)

Return a new `Model` with the same evaluation function and other arguments, but
with its underlying context set to `context`.
"""
function contextualize(model::Model, context::AbstractContext)
    return _reconstruct_model(model; context)
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
        Model{threadsafe}(
            model.f,
            model.args,
            model.defaults,
            model.context,
            model.prefix,
            model.values,
            model.prefix_template,
        )
    end
end

"""
    model | (x = 1.0, ...)

Return a `Model` which now treats variables on the right-hand side as observations.

See [`condition`](@ref) for more information and examples.
"""
Base.:|(model::Model, values::Union{NamedTuple,AbstractDict,Pair,Tuple,VarNamedTuple}) =
    condition(model, values)

"""
    condition(model::Model; values...)
    condition(model::Model, values::NamedTuple)

Return a `Model` which now treats the variables in `values` as observations.

See also: [`decondition`](@ref), [`conditioned`](@ref)

Supplied values override model arguments and earlier conditioned or fixed values at the
same address. Parent-model values override submodel values. Sites without supplied values
remain latent; `missing` is not a latent-variable marker.

A complete argument replacement supplies its value, shape, and dispatch type parameters
from the start of the model body. Partial updates preserve the remaining stored values and
their array templates. Arguments with unobserved entries retain their original storage
template; the corresponding tilde statements fill those entries during evaluation.

# Examples
## Simple univariate model
```jldoctest condition
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> m, x = model(); (m != 1.0 && x != 100.0)
true

julia> # Create a new instance which treats `x` as observed
       # with value `100.0`, and similarly for `m=1.0`.
       conditioned_model = condition(model, x=100.0, m=1.0);

julia> m, x = conditioned_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only condition on `x = 100.0`.
       conditioned_model = condition(model, x = 100.0);

julia> m, x = conditioned_model(); (m != 1.0 && x == 100.0)
true

julia> # We can also use the nicer `|` syntax.
       conditioned_model = model | (x = 100.0, );

julia> m, x = conditioned_model(); (m != 1.0 && x == 100.0)
true
```

In the above we have specified the conditioning variables via keyword arguments. You can also
provide a `NamedTuple`, `AbstractDict{<:VarName}`, or a `VarNamedTuple`; internally these are
all converted to a `VarNamedTuple`.

For example, here we use a `Dict`:

```jldoctest condition
julia> conditioned_model_dict = condition(model, Dict(@varname(x) => 100.0));

julia> m, x = conditioned_model_dict(); (m != 1.0 && x == 100.0)
true

julia> # There's also an option using `|` by letting the right-hand side be a tuple
       # with elements of type `Pair{<:VarName}`, i.e. `vn => value` with `vn isa VarName`.
       conditioned_model_pairs = model | (@varname(x) => 100.0);

julia> m, x = conditioned_model_pairs(); (m != 1.0 && x == 100.0)
true
```

## Condition only a part of a multivariate variable

Supply only the indices to observe; omitted sites remain latent.

However, note that in this case each element of the multivariate random variable must be on
its own tilde-statement. In other words, if we write `m ~ MvNormal(...)`, then we cannot
condition on only `m[1]`. Attempting to do so may abort model evaluation with an unrelated
`DimensionMismatch`, or the conditioning may be silently ignored, with `m` sampled afresh.
(In principle, for some distributions this can be possible, specifically when the
distribution can be factorised into independent components, like an MvNormal with a
diagonal covariance matrix. However, this is not currently implemented.)

```jldoctest condition
julia> @model function demo_mv(::Type{TV}=Float64) where {TV}
           m = Vector{TV}(undef, 2)
           m[1] ~ Normal()
           m[2] ~ Normal()
           return m
       end
demo_mv (generic function with 4 methods)

julia> model = demo_mv();

julia> observations = @vnt begin
           @template m=zeros(2)
           m[2] := 1.0
       end;

julia> conditioned_model = condition(model, observations);

julia> # `m[1]` is sampled while `m[2]` is observed.
       m = conditioned_model(); (m[1] != 1.0 && m[2] == 1.0)
true
```

Intuitively one might also expect to be able to write `model | (m[2] = 1.0, )`. You cannot
do this with a `NamedTuple` because the `VarName` `m[2]` cannot be represented as a `Symbol`
(i.e., `Symbol("m[2]")` is not the same as `@varname(m[2])`).

```jldoctest condition
julia> # (×) `m[2]` is not set to 1.0.
       m = condition(model, var"m[2]" = 1.0)(); m[2] == 1.0
false
```

But you _can_ do this if you use a `Dict` or a `VarNamedTuple` as the underlying storage
instead:

```jldoctest condition
julia> vnt = @vnt begin
           @template m = zeros(2)
           m[2] := 1.0
       end
VarNamedTuple
└─ m => PartialArray size=(2,) data::Vector{Float64}
        └─ (2,) => 1.0

julia> m = condition(model, vnt)(); (m[1] != 1.0 && m[2] == 1.0)
true
```

## Nested models

`condition` also supports the use of nested models through the use of [`to_submodel`](@ref).

```jldoctest condition
julia> @model demo_inner() = m ~ Normal()
demo_inner (generic function with 2 methods)

julia> @model function demo_outer()
           # By default, `to_submodel` prefixes the variables using the left-hand side of `~`.
           inner ~ to_submodel(demo_inner())
           return inner
       end
demo_outer (generic function with 2 methods)

julia> model = demo_outer();

julia> model() ≠ 1.0
true

julia> # To condition the variable inside `demo_inner` we need to refer to it as `inner.m`.
       conditioned_model = model | (@varname(inner.m) => 1.0, );

julia> conditioned_model()
1.0

julia> # If you attempt to condition on `inner` itself, it must refer to the prefixed
       # latent variables, not the return value. For example, this will work:
       conditioned_model2 = model | (inner = (m = 1.0,), );

julia> conditioned_model2()
1.0

julia> # Conditioning a submodel's return value is not supported.
       conditioned_model_fail = model | (inner = "something else", );

julia> conditioned_model_fail()
ERROR: ArgumentError: Cannot condition or fix a submodel's return value. Supply its internal variable names instead; decondition arguments used only as return-value buffers.
```
"""
function AbstractPPL.condition(model::Model, values...)
    values = merge(
        model.values, _tag_model_values(Condition, _make_condfix_values(values...))
    )
    return _reconstruct_model(model; values)
end
function AbstractPPL.condition(model::Model; values...)
    return condition(model, NamedTuple(values))
end

"""
    _make_condfix_values(vals...)

Convert different types of input to a `VarNamedTuple` of values, suitable for storage in a
`Model`.

This handles all the cases where `vals` is either already a `NamedTuple` or `AbstractDict`
(e.g. `model | (x=1, y=2)`), as well as if they are splatted (e.g. `condition(model, x=1,
y=2)`).
"""
_make_condfix_values(values::NamedTuple) = VarNamedTuple(values)
_make_condfix_values(values::VarNamedTuple) = values
_make_condfix_values(values::AbstractDict{<:VarName}) = VarNamedTuple(pairs(values))
function _make_condfix_values(values::Pair{<:Union{VarName,Symbol}}...)
    pairs = map(
        v -> ((v.first isa Symbol ? VarName{v.first}() : v.first) => v.second), values
    )
    return VarNamedTuple(pairs)
end
function _make_condfix_values(values::NTuple{N,Pair{<:Union{VarName,Symbol}}}) where {N}
    return _make_condfix_values(values...)
end

"""
    decondition(model::Model)
    decondition(model::Model, variables...)

Return a `Model` for which `variables...` are _not_ conditioned on. If no `variables` are
provided, then all conditioned variables will be removed.

This also removes observations supplied as model arguments. After deconditioning, a site's
sampled value replaces its local argument value and is used by subsequent model statements.

This is essentially the inverse of [`condition`](@ref).

# Examples
```jldoctest decondition
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> conditioned_model = condition(demo(), m = 1.0, x = 10.0);

julia> conditioned_model()
(m = 1.0, x = 10.0)

julia> # By specifying the `VarName` to `decondition`.
       model = decondition(conditioned_model, @varname(m));

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # `decondition` also accepts symbols, although VarNames are preferable for
       # type stability reasons.
       model = decondition(conditioned_model, :m);

julia> (m, x) = model(); (m ≠ 1.0 && x == 10.0)
true

julia> # `decondition` multiple at once:
       (m, x) = decondition(model, :m, :x)(); (m ≠ 1.0 && x ≠ 10.0)
true

julia> # `decondition` without any symbols will `decondition` all variables.
       (m, x) = decondition(model)(); (m ≠ 1.0 && x ≠ 10.0)
true
```

Note that `decondition` is only guaranteed to work when you decondition variables that were
explicitly provided to `condition` earlier. In this example we condition on `@varname(m)`
but decondition on `@varname(m[1])`, which fails because `m[1]` was not explicitly
conditioned on:

```jldoctest decondition
julia> @model function demo_mv(::Type{TV}=Float64) where {TV}
           m = Vector{TV}(undef, 2)
           m[1] ~ Normal()
           m[2] ~ Normal()
           return m
       end
demo_mv (generic function with 4 methods)

julia> model = demo_mv();

julia> conditioned_model = condition(model, @varname(m) => [1.0, 2.0]);

julia> conditioned_model()
2-element Vector{Float64}:
 1.0
 2.0

julia> deconditioned_model = decondition(conditioned_model, @varname(m[1]));

julia> deconditioned_model()  # (×) `m[1]` is still conditioned
2-element Vector{Float64}:
 1.0
 2.0
```
"""
function AbstractPPL.decondition(model::Model, syms::Union{Symbol,VarName}...)
    values = _remove_model_values(Condition, model.values, syms...)
    return _reconstruct_model(model; values)
end

function _remove_model_values(
    ::Type{R}, values::VarNamedTuple, args::Union{Symbol,VarName}...
) where {R}
    vns = map(arg -> arg isa VarName ? arg : VarName{arg}(), args)
    retained_keys = filter(keys(values)) do key
        !(values[key] isa ModelValue{R}) ||
            (!isempty(args) && all(vn -> !subsumes(vn, key), vns))
    end
    return subset(values, retained_keys)
end

"""
    conditioned(model::Model)

Return the conditioned values in `model`.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: conditioned, prefix

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns all the variables we have conditioned on + their values.
       conditioned(condition(m, x=100.0, m=1.0))
VarNamedTuple
├─ x => 100.0
└─ m => 1.0

julia> # Prefixing also prefixes values already stored on the model.
       cm = condition(m, m=1.0) |> model -> prefix(model, @varname(a));

julia> conditioned(cm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Since we conditioned on `a.m`, it is not treated as a random variable.
       # However, `a.x` is still a random variable.
       keys(VarInfo(cm))
1-element Vector{VarName}:
 a.x

julia> # Values added after prefixing use their supplied names unchanged.
       cm = condition(prefix(m, @varname(a)), (@varname(a.m) => 1.0));

julia> conditioned(cm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Now `a.x` will be sampled.
       keys(VarInfo(cm))
1-element Vector{VarName}:
 a.x
```
"""
conditioned(model::Model) = _select_model_values(Condition, model.values)

"""
    fix(model::Model; values...)
    fix(model::Model, values::NamedTuple)

Return a `Model` which now treats the variables in `values` as fixed.

See also: [`unfix`](@ref), [`fixed`](@ref)

!!! warning "Fixing applies to whole variables"
    Variables are treated as they occur in the model. A variable drawn from a multivariate
    distribution in a single tilde-statement (e.g. `x ~ MvNormal(...)`) is a *single* random
    variable, so a subset of its components cannot be fixed independently; only fixing the
    variable in its entirety is supported. Attempting to fix a subset may silently collapse
    the variable to just the supplied components, or leave it entirely unfixed and sampled
    from the prior. Declare components in a loop (`x[i] ~ ...`) if you need to fix them
    individually.

# Examples
## Simple univariate model
```jldoctest fix
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> m, x = model(); (m ≠ 1.0 && x ≠ 100.0)
true

julia> # Create a new instance which treats `x` as observed
       # with value `100.0`, and similarly for `m=1.0`.
       fixed_model = fix(model, x=100.0, m=1.0);

julia> m, x = fixed_model(); (m == 1.0 && x == 100.0)
true

julia> # Let's only fix on `x = 100.0`.
       fixed_model = fix(model, x = 100.0);

julia> m, x = fixed_model(); (m != 1.0 && x == 100.0)
true
```

## Other ways of specifying fixed values

Specifying fixed values can be done exactly in the same way as for [`condition`](@ref);
please see its docstring for more examples.

## Difference from `condition`

The only difference between fixing and conditioning is as follows:

- Conditioned variables are considered to be observations, and are thus included in the
  computation log-joint and log-likelihood, but not the log-prior.
- Fixed variables are considered to be constant, and are thus not included
  in any log-probability computations.

```jldoctest; setup=:(using DynamicPPL, Distributions)
julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> model_fixed = fix(model, m = 1.0);

julia> model_conditioned = condition(model, m = 1.0);

julia> logjoint(model_fixed, (x=1.0,))
-0.9189385332046728

julia> logjoint(model_conditioned, (x=1.0,))
-2.3378770664093453

julia> # The difference is the missing log-probability of `m`:
       logpdf(Normal(), 1.0)
-1.4189385332046727
```
"""
function fix(model::Model, values...)
    values = merge(model.values, _tag_model_values(Fix, _make_condfix_values(values...)))
    return _reconstruct_model(model; values)
end
function fix(model::Model; values...)
    return fix(model, NamedTuple(values))
end

"""
    unfix(model::Model)
    unfix(model::Model, variables...)

Return a `Model` for which `variables...` are _not_ considered fixed. If no `variables` are
provided, then all fixed variables will be removed.

This is essentially the inverse of [`fix`](@ref).

Conceptually this is very similar to [`decondition`](@ref) and thus the same limitations
apply; please see its docstring for more details.

# Examples
```jldoctest unfix
julia> using Distributions

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
           return (; m=m, x=x)
       end
demo (generic function with 2 methods)

julia> fixed_model = fix(demo(), m = 1.0, x = 10.0);

julia> fixed_model()
(m = 1.0, x = 10.0)

julia> # By specifying the `VarName` to `unfix`.
       model = unfix(fixed_model, @varname(m));

julia> (m, x) = model(); (m != 1.0 && x == 10.0)
true

julia> # When `NamedTuple` is used as the underlying, you can also provide
       # the symbol directly (though the `@varname` approach is preferable if
       # if the variable is known at compile-time).
       model = unfix(fixed_model, :m);

julia> (m, x) = model(); (m != 1.0 && x == 10.0)
true

julia> # `unfix` multiple at once:
       (m, x) = unfix(model, :m, :x)(); (m != 1.0 && x != 10.0)
true

julia> # `unfix` without any symbols will `unfix` all variables.
       (m, x) = unfix(model)(); (m != 1.0 && x != 10.0)
true
```
"""
function unfix(model::Model, syms::Union{Symbol,VarName}...)
    values = _remove_model_values(Fix, model.values, syms...)
    return _reconstruct_model(model; values)
end

"""
    fixed(model::Model)

Return the fixed values in `model`.

# Examples
```jldoctest
julia> using Distributions

julia> using DynamicPPL: fixed, prefix

julia> @model function demo()
           m ~ Normal()
           x ~ Normal(m, 1)
       end
demo (generic function with 2 methods)

julia> m = demo();

julia> # Returns all the variables we have fixed on + their values.
       fixed(fix(m, x=100.0, m=1.0))
VarNamedTuple
├─ x => 100.0
└─ m => 1.0

julia> # Prefixing also prefixes values already stored on the model.
       fm = prefix(fix(m, m=1.0), @varname(a));

julia> fixed(fm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> keys(VarInfo(fm))
1-element Vector{VarName}:
 a.x

julia> # Values added after prefixing use their supplied names unchanged.
       fm = fix(prefix(m, @varname(a)), (@varname(a.m) => 1.0));

julia> fixed(fm)
VarNamedTuple
└─ a => VarNamedTuple
        └─ m => 1.0

julia> # Now `a.x` will be sampled.
       keys(VarInfo(fm))
1-element Vector{VarName}:
 a.x
```
"""
fixed(model::Model) = _select_model_values(Fix, model.values)

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
    x = _concretize_prefix(x, template)
    model_prefix = maybe_prefix(model.prefix, x)
    values = _prefix_values(model.values, x, template)
    prefix_template = if template isa NoTemplate && model.prefix_template === nothing
        nothing
    else
        inner = model.prefix_template === nothing ? model.prefix : model.prefix_template
        PrefixTemplate(x, template, inner)
    end
    return _reconstruct_model(model; prefix=model_prefix, values, prefix_template)
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
    prefix = model.prefix
    prefix === nothing && return vn, template
    pt = model.prefix_template === nothing ? prefix : model.prefix_template
    return AbstractPPL.prefix(vn, prefix), _apply_prefix_template(pt, template)
end

function tilde_assume!!(
    model::Model, right::Distribution, vn::VarName, template::Any, vi::AbstractVarInfo
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return tilde_assume!!(model.context, right, vn, template, vi)
end

function tilde_observe!!(
    model::Model,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template::Any,
    vi::AbstractVarInfo,
)
    vn, template = if vn === nothing
        vn, NoTemplate()
    else
        _prefix_varname_and_template(vn, template, model)
    end
    return tilde_observe!!(model.context, right, left, vn, template, vi)
end

function store_coloneq_value!!(
    model::Model, vn::VarName, right::Any, template::Any, vi::AbstractVarInfo
)
    vn, template = _prefix_varname_and_template(vn, template, model)
    return store_coloneq_value!!(model.context, vn, right, template, vi)
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
function (model::Model)(
    rng::Random.AbstractRNG, varinfo::AbstractVarInfo=OnlyAccsVarInfo(())
)
    return first(init!!(rng, model, varinfo, InitFromPrior(), UnlinkAll()))
end

"""
    init!!(
        [rng::Random.AbstractRNG,]
        model::Model,
        varinfo::AbstractVarInfo,
        init_strategy::AbstractInitStrategy,
        [transform_strategy::AbstractTransformStrategy=get_transform_strategy(varinfo),]
    )

Evaluate the `model` and replace the values of the model's random variables in the given
`varinfo` with new values, using a specified initialisation strategy. If the values in
`varinfo` are not set, they will be added using a specified initialisation strategy.

`transform_strategy` tells the model evaluation whether variables should be interpreted as
linked or unlinked. Right now, it is slightly complicated because the default behaviour
depends on the `varinfo` provided. If `varinfo isa VarInfo`, then the transform strategy is
inferred from the VarInfo, i.e., linked variables in the VarInfo are treated as linked
during evaluation. Conversely, if `varinfo isa OnlyAccsVarInfo`, then you must specify the
transform strategy explicitly, since an `OnlyAccsVarInfo` does not contain any information
about which variables are transformed.

Returns a tuple of the model's return value, plus the updated `varinfo` object.
"""
function init!!(
    rng::Random.AbstractRNG,
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy,
    transform_strategy::AbstractTransformStrategy=get_transform_strategy(vi),
)
    ctx = InitContext(rng, init_strategy, transform_strategy)
    model = DynamicPPL.contextualize(model, ctx)
    return DynamicPPL.evaluate_nowarn!!(model, vi)
end
function init!!(
    model::Model,
    vi::AbstractVarInfo,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
    transform_strategy::AbstractTransformStrategy=get_transform_strategy(vi),
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
    five-argument `init!!([rng,] model, ::OnlyAccsVarInfo, init_strategy,
    transform_strategy)` method, which has more explicit semantics and allows you to have
    more control over each part of the evaluation process.

The exact semantics depend on the `model`'s context. Fundamentally, this method executes the
model evaluation function (i.e., the function used to define the model) using the given
`varinfo` as an argument. At each tilde-statement, `tilde_assume!!` or `tilde_observe!!` is
called, whose behaviour depends on the model's context.

Broadly speaking, if the context is an `InitContext`, then this function:

- uses the initialisation strategy inside the `InitContext`;
- uses the transform strategy inside the `InitContext`;
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- overwrites the values in `varinfo` with the new values obtained from the initialisation strategy.

If the context is a `DefaultContext`, then this function:

- uses the values inside the `varinfo` as the initialisation strategy;
- derives a transform strategy from the `varinfo`'s stored variables (if a linked variable is
  stored, then the transform strategy will treat that variable as linked; likewise for
  unlinked)
- uses the accumulators inside `varinfo` (resetting them before evaluation);
- does not overwrite the values in the `varinfo` (that is unnecessary since the values used
  for evaluation are already stored in `varinfo`).

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
        " ::OnlyAccsVarInfo, init_strategy, transform_strategy)` instead, which" *
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
        return result, setaccs!!(wrapper_new.varinfo, getaccs(wrapper_new))
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

is_splat_symbol(s::Symbol) = startswith(string(s), "#splat#")

"""
    make_evaluate_args_and_kwargs(model, varinfo)

Return the arguments and keyword arguments to be passed to the evaluator of the model, i.e. `model.f`e.
"""
@generated function make_evaluate_args_and_kwargs(
    model::Model{_F,argnames}, varinfo::AbstractVarInfo
) where {_F,argnames}
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
    return quote
        args = (model, varinfo, $(unwrap_args...))
        kwargs = model.defaults
        return args, kwargs
    end
end

"""
    get_param_eltype(varinfo::AbstractVarInfo, context::AbstractContext)

Get the element type of the parameters being used to evaluate a model, using a `varinfo`
under the given `context`. For example, when evaluating a model with ForwardDiff AD, this
should return `ForwardDiff.Dual`.

By default, this uses `eltype(varinfo)` which is slightly cursed. This relies on the fact
that typically, before evaluation, the parameters will have been inserted into the VarInfo's
metadata field.

For `InitContext`, it's quite different: because `InitContext` is responsible for supplying
the parameters, we can avoid using `eltype(varinfo)` and instead query the parameters inside
it. See the docstring of `get_param_eltype(strategy::AbstractInitStrategy)` for more
explanation.
"""
get_param_eltype(vi::AbstractVarInfo, ::AbstractContext) = eltype(vi)
function get_param_eltype(::AbstractVarInfo, ctx::InitContext)
    return get_param_eltype(ctx.strategy)
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
    vi = OnlyAccsVarInfo((RawValueAccumulator(false),))
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
demo (generic function with 2 methods)

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
    vi = OnlyAccsVarInfo(
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator())
    )
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
demo (generic function with 2 methods)

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
    vi = OnlyAccsVarInfo(AccumulatorTuple(LogPriorAccumulator()))
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
demo (generic function with 2 methods)

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
    vi = OnlyAccsVarInfo(AccumulatorTuple(LogLikelihoodAccumulator()))
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
            DynamicPPL.OnlyAccsVarInfo(DynamicPPL.AccumulatorTuple()),
            # Use `nothing` as the fallback to ensure that any missing parameters cause an
            # error
            InitFromParams(parameters..., nothing),
            UnlinkAll(),
        ),
    )
end
