const INTERNALNAMES = (:__model__, :__context__, :__varinfo__)

drop_escape(x) = x
function drop_escape(expr::Expr)
    Meta.isexpr(expr, :escape) && return drop_escape(expr.args[1])
    return Expr(expr.head, map(x -> drop_escape(x), expr.args)...)
end

get_top_level_symbol(expr::Symbol) = expr
function get_top_level_symbol(expr::Expr)
    # TODO(penelopeysm): what about Meta.isexpr(expr, :$)?
    if Meta.isexpr(expr, :ref)
        return get_top_level_symbol(expr.args[1])
    elseif Meta.isexpr(expr, :.)
        return get_top_level_symbol(expr.args[1])
    else
        error("unreachable")
    end
end

"""
    make_varname_expression(expr)

Return a `VarName` based on `expr`.
"""
function make_varname_expression(expr)
    # HACK: Usage of `drop_escape` is unfortunate. It's a consequence of the fact that in
    # DynamicPPL we the entire function body. Instead we should be more selective with our
    # escape. Until that's the case, we remove them all.
    return drop_escape(AbstractPPL.varname(expr, false))
end

"""
    contextual_isassumption(context, vn)

Return `true` if `vn` is considered an assumption by `context`.
"""
function contextual_isassumption(context::AbstractContext, vn)
    if hasconditioned_nested(context, vn)
        val = getconditioned_nested(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return true
        else
            return false
        end
    else
        return true
    end
end

"""
    contextual_isfixed(context, vn)

Return `true` if `vn` is considered fixed by `context`.
"""
function contextual_isfixed(context::AbstractContext, vn)
    if hasfixed_nested(context, vn)
        val = getfixed_nested(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return false
        else
            return true
        end
    else
        return false
    end
end

"""
    isliteral(expr)

Return `true` if `expr` is a literal, e.g. `1.0` or `[1.0, ]`, and `false` otherwise.
"""
isliteral(e) = false
isliteral(::Number) = true
function isliteral(e::Expr)
    # In the special case that the expression is of the form `abc[blahblah]`, we consider it
    # to be a literal if `abc` is a literal. This is necessary for cases like
    # [1.0, 2.0][idx...] ~ Normal()
    # which are generated when turning `.~` expressions into loops over `~` expressions.
    if e.head == :ref
        return isliteral(e.args[1])
    end
    return !isempty(e.args) && all(isliteral, e.args)
end

"""
    check_tilde_rhs(x)

Check if the right-hand side `x` of a `~` is a `Distribution` or an array of
`Distributions`, then return `x`.
"""
function check_tilde_rhs(@nospecialize(x))
    return throw(
        ArgumentError(
            "the right-hand side of a `~` must be a `Distribution`, an array of `Distribution`s, or a submodel",
        ),
    )
end
check_tilde_rhs(x::Distribution) = x
check_tilde_rhs(x::AbstractArray{<:Distribution}) = x
check_tilde_rhs(x::Submodel{M,AutoPrefix}) where {M,AutoPrefix} = x

"""
    check_dot_tilde_rhs(x)

Check if the right-hand side `x` of a `.~` is a `UnivariateDistribution`, then return `x`.
"""
function check_dot_tilde_rhs(@nospecialize(x))
    return throw(
        ArgumentError("the right-hand side of a `.~` must be a `UnivariateDistribution`")
    )
end
function check_dot_tilde_rhs(::AbstractArray{<:Distribution})
    msg = """
        As of v0.35, DynamicPPL does not allow arrays of distributions in `.~`. \
        Please use `product_distribution` instead, or write a loop if necessary. \
        See https://github.com/TuringLang/DynamicPPL.jl/releases/tag/v0.35.0 for more \
        details.\
    """
    return throw(ArgumentError(msg))
end
check_dot_tilde_rhs(x::UnivariateDistribution) = x

#################
# Main Compiler #
#################

"""
    @model(expr[, warn = false])

Macro to specify a probabilistic model.

If `warn` is `true`, a warning is displayed if internal variable names are used in the model
definition.

Samples from a multivariate distribution instance on the right-hand side of `~` must all
have the same dimension. A model may use an earlier random variable to construct instances
with different dimensions.

# Examples

Model definition:

```julia
@model function model(x, y = 42)
    ...
end
```

To generate a `Model`, call `model(xvalue)` or `model(xvalue, yvalue)`.

This custom distribution is invalid because samples from the same instance can have
different dimensions:

```julia
using DynamicPPL, Distributions, LinearAlgebra, Random

struct VariableLengthDistribution <: DiscreteMultivariateDistribution end
Base.length(::VariableLengthDistribution) = 5
function Base.rand(rng::Random.AbstractRNG, ::VariableLengthDistribution)
    return rand(rng, 0:1, rand(rng, 1:10))
end
function Distributions._logpdf(::VariableLengthDistribution, x::AbstractVector)
    return -length(x) * log(2)
end

@model function invalid_variable_length()
    x ~ VariableLengthDistribution()
end
```

Evaluating this model throws `DimensionMismatch` unless the draw happens to match the
declared dimension.

Instead, sample the dimension separately. Each `MvNormal` instance then has a fixed sample
dimension:

```julia
@model function variable_dimension()
    n ~ DiscreteUniform(1, 10)
    x ~ MvNormal(zeros(n), I)
end
```
"""
macro model(expr, warn=false)
    # include `LineNumberNode` with information about the call site in the
    # generated function for easier debugging and interpretation of error messages
    return esc(model(__module__, __source__, expr, warn))
end

# Binding-time namespace checks must not evaluate indices from the model body.
_static_lhs_address(name::Symbol) = VarName{name}()
_static_lhs_address(other) = nothing
function _static_lhs_address(expr::Expr)
    optic = if Meta.isexpr(expr, :.) && expr.args[2] isa QuoteNode
        AbstractPPL.Property{expr.args[2].value}()
    elseif Meta.isexpr(expr, :ref) && all(i -> i isa Integer, expr.args[2:end])
        AbstractPPL.Index(Tuple(expr.args[2:end]), NamedTuple())
    else
        return nothing
    end
    parent = _static_lhs_address(expr.args[1])
    parent === nothing && return nothing
    return VarName{AbstractPPL.getsym(parent)}(optic ∘ AbstractPPL.getoptic(parent))
end

function model(mod, linenumbernode, expr, warn)
    modeldef = build_model_definition(expr)

    # Generate main body
    lhs_names = Symbol[]
    lhs_addresses = Tuple{VarName,Bool}[]
    submodel_arguments = Tuple{Union{VarName,Nothing},Union{Symbol,Nothing}}[]
    unprefixed_roots = Symbol[]
    arguments = map(
        arg -> first(MacroTools.splitarg(arg)), vcat(modeldef[:args], modeldef[:kwargs])
    )
    modeldef[:body] = generate_mainbody(
        mod,
        modeldef[:body],
        warn,
        true;
        lhs_names,
        arguments,
        lhs_addresses,
        submodel_arguments,
        unprefixed_roots,
    )

    return build_output(
        modeldef,
        linenumbernode,
        lhs_names,
        Tuple(lhs_addresses),
        Tuple(submodel_arguments),
        Tuple(unique(unprefixed_roots)),
    )
end

"""
    build_model_definition(input_expr)

Builds the `modeldef` dictionary from the model's expression, where
`modeldef` is a dictionary compatible with `MacroTools.combinedef`.
"""
function build_model_definition(input_expr)
    # Break up the model definition and extract its name, arguments, and function body
    modeldef = MacroTools.splitdef(input_expr)

    # Check that the function has a name
    # https://github.com/TuringLang/DynamicPPL.jl/issues/260
    haskey(modeldef, :name) ||
        throw(ArgumentError("anonymous functions without name are not supported"))

    # Print a warning if function body of the model is empty
    warn_empty(modeldef[:body])

    ## Construct model_info dictionary

    # Shortcut if the model does not have any arguments
    if !haskey(modeldef, :args) && !haskey(modeldef, :kwargs)
        return modeldef
    end

    # Ensure that all arguments have a name, i.e., are of the form `name` or `name::T`
    addargnames!(modeldef[:args])

    return modeldef
end

"""
    generate_mainbody(mod, expr, warn, warn_threads)

Generate the body of the main evaluation function from expression `expr` and arguments
`args`.

If `warn` is true, a warning is displayed if internal variables are used in the model
definition.
"""
function generate_mainbody(
    mod,
    expr,
    warn,
    warn_threads;
    lhs_names=Symbol[],
    arguments=Symbol[],
    lhs_addresses=Tuple{VarName,Bool}[],
    submodel_arguments=Tuple{Union{VarName,Nothing},Union{Symbol,Nothing}}[],
    unprefixed_roots=Symbol[],
)
    assigned = Symbol[]
    body = generate_mainbody!(
        mod,
        (;
            internal=Symbol[],
            assigned,
            lhs_names,
            arguments,
            lhs_addresses,
            submodel_arguments,
            unprefixed_roots,
            shadowed=Symbol[],
        ),
        expr,
        warn,
        warn_threads,
    )
    map!(submodel_arguments, submodel_arguments) do (lhs, source)
        (lhs, source in assigned ? nothing : source)
    end
    return body
end

function _record_assigned_argument!(found, lhs)
    if lhs isa Symbol
        lhs in found.arguments && lhs ∉ found.shadowed && push!(found.assigned, lhs)
    elseif lhs isa Expr
        if lhs.head in (:tuple, :parameters)
            foreach(arg -> _record_assigned_argument!(found, arg), lhs.args)
        elseif lhs.head in (:(::), :(...))
            _record_assigned_argument!(found, first(lhs.args))
        end
    end
    return nothing
end

generate_mainbody!(mod, found, x, warn, warn_threads) = x
function generate_mainbody!(mod, found, sym::Symbol, warn, warn_threads)
    if warn && sym in INTERNALNAMES && sym ∉ found.internal
        @warn "you are using the internal variable `$sym`"
        push!(found.internal, sym)
    end

    return sym
end
function generate_mainbody!(mod, found, expr::Expr, warn, warn_threads)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Flag to determine whether we've issued a warning for threadsafe macros Note that this
    # detection is not fully correct. We can only detect the presence of a macro that has
    # the symbol `Threads.@threads`, however, we can't detect if that *is actually*
    # Threads.@threads from Base.Threads.

    # Do we don't want escaped expressions because we unfortunately
    # escape the entire body afterwards.
    Meta.isexpr(expr, :escape) &&
        return generate_mainbody!(mod, found, expr.args[1], warn, warn_threads)

    if expr.head === :-> || MacroTools.isdef(expr)
        definition = MacroTools.splitdef(expr)
        names = [
            first(MacroTools.splitarg(arg)) for
            arg in vcat(get(definition, :args, []), get(definition, :kwargs, []))
        ]
        found = merge(
            found, (; shadowed=union(found.shadowed, intersect(names, found.arguments)))
        )
    end

    if expr.head === :(=)
        _record_assigned_argument!(found, first(expr.args))
    end

    # If it's a macro, we expand it
    if Meta.isexpr(expr, :macrocall)
        if (
            expr.args[1] == Symbol("@threads") ||
            expr.args[1] == Expr(:., :Threads, QuoteNode(Symbol("@threads"))) &&
            warn_threads
        )
            warn_threads = false
            @warn (
                "It looks like you are using `Threads.@threads` in your model definition." *
                "\n\nNote that since version 0.39 of DynamicPPL, threadsafe evaluation of models is disabled by default." *
                " If you need it, you will need to explicitly enable it by creating the model, and then running `model = setthreadsafe(model, true)`." *
                "\n\nThreadsafe model evaluation is only needed when parallelising tilde-statements (not arbitrary Julia code), and avoiding it can often lead to significant performance improvements." *
                "\n\nPlease see https://turinglang.org/docs/usage/threadsafe-evaluation/ for more details of when threadsafe evaluation is actually required."
            )
        end
        return generate_mainbody!(
            mod, found, macroexpand(mod, expr; recursive=true), warn, warn_threads
        )
    end

    # Modify dotted tilde operators.
    args_dottilde = getargs_dottilde(expr)
    if args_dottilde !== nothing
        L, R = args_dottilde
        return generate_mainbody!(
            mod, found, Base.remove_linenums!(generate_dot_tilde(L, R)), warn, warn_threads
        )
    end

    # Modify tilde operators.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        L = generate_mainbody!(mod, found, L, warn, warn_threads)
        if !isliteral(L)
            root = get_top_level_symbol(L)
            f = Meta.isexpr(R, :call) ? R.args[1] : nothing
            call =
                f in (:to_submodel, GlobalRef(DynamicPPL, :to_submodel)) ||
                Meta.isexpr(f, :.) && f.args[end] == QuoteNode(:to_submodel)
            # A non-literal `auto_prefix` can select either form when the model runs.
            auto_prefix = call && length(R.args) > 2 ? R.args[3] : true
            is_submodel = call && auto_prefix !== false
            # Dynamic indices may overlap any address under their root.
            address = something(_static_lhs_address(L), VarName{root}())
            push!(found.lhs_addresses, (address, is_submodel))
            if is_submodel
                child = R.args[2]
                named =
                    child isa Symbol && child in found.arguments && child ∉ found.shadowed
                source = named && _static_lhs_address(L) !== nothing ? child : nothing
                push!(found.submodel_arguments, (address, source))
            end
            call && auto_prefix !== true && push!(found.unprefixed_roots, root)
            root in found.shadowed && throw(
                ArgumentError(
                    "LHS root `$root` shadows model argument `$root`; rename the inner function parameter.",
                ),
            )
            push!(found.lhs_names, root)
        end
        return Base.remove_linenums!(
            generate_tilde(
                L,
                generate_mainbody!(mod, found, R, warn, warn_threads);
                is_argument=!isliteral(L) && get_top_level_symbol(L) in found.arguments,
            ),
        )
    end

    # Modify the assignment operators.
    args_assign = getargs_coloneq(expr)
    if args_assign !== nothing
        L, R = args_assign
        _record_assigned_argument!(found, L)
        return Base.remove_linenums!(
            generate_assign(
                generate_mainbody!(mod, found, L, warn, warn_threads),
                generate_mainbody!(mod, found, R, warn, warn_threads),
            ),
        )
    end

    return Expr(
        expr.head,
        map(x -> generate_mainbody!(mod, found, x, warn, warn_threads), expr.args)...,
    )
end

function generate_assign(left, right)
    # A statement `x := y` reduces to `x = y`, but if __varinfo__ has an accumulator for
    # RawValues then in addition we push!! the pair of `x` and `y` to the accumulator.
    @gensym acc right_val vn
    template = if left isa Symbol  # i.e. identity optic
        :($(NoTemplate)())
    else
        get_top_level_symbol(left)
    end
    return quote
        $right_val = $right
        if $(DynamicPPL.is_extracting_colon_eq_values)(__varinfo__)
            $vn = $(make_varname_expression(left))
            __varinfo__ = $(DynamicPPL.store_coloneq_value!!)(
                __model__, $vn, $right_val, $template, __varinfo__
            )
        end
        $left = $right_val
    end
end

function generate_tilde_literal(left, right)
    # If the LHS is a literal, it is always an observation
    @gensym value
    return quote
        $value, __varinfo__ = $(DynamicPPL.tilde_observe!!)(
            $(DynamicPPL._model_prefix)(__model__),
            $(DynamicPPL._model_prefix_template)(__model__),
            $(DynamicPPL.check_tilde_rhs)($right),
            $left,
            nothing,
            $(NoTemplate()),
            __varinfo__,
        )
        $value
    end
end

function set_tilde_value!!(::AbstractVarInfo, object, value, vn::VarName; create=false)
    return _set_lhs(object, value, AbstractPPL.getoptic(vn), vn; create)
end
_set_lhs(object, value, optic, vn; create) = _set_optic!!(object, value, optic, vn; create)
# A keyword splat must stay `Base.Pairs`, as plain Julia presents it to the body.
function _set_lhs(object::Base.Pairs, value, optic, vn; create)
    return pairs(_set_optic!!(values(object), value, optic, vn; create))
end

function _check_argument_write(object, optic, vn)
    optic isa AbstractPPL.Iden && return nothing
    _element_optic(object) !== nothing || _throw_unsupported_storage(object, vn)
    optic.child isa AbstractPPL.Iden && return nothing
    return _check_argument_write(AbstractPPL.ohead(optic)(object), optic.child, vn)
end
assign_or_set!!(lhs::Symbol, rhs, vn, set=set_tilde_value!!) = drop_escape(:($lhs = $rhs))
function assign_or_set!!(lhs::Expr, rhs, vn, set=set_tilde_value!!)
    left_top_sym = get_top_level_symbol(lhs)
    return drop_escape(:($left_top_sym = $(set)(__varinfo__, $left_top_sym, $rhs, $vn)))
end

_set_optic!!(object, value, ::AbstractPPL.Iden, vn; create=false) = value
function _set_optic!!(
    object::NamedTuple, value, ::AbstractPPL.Property{S,AbstractPPL.Iden}, vn; create=false
) where {S}
    create || getproperty(object, S)  # throws for an absent field, as plain Julia does
    return merge(object, NamedTuple{(S,)}((value,)))
end
function _set_optic!!(
    object, value, optic::AbstractPPL.Property{S}, vn; create=false
) where {S}
    child = _set_optic!!(getproperty(object, S), value, optic.child, vn; create)
    getproperty(object, S) === child && return object
    if ismutabletype(typeof(object)) &&
        hasfield(typeof(object), S) &&
        isconst(typeof(object), S)
        throw(
            ArgumentError(
                "Cannot replace const property `$S` of argument type $(typeof(object)); bind the whole value instead.",
            ),
        )
    end
    return BangBang.setproperty!!(object, S, child)
end
function _set_optic!!(object, value, optic::AbstractPPL.Index, vn; create=false)
    object isa Tuple && VarNamedTuples._tuple_index(optic, object, vn)
    (; ix, kw) = AbstractPPL.concretize_top_level(optic, object)
    if !(optic.child isa AbstractPPL.Iden)
        value = _set_optic!!(getindex(object, ix...; kw...), value, optic.child, vn; create)
    end
    if !isempty(kw)
        setindex!(object, value, ix...; kw...)
        return object
    end
    if object isa AbstractArray
        # Needed while BangBang's `_index_dimension(())` throws (it sums an empty tuple).
        # `isa`, not `isempty`: Mooncake loses inference on `isempty(ix)` here.
        ix = ix isa Tuple{} ? (CartesianIndex(),) : ix
        if _is_read_only(object)
            # Observations write back what they read; thread-safe mode needs `===` storage.
            isequal(object[ix...], value) && return object
            return Accessors.set(object, Accessors.IndexLens(ix), value)
        end
        # BangBang applies slice checks to array-valued scalar writes whose type differs
        # from the eltype; remove this path when its applicability check handles them.
        if ix isa Tuple{Vararg{Union{Integer,CartesianIndex}}} &&
            value isa eltype(object) &&
            BangBang.implements(setindex!, typeof(object))
            object[ix...] = value
            return object
        end
    end
    # Widening converts retained elements unchecked, which can round them.
    return BangBang.setindex!!(object, value, ix...)
end

# Must copy these: `BangBang.implements(setindex!, T)` holds for them, but `setindex!` throws.
_is_read_only(x) = false
_is_read_only(x::AbstractArray) = parent(x) !== x && _is_read_only(parent(x))
# The `AbstractFill` case can move to a FillArrays package extension.
_is_read_only(::Union{AbstractRange,FillArrays.AbstractFill}) = true

# A whole fixed NamedTuple argument may omit latent fields; their tildes add them.
function _can_create_argument_field(model, vn)
    root = _binding_root(model, vn)
    node = root === nothing ? nothing : root.value
    optic = AbstractPPL.getoptic(vn)
    while !(node isa BoundValue) && !(optic isa AbstractPPL.Iden)
        node = _binding_node_at(node, AbstractPPL.ohead(optic))
        optic = optic.child
    end
    return node isa BoundValue{FixRole}
end
function _set_latent_argument(model, vi, object, value, vn)
    return set_tilde_value!!(
        vi, object, value, vn; create=_can_create_argument_field(model, vn)
    )
end

# A keyword splat is observed as the NamedTuple its bindings store, not as `Base.Pairs`.
_observed_argument(value) = value
_observed_argument(value::Base.Pairs) = values(value)
# Skip identical (`===`) immutable values so read-only storage works; equality could
# hide latent aliases. Always write mutable values to avoid allocating comparison slices.
function _set_fixed_lhs(vi, object, value, vn)
    ismutable(value) && return set_tilde_value!!(vi, object, value, vn)
    return if AbstractPPL.getoptic(vn)(object) === value
        object
    else
        set_tilde_value!!(vi, object, value, vn)
    end
end

"""
    generate_tilde(left, right; is_argument=false)

Generate code for `left ~ right` that selects the latent, observed or fixed path at run time.
Set `is_argument=true` for a model argument; an observed argument then uses its current
local value, including changes made by the body.
"""
function generate_tilde(left, right; is_argument=false)
    isliteral(left) && return generate_tilde_literal(left, right)
    top = get_top_level_symbol(left)
    template = left isa Symbol ? :($(NoTemplate)()) : top

    @gensym vn role value dist supplied_val
    lookup_role = if is_argument
        :($(DynamicPPL._argument_role)(__model__, $vn, $(VarName{top}()), $(top)))
    else
        :($(DynamicPPL._local_role)(
            $(DynamicPPL._binding_at)(__model__, $(VarName{top}())),
            $vn,
            if $(Expr(:isdefined, top))
                $(top)
            else
                $(NoTemplate)()
            end,
            __model__,
        ))
    end

    fixed_data = :($(DynamicPPL._bound_data_at)(__model__, $vn, $(top)))
    if !is_argument
        fixed_data = quote
            if $(Expr(:isdefined, top))
                $fixed_data
            else
                $(DynamicPPL._bound_data_at)(__model__, $vn)
            end
        end
    end

    return quote
        $dist = $right
        $vn = $(make_varname_expression(left))
        $(DynamicPPL._check_address_kind)(
            $(DynamicPPL.BoundValue){$(DynamicPPL.ConditionRole)}($template),
            $(AbstractPPL.getoptic)($vn),
            $(AbstractPPL.Property{top}()),
        )
        $role = $dist isa $(DynamicPPL.Submodel) ? nothing : $lookup_role
        if $role === $(DynamicPPL.FixRole)
            $value = $(DynamicPPL._check_no_placeholder)(
                $fixed_data,
                $(DynamicPPL.maybe_prefix)($vn, $(DynamicPPL._model_prefix)(__model__)),
            )
            $(assign_or_set!!(
                left, value, vn, is_argument ? _set_fixed_lhs : set_tilde_value!!
            ))
        elseif $role === nothing
            if !($dist isa $(DynamicPPL.Submodel))
                $(generate_input_provenance_check(left, vn))
            end
            $value, __varinfo__ = $(DynamicPPL.tilde_assume!!)(
                __model__,
                __context__,
                $(DynamicPPL.check_tilde_rhs)($dist),
                $vn,
                $template,
                __varinfo__,
            )
            $(
                if is_argument && left isa Expr
                    quote
                        $top isa $(Union{Missing,Nothing}) ||
                            $(DynamicPPL._check_argument_write)(
                                $top, $(AbstractPPL.getoptic)($vn), $vn
                            )
                        $(drop_escape(
                            :(
                                $top = $(DynamicPPL._set_latent_argument)(
                                    __model__, __varinfo__, $top, $value, $vn
                                )
                            ),
                        ))
                    end
                else
                    assign_or_set!!(left, value, vn)
                end
            )
            $value
        else
            $supplied_val = $(
                if is_argument
                    :($(DynamicPPL._observed_argument)($(Base).@views($left)))
                else
                    quote
                        if $(Expr(:isdefined, top))
                            $(DynamicPPL._local_bound_data)(__model__, $vn, $(top))
                        else
                            $(DynamicPPL._bound_data_at)(__model__, $vn)
                        end
                    end
                end
            )

            $value, __varinfo__ = $(DynamicPPL.tilde_observe!!)(
                $(DynamicPPL._model_prefix)(__model__),
                $(DynamicPPL._model_prefix_template)(__model__),
                $(DynamicPPL.check_tilde_rhs)($dist),
                $supplied_val,
                $vn,
                $template,
                __varinfo__,
            )
            # A whole argument takes the observed value, so a keyword splat's `Base.Pairs`
            # becomes its NamedTuple, as after a latent or fixed tilde.
            $(is_argument && left isa Expr ? nothing : assign_or_set!!(left, value, vn))
            $value
        end
    end
end

const INPUT_PROVENANCE_ACCNAME = :InputProvenance

# The input-provenance extension implements this hook.
check_input_provenance!!(vi::AbstractVarInfo, value, vn::VarName) = vi

# An indexed or property location can be assignable without having a readable value. The
# read is guarded in this function rather than in the model body, because Libtask cannot
# tape a `try` block and a closure over the left-hand side would box it for the whole model.
@noinline function read_input_provenance(f::F, args...) where {F}
    return try
        f(args...)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
end

generate_input_provenance_check(::Any, ::Any) = nothing
function generate_input_provenance_check(left::Union{Expr,Symbol}, vn)
    @gensym value
    top_symbol = get_top_level_symbol(left)
    # The accumulator guard prevents an extra LHS read during ordinary evaluation.
    return quote
        if $(DynamicPPL.hasacc)(__varinfo__, $(Val(INPUT_PROVENANCE_ACCNAME))) &&
            $(Expr(:isdefined, top_symbol))
            $value = $(DynamicPPL.read_input_provenance)(
                $(AbstractPPL.getoptic)($vn), $top_symbol
            )
            __varinfo__ = $(DynamicPPL.check_input_provenance!!)(
                __varinfo__,
                $value,
                $(DynamicPPL.maybe_prefix)($vn, $(DynamicPPL._model_prefix)(__model__)),
            )
        end
    end
end

"""
    generate_dot_tilde(left, right)

Generate the expression that replaces `left .~ right` in the model body.
"""
function generate_dot_tilde(left, right)
    @gensym dist left_axes idx
    return quote
        $dist = $(DynamicPPL.check_dot_tilde_rhs)($right)
        $left_axes = axes($left)
        for $idx in Iterators.product($left_axes...)
            $left[$idx...] ~ $dist
        end
    end
end

# Note that we cannot use `MacroTools.isdef` because
# of https://github.com/FluxML/MacroTools.jl/issues/154.
"""
    isfuncdef(expr)

Return `true` if `expr` is any form of function definition, and `false` otherwise.
"""
function isfuncdef(e::Expr)
    return if Meta.isexpr(e, :function)
        # Classic `function f(...)`
        true
    elseif Meta.isexpr(e, :->)
        # Anonymous functions/lambdas, e.g. `do` blocks or `->` defs.
        true
    elseif Meta.isexpr(e, :(=)) && Meta.isexpr(e.args[1], :call)
        # Short function defs, e.g. `f(args...) = ...`.
        true
    else
        false
    end
end

"""
    replace_returns(expr)

Return `Expr` with all `return ...` statements replaced with
`return ..., DynamicPPL.return_values(__varinfo__)`.

Note that this method will _not_ replace `return` statements within function
definitions. This is checked using [`isfuncdef`](@ref).
"""
replace_returns(e) = e
function replace_returns(e::Expr)
    isfuncdef(e) && return e

    if Meta.isexpr(e, :return)
        # We capture the original return-value in `retval` and return
        # a `Tuple{typeof(retval),typeof(__varinfo__)}`.
        # If we don't capture the return-value separately, cases such as
        # `return x = 1` will result in `(x = 1, __varinfo__)` which will
        # mistakenly attempt to construct a `NamedTuple` (which fails on Julia 1.3
        # and is not our intent).
        @gensym retval
        return quote
            $retval = $(map(replace_returns, e.args)...)
            return $retval, __varinfo__
        end
    end

    return Expr(e.head, map(replace_returns, e.args)...)
end

# If it's just a symbol, e.g. `f(x) = 1`, then we make it `f(x) = return 1`.
add_return_to_last_statment(body) = Expr(:return, body)
function add_return_to_last_statment(body::Expr)
    # If the last statement is a return-statement, we don't do anything.
    # Otherwise we replace the last statement with a `return` statement.
    Meta.isexpr(body.args[end], :return) && return body
    # We need to copy the arguments since we are modifying them.
    new_args = copy(body.args)
    new_args[end] = Expr(:return, body.args[end])
    return Expr(body.head, new_args...)
end

hasmissing(::Type) = false
hasmissing(::Type{>:Missing}) = true
hasmissing(::Type{<:AbstractArray{TA}}) where {TA} = hasmissing(TA)
hasmissing(::Type{Union{}}) = false # issue #368

"""
    TypeWrap{T}

A wrapper type used internally to make expressions such as `::Type{TV}` in the model arguments
not ending up as a `DataType`.
"""
struct TypeWrap{T} end

function arg_type_is_type(e)
    return Meta.isexpr(e, :curly) && length(e.args) > 1 && e.args[1] === :Type
end

function splitarg_to_expr((arg_name, arg_type, is_splat, default))
    return is_splat ? :($arg_name...) : arg_name
end

"""
    transform_args(args)

Return transformed `args` used in both the model constructor and evaluator.

Specifically, this replaces expressions of the form `::Type{TV}=Vector{Float64}`
with `::TypeWrap{TV}=TypeWrap{Vector{Float64}}()` to avoid introducing `DataType`.
"""
function transform_args(args)
    splitargs = map(args) do arg
        arg_name, arg_type, is_splat, default = MacroTools.splitarg(arg)
        return if arg_type_is_type(arg_type)
            arg_name, :($TypeWrap{$(arg_type.args[2])}), is_splat, :($TypeWrap{$default}())
        else
            arg_name, arg_type, is_splat, default
        end
    end
    return map(Base.splat(MacroTools.combinearg), splitargs)
end

function namedtuple_from_splitargs(splitargs)
    names = map(splitargs) do (arg_name, arg_type, is_splat, default)
        is_splat ? Symbol("#splat#$(arg_name)") : arg_name
    end
    names_expr = Expr(:tuple, map(QuoteNode, names)...)
    vals = Expr(:tuple, map(first, splitargs)...)
    return :(NamedTuple{$names_expr}($vals))
end

"""
    build_output(modeldef, linenumbernode)

Builds the output expression.
"""
function build_output(
    modeldef, linenumbernode, lhs_names, lhs_addresses, submodel_arguments, unprefixed_roots
)
    args = transform_args(modeldef[:args])
    kwargs = transform_args(modeldef[:kwargs])

    # Need to update `args` and `kwargs` since we might have added `TypeWrap` to the types.
    modeldef[:args] = args
    modeldef[:kwargs] = kwargs

    ## Build the anonymous evaluator from the user-provided model definition.
    evaluatordef = copy(modeldef)

    # Add the internal arguments to the user-specified arguments (positional + keywords).
    evaluatordef[:args] = vcat(
        [:(__model__::$(DynamicPPL.Model)), :(__varinfo__::$(DynamicPPL.AbstractVarInfo))],
        args,
    )

    # Replace the user-provided function body with the version created by DynamicPPL.
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site.
    # NOTE: We need to replace statements of the form `return ...` with
    # `return (..., __varinfo__)` to ensure that the second
    # element in the returned value is always the most up-to-date `__varinfo__`.
    # See the docstrings of `replace_returns` for more info.
    evaluatordef[:body] = MacroTools.@q begin
        $(linenumbernode)
        __context__ = first($(DynamicPPL.extract_prefixes)(__model__.context))
        $(replace_returns(add_return_to_last_statment(modeldef[:body])))
    end

    ## Build the model function.

    # Obtain or generate the name of the model to support functors:
    # https://github.com/TuringLang/DynamicPPL.jl/issues/367
    if MacroTools.@capture(modeldef[:name], ::T_)
        name = gensym(:f)
        modeldef[:name] = Expr(:(::), name, T)
    elseif MacroTools.@capture(modeldef[:name], (name_::_ | name_))
    else
        throw(ArgumentError("unsupported format of model function"))
    end

    args_split = map(MacroTools.splitarg, args)
    kwargs_split = map(MacroTools.splitarg, kwargs)
    args_nt = namedtuple_from_splitargs(args_split)
    kwargs_nt = namedtuple_from_splitargs(kwargs_split)
    normalize_kwargs = [
        :($name = $(NamedTuple)($name)) for
        (name, _, is_splat, _) in kwargs_split if is_splat
    ]
    args_on_lhs = unique([
        name for (name, _, _, _) in vcat(args_split, kwargs_split) if name in lhs_names
    ])
    type_names = Tuple(
        is_splat ? Symbol("#splat#", n) : n for
        (n, _, is_splat, _) in vcat(args_split, kwargs_split)
    )
    types = Any[is_splat ? :(Tuple{Vararg{$t}}) : t for (_, t, is_splat, _) in args_split]
    append!(types, [is_splat ? NamedTuple : t for (_, t, is_splat, _) in kwargs_split])
    if !isempty(modeldef[:whereparams])
        types = [Expr(:where, t, modeldef[:whereparams]...) for t in types]
    end
    argument_types = :(NamedTuple{$(QuoteNode(type_names)),Tuple{$(types...)}})
    @gensym replaced prepared
    prepare_args = map(args_on_lhs) do name
        return quote
            $prepared = $(prepare_model_argument)(__model__, $(VarName{name}()), $name)
            $replaced |= $prepared !== $name
            $name = $prepared
        end
    end
    debugdef = nothing
    bodydef = if isempty(args_on_lhs)
        nothing
    else
        definition = copy(evaluatordef)
        definition[:name] = gensym(:model_body)
        # Pass prepared keywords positionally so applicability checks their types too.
        definition[:kwargs] = []
        definition[:args] = vcat(
            definition[:args][1:2],
            [MacroTools.combinearg(n, t, false, nothing) for (n, t, _, _) in kwargs_split],
            args,
        )
        callargs = Any[
            :__model__,
            :__varinfo__,
            [
                is_splat ? :($(Base.pairs)($(NamedTuple)($n))) : n for
                (n, _, is_splat, _) in kwargs_split
            ]...,
            map(splitarg_to_expr, args_split)...,
        ]
        if Meta.isexpr(evaluatordef[:name], :(::))
            definition[:args] = vcat([evaluatordef[:name]], definition[:args])
            pushfirst!(callargs, :(__model__.f))
        end
        descriptions = [
            :(
                if $(
                    if is_splat
                        :($(Base.all)($(Base.Fix2)($(Core.isa), $t), $(Base.values)($n)))
                    else
                        :($n isa $t)
                    end
                )
                    nothing
                else
                    $(Base.string)($("`$n` declared as $t, supplied "), $(Core.typeof)($n))
                end
            ) for
            (n, t, is_splat, _) in vcat(args_split, kwargs_split) if n in args_on_lhs
        ]
        # Redispatch prepared arguments to resolve body type parameters. Static
        # applicability lets Mooncake eliminate the identity check during child evaluation.
        evaluatordef[:body] = MacroTools.@q begin
            $replaced = false
            $(prepare_args...)
            if !$(Base.applicable)($(definition[:name]), $(callargs...)) && $replaced
                $(Core.throw)(
                    $(ArgumentError)(
                        $(Base.string)(
                            "Bound value does not match the declared argument type in model `",
                            $(Base.nameof)(__model__),
                            "`: ",
                            $(Base.join)(
                                $(Base.filter)($(!isnothing), ($(descriptions...),)),
                                "; ",
                            ),
                        ),
                    ),
                )
            end
            return $(Base).@inline $(definition[:name])($(callargs...))
        end
        # The debug entry point is a method of the model function itself: a method of a
        # DynamicPPL function cannot be defined when `@model` is used in local scope.
        debug_definition = copy(evaluatordef)
        debug_definition[:args] = vcat(
            [:(::$(Core.Typeof)($(_model_evaluator)))], evaluatordef[:args]
        )
        debug_definition[:body] = Expr(
            :block,
            evaluatordef[:body].args[1:(end - 1)]...,
            :(return ($(definition[:name]), ($(callargs...),), $(NamedTuple)())),
        )
        debugdef = MacroTools.combinedef(debug_definition)
        MacroTools.combinedef(definition)
    end

    # Update the function body of the user-specified model.
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site
    modeldef[:body] = MacroTools.@q begin
        $(linenumbernode)
        $(normalize_kwargs...)
        return $(DynamicPPL.Model){false}(
            $name,
            $args_nt,
            $kwargs_nt,
            $(DynamicPPL.DefaultContext)();
            args_on_lhs=$(ModelBindingMetadata){
                $(QuoteNode(Tuple(args_on_lhs))),
                $(QuoteNode(Tuple(unique(lhs_names)))),
                $argument_types,
                $(QuoteNode(Tuple(unique(lhs_addresses)))),
                $(QuoteNode(Tuple(unique(submodel_arguments)))),
                $(QuoteNode(unprefixed_roots)),
            }(),
        )
    end

    return MacroTools.@q begin
        $bodydef
        $(MacroTools.combinedef(evaluatordef))
        $debugdef
        $(Base).@__doc__ $(MacroTools.combinedef(modeldef))
    end
end

function _model_evaluator(f, args, kwargs)
    isempty(_lhs_arguments(first(args))) && return (f, args, kwargs)
    types = Base.typesof(_model_evaluator, args...)
    if hasmethod(f, types)
        signature = Base.unwrap_unionall(which(f, types).sig)
        # Require a method that names the debug marker; a generic `f(args...)` also
        # matches but is not the debug entry point.
        if signature.parameters[2] === typeof(_model_evaluator)
            return f(_model_evaluator, args...; kwargs...)
        end
    end
    return f, args, kwargs
end

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return nothing
end

"""
    convert_model_argument(param_eltype, model_argument)
    convert_model_argument(varinfo, context, model_argument)

Promote type arguments to the parameter element type. Deep-copy value arguments whose
type admits `missing`, so body writes do not reach the caller; leave other values unchanged.
The three-argument form queries the parameter element type only for type arguments.
"""
function convert_model_argument(param_eltype, model_argument::T) where {T}
    return hasmissing(T) ? deepcopy(model_argument) : model_argument
end
# These methods handle arguments that are types rather than values.
function convert_model_argument(param_eltype, t::Type{<:Union{Real,AbstractArray}})
    return promote_model_type_argument(param_eltype, t)
end
function convert_model_argument(param_eltype, ::TypeWrap{T}) where {T}
    return TypeWrap{promote_model_type_argument(param_eltype, T)}()
end
# An unknown parameter element type must not erase concrete type arguments.
convert_model_argument(::Type{Any}, t::Type{<:Union{Real,AbstractArray}}) = t
convert_model_argument(::Type{Any}, t::TypeWrap{T}) where {T} = t

convert_model_argument(varinfo, context, argument) = convert_model_argument(Any, argument)
function convert_model_argument(varinfo, context, argument::Union{Type,TypeWrap})
    return convert_model_argument(get_param_eltype(varinfo, context), argument)
end

"""
    promote_model_type_argument(param_eltype, ::Type{T}) where {T}
    promote_model_type_argument(param_eltype, ::TypeWrap{T}) where {T}

For arguments to a model that are types rather than values, promote the type `T` to
match the element type of the parameters being used to evaluate the model.
"""
promote_model_type_argument(_, ::Type{T}) where {T} = T
function promote_model_type_argument(param_eltype, ::Type{T}) where {T<:Real}
    # TODO(penelopeysm): This actually might still be over-aggressive. For example, if
    # `param_eltype` is `Float32` and `T` is `Vector{Int}`, then (after going through the
    # Array method) we will promote to `Vector{Float64}`, which seems unnecessary. However,
    # there's no way to actually check if `T` is the type of something that will later be
    # assigned to, so this is 'safe'.
    return Base.promote_type(param_eltype, T)
end
# NOTE(penelopeysm): This doesn't work with other types of AbstractArray. To get around
# that, one could in principle use ArrayInterface.promote_eltype. However, it doesn't seem
# like there is (1) demand for that; and (2) sufficiently strong adoption of ArrayInterface
# to make that worth adding as a dependency.
function promote_model_type_argument(param_eltype, ::Type{Array{T,N}}) where {T,N}
    promoted_eltype = promote_model_type_argument(param_eltype, T)
    return Array{promoted_eltype,N}
end
