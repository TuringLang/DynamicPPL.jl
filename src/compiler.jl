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

function model(mod, linenumbernode, expr, warn)
    modeldef = build_model_definition(expr)

    # Generate main body
    sites = Symbol[]
    arguments = map(
        arg -> first(MacroTools.splitarg(arg)), vcat(modeldef[:args], modeldef[:kwargs])
    )
    modeldef[:body] = generate_mainbody(mod, modeldef[:body], warn, true; sites, arguments)

    return build_output(modeldef, linenumbernode, sites)
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
generate_mainbody(mod, expr, warn, warn_threads; sites=Symbol[], arguments=Symbol[]) =
    generate_mainbody!(
        mod, (; internal=Symbol[], sites, arguments), expr, warn, warn_threads
    )

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
            push!(found.sites, get_top_level_symbol(L))
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
        $value, __varinfo__ = $(DynamicPPL._tilde_observe!!)(
            __model__.prefix,
            __model__.prefix_template,
            $(DynamicPPL.check_tilde_rhs)($right),
            $left,
            nothing,
            $(NoTemplate()),
            __varinfo__,
        )
        $value
    end
end

assign_or_set!!(lhs::Symbol, rhs, vn) = drop_escape(:($lhs = $rhs))
function assign_or_set!!(lhs::Expr, rhs, vn)
    left_top_sym = get_top_level_symbol(lhs)
    return drop_escape(
        :(
            $left_top_sym = $(Accessors.set)(
                $left_top_sym,
                $(AbstractPPL.with_mutation)($(AbstractPPL.getoptic)($vn)),
                $rhs,
            )
        ),
    )
end

"""
    generate_tilde(left, right; is_argument=false)

Generate latent, observed, or fixed evaluation for a tilde expression.
Observed arguments use their prepared local value, including body computations.
"""
function generate_tilde(left, right; is_argument=false)
    isliteral(left) && return generate_tilde_literal(left, right)
    template = if left isa Symbol  # i.e. identity optic
        :($(NoTemplate)())
    else
        get_top_level_symbol(left)
    end

    @gensym vn role value dist supplied_val
    lookup_role = if is_argument
        :($(DynamicPPL._get_argument_role)(
            __model__, $vn, $(VarName{get_top_level_symbol(left)}())
        ))
    else
        :($(DynamicPPL._get_model_role)(__model__, $vn))
    end

    return quote
        $dist = $right
        $vn = $(make_varname_expression(left))
        $role = if $dist isa $(DynamicPPL.Submodel)
            nothing
        else
            $lookup_role
        end
        if $role isa $(DynamicPPL.Fix)
            $(assign_or_set!!(left, :($(DynamicPPL._get_model_data)(__model__, $vn)), vn))
        elseif $role === nothing
            $(generate_tilde_assume(left, dist, vn))
        else
            $supplied_val = $(
                if is_argument
                    left
                else
                    :($(DynamicPPL._get_model_data)(__model__, $vn))
                end
            )

            $value, __varinfo__ = $(DynamicPPL._tilde_observe!!)(
                __model__.prefix,
                __model__.prefix_template,
                $(DynamicPPL.check_tilde_rhs)($dist),
                $supplied_val,
                $vn,
                $template,
                __varinfo__,
            )
            $(assign_or_set!!(left, value, vn))
            $value
        end
    end
end

function generate_tilde_assume(left, right, vn)
    # HACK: Because the Setfield.jl macro does not support assignment
    # with multiple arguments on the LHS, we need to capture the return-values
    # and then update the LHS variables one by one.
    @gensym value
    expr = if left isa Expr # as opposed to Symbol
        left_top_sym = get_top_level_symbol(left)
        :(
            $left_top_sym = $(Accessors.set)(
                $left_top_sym,
                $(AbstractPPL.with_mutation)($(AbstractPPL.getoptic)($vn)),
                $value,
            )
        )
    else
        :($left = $value)
    end
    template = if left isa Symbol  # i.e. identity optic
        :($(NoTemplate)())
    else
        left_top_sym
    end
    return quote
        $value, __varinfo__ = $(DynamicPPL.tilde_assume!!)(
            __model__,
            __context__,
            $(DynamicPPL.check_tilde_rhs)($right),
            $vn,
            $template,
            __varinfo__,
        )
        $expr
        $value
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
function build_output(modeldef, linenumbernode, sites)
    args = transform_args(modeldef[:args])
    kwargs = transform_args(modeldef[:kwargs])

    # Need to update `args` and `kwargs` since we might have added `TypeWrap` to the types.
    modeldef[:args] = args
    modeldef[:kwargs] = kwargs

    ## Build the anonymous evaluator from the user-provided model definition.
    evaluatordef = copy(modeldef)

    # Add the internal arguments to the user-specified arguments (positional + keywords).
    evaluatordef[:args] = vcat(
        [
            :(__model__::$(DynamicPPL.Model)),
            :(__context__::$(DynamicPPL.Context)),
            :(__varinfo__::$(DynamicPPL.AbstractVarInfo)),
        ],
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
    kwargs_inclusion = map(splitarg_to_expr, kwargs_split)
    observed_args = unique([
        name for (name, _, _, _) in vcat(args_split, kwargs_split) if name in sites
    ])
    observations = Expr(:tuple, [Expr(:(=), name, name) for name in observed_args]...)
    prepare_args = [
        :($name = $(prepare_model_argument)(__model__, $(VarName{name}()), $name)) for
        name in observed_args
    ]
    bodydef = if isempty(observed_args)
        nothing
    else
        definition = copy(evaluatordef)
        definition[:name] = gensym(:model_body)
        callargs = Any[
            :__model__, :__context__, :__varinfo__, map(splitarg_to_expr, args_split)...
        ]
        if Meta.isexpr(evaluatordef[:name], :(::))
            definition[:args] = vcat([evaluatordef[:name]], definition[:args])
            pushfirst!(callargs, :(__model__.f))
        end
        # Dispatch again after replacement so the body's type parameters match its inputs.
        evaluatordef[:body] = MacroTools.@q begin
            $(prepare_args...)
            return $(definition[:name])($(callargs...); $(kwargs_inclusion...))
        end
        MacroTools.combinedef(definition)
    end

    # Update the function body of the user-specified model.
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site
    modeldef[:body] = MacroTools.@q begin
        $(linenumbernode)
        return $(DynamicPPL.Model){false}(
            $name,
            $args_nt,
            (; $(kwargs_inclusion...)),
            nothing,
            $(_tag_model_values)($(Condition), $(VarNamedTuple)($observations)),
        )
    end

    return MacroTools.@q begin
        $bodydef
        $(MacroTools.combinedef(evaluatordef))
        $(Base).@__doc__ $(MacroTools.combinedef(modeldef))
    end
end

function prepare_model_argument(model::Model, vn::VarName, value)
    vn = _model_value_varname(model.values, vn, model.prefix)
    binding = _model_argument_binding(
        _model_values(model.values), AbstractPPL.varname_to_optic(vn)
    )
    return _model_argument_value(binding, value)
end

# Whole-variable hasvalue checks reject partial containers needed for argument preparation.
_model_argument_binding(values, ::AbstractPPL.Iden) = values
function _model_argument_binding(
    values::VarNamedTuple, optic::AbstractPPL.Property{S}
) where {S}
    return if haskey(values.data, S)
        _model_argument_binding(values.data[S], optic.child)
    else
        nothing
    end
end
function _model_argument_binding(
    values::VarNamedTuples.PartialArray, optic::AbstractPPL.Index
)
    optic = AbstractPPL.concretize_top_level(optic, values.data)
    return if haskey(values, optic.ix...; optic.kw...)
        selected = if VarNamedTuples._is_multiindex(values.data, optic.ix...; optic.kw...)
            VarNamedTuples._subset_partialarray(values, optic.ix...; optic.kw...)
        else
            getindex(values, optic.ix...; optic.kw...)
        end
        _model_argument_binding(selected, optic.child)
    else
        nothing
    end
end
function _model_argument_binding(values::ModelValue, optic::AbstractPPL.AbstractOptic)
    return if VarNamedTuples._haskey_optic(values, optic)
        VarNamedTuples._getindex_optic(values, optic, @varname(_))
    else
        nothing
    end
end
_model_argument_binding(values::ModelValue, ::AbstractPPL.Iden) = values

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return nothing
end

"""
    convert_model_argument(param_eltype, model_argument)

Promote type arguments to the parameter element type; leave value arguments unchanged.
"""
convert_model_argument(param_eltype, model_argument) = model_argument
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
