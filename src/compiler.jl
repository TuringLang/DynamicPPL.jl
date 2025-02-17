const INTERNALNAMES = (:__model__, :__context__, :__varinfo__)

"""
    need_concretize(expr)

Return `true` if `expr` needs to be concretized, i.e., if it contains a colon `:` or
requires a dynamic optic.

# Examples

```jldoctest; setup=:(using Accessors)
julia> DynamicPPL.need_concretize(:(x[1, :]))
true

julia> DynamicPPL.need_concretize(:(x[1, end]))
true

julia> DynamicPPL.need_concretize(:(x[1, 1]))
false
"""
function need_concretize(expr)
    return Accessors.need_dynamic_optic(expr) || begin
        flag = false
        MacroTools.postwalk(expr) do ex
            # Concretise colon by default
            ex == :(:) && (flag = true) && return ex
        end
        flag
    end
end

"""
    isassumption(expr[, vn])

Return an expression that can be evaluated to check if `expr` is an assumption in the
model.

Let `expr` be `:(x[1])`. It is an assumption in the following cases:
    1. `x` is not among the input data to the model,
    2. `x` is among the input data to the model but with a value `missing`, or
    3. `x` is among the input data to the model with a value other than missing,
       but `x[1] === missing`.

When `expr` is not an expression or symbol (i.e., a literal), this expands to `false`.

If `vn` is specified, it will be assumed to refer to a expression which
evaluates to a `VarName`, and this will be used in the subsequent checks.
If `vn` is not specified, `AbstractPPL.varname(expr, need_concretize(expr))` will be
used in its place.
"""
function isassumption(
    expr::Union{Expr,Symbol},
    vn=AbstractPPL.drop_escape(varname(expr, need_concretize(expr))),
)
    return quote
        if $(DynamicPPL.contextual_isassumption)(__context__, $vn)
            # Considered an assumption by `__context__` which means either:
            # 1. We hit the default implementation, e.g. using `DefaultContext`,
            #    which in turn means that we haven't considered if it's one of
            #    the model arguments, hence we need to check this.
            # 2. We are working with a `ConditionContext` _and_ it's NOT in the model arguments,
            #    i.e. we're trying to condition one of the latent variables.
            #    In this case, the below will return `true` since the first branch
            #    will be hit.
            # 3. We are working with a `ConditionContext` _and_ it's in the model arguments,
            #    i.e. we're trying to override the value. This is currently NOT supported.
            #    TODO: Support by adding context to model, and use `model.args`
            #    as the default conditioning. Then we no longer need to check `inargnames`
            #    since it will all be handled by `contextual_isassumption`.
            if !($(DynamicPPL.inargnames)($vn, __model__)) ||
                $(DynamicPPL.inmissings)($vn, __model__)
                true
            else
                $(maybe_view(expr)) === missing
            end
        else
            false
        end
    end
end

# failsafe: a literal is never an assumption
isassumption(expr, vn) = :(false)
isassumption(expr) = :(false)

"""
    contextual_isassumption(context, vn)

Return `true` if `vn` is considered an assumption by `context`.

The default implementation for `AbstractContext` always returns `true`.
"""
contextual_isassumption(::IsLeaf, context, vn) = true
function contextual_isassumption(::IsParent, context, vn)
    return contextual_isassumption(childcontext(context), vn)
end
function contextual_isassumption(context::AbstractContext, vn)
    return contextual_isassumption(NodeTrait(context), context, vn)
end
function contextual_isassumption(context::ConditionContext, vn)
    if hasconditioned(context, vn)
        val = getconditioned(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return true
        else
            return false
        end
    end

    # We might have nested contexts, e.g. `ConditionContext{.., <:PrefixContext{..., <:ConditionContext}}`
    # so we defer to `childcontext` if we haven't concluded that anything yet.
    return contextual_isassumption(childcontext(context), vn)
end
function contextual_isassumption(context::PrefixContext, vn)
    return contextual_isassumption(childcontext(context), prefix(context, vn))
end

isfixed(expr, vn) = false
isfixed(::Union{Symbol,Expr}, vn) = :($(DynamicPPL.contextual_isfixed)(__context__, $vn))

"""
    contextual_isfixed(context, vn)

Return `true` if `vn` is considered fixed by `context`.
"""
contextual_isfixed(::IsLeaf, context, vn) = false
function contextual_isfixed(::IsParent, context, vn)
    return contextual_isfixed(childcontext(context), vn)
end
function contextual_isfixed(context::AbstractContext, vn)
    return contextual_isfixed(NodeTrait(context), context, vn)
end
function contextual_isfixed(context::PrefixContext, vn)
    return contextual_isfixed(childcontext(context), prefix(context, vn))
end
function contextual_isfixed(context::FixedContext, vn)
    if hasfixed(context, vn)
        val = getfixed(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return false
        else
            return true
        end
    end

    # We might have nested contexts, e.g. `FixedContext{.., <:PrefixContext{..., <:FixedContext}}`
    # so we defer to `childcontext` if we haven't concluded that anything yet.
    return contextual_isfixed(childcontext(context), vn)
end

# If we're working with, say, a `Symbol`, then we're not going to `view`.
maybe_view(x) = x
maybe_view(x::Expr) = :(@views($x))

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
check_tilde_rhs(x::Model) = x
check_tilde_rhs(x::ReturnedModelWrapper) = x
function check_tilde_rhs(x::Sampleable{<:Any,AutoPrefix}) where {AutoPrefix}
    model = check_tilde_rhs(x.model)
    return Sampleable{typeof(model),AutoPrefix}(model)
end

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

"""
    unwrap_right_vn(right, vn)

Return the unwrapped distribution on the right-hand side and variable name on the left-hand
side of a `~` expression such as `x ~ Normal()`.

This is used mainly to unwrap `NamedDist` distributions.
"""
unwrap_right_vn(right, vn) = right, vn
unwrap_right_vn(right::NamedDist, vn) = unwrap_right_vn(right.dist, right.name)

"""
    unwrap_right_left_vns(right, left, vns)

Return the unwrapped distributions on the right-hand side and values and variable names on the
left-hand side of a `.~` expression such as `x .~ Normal()`.

This is used mainly to unwrap `NamedDist` distributions and adjust the indices of the
variables.

# Example
```jldoctest; setup=:(using Distributions, LinearAlgebra)
julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(MvNormal(ones(2), I), randn(2, 2), @varname(x)); vns[end]
x[:, 2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(1, 2), @varname(x)); vns[end]
x[1, 2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(1, 2), @varname(x[:])); vns[end]
x[:][1, 2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(3), @varname(x[1])); vns[end]
x[1][3]
```
"""
unwrap_right_left_vns(right, left, vns) = right, left, vns
function unwrap_right_left_vns(right::NamedDist, left, vns)
    return unwrap_right_left_vns(right.dist, left, right.name)
end
function unwrap_right_left_vns(
    right::MultivariateDistribution, left::AbstractMatrix, vn::VarName
)
    # This an expression such as `x .~ MvNormal()` which we interpret as
    #     x[:, i] ~ MvNormal()
    # for `i = size(left, 2)`. Hence the symbol should be `x[:, i]`,
    # and we therefore add the `Colon()` below.
    vns = map(axes(left, 2)) do i
        return AbstractPPL.concretize(Accessors.IndexLens((Colon(), i)) ∘ vn, left)
    end
    return unwrap_right_left_vns(right, left, vns)
end
function unwrap_right_left_vns(
    right::Union{Distribution,AbstractArray{<:Distribution}},
    left::AbstractArray,
    vn::VarName,
)
    vns = map(CartesianIndices(left)) do i
        return Accessors.IndexLens(Tuple(i)) ∘ vn
    end
    return unwrap_right_left_vns(right, left, vns)
end

resolve_varnames(vn::VarName, _) = vn
resolve_varnames(vn::VarName, dist::NamedDist) = dist.name

#################
# Main Compiler #
#################

"""
    @model(expr[, warn = false])

Macro to specify a probabilistic model.

If `warn` is `true`, a warning is displayed if internal variable names are used in the model
definition.

# Examples

Model definition:

```julia
@model function model(x, y = 42)
    ...
end
```

To generate a `Model`, call `model(xvalue)` or `model(xvalue, yvalue)`.
"""
macro model(expr, warn=false)
    # include `LineNumberNode` with information about the call site in the
    # generated function for easier debugging and interpretation of error messages
    return esc(model(__module__, __source__, expr, warn))
end

function model(mod, linenumbernode, expr, warn)
    modeldef = build_model_definition(expr)

    # Generate main body
    modeldef[:body] = generate_mainbody(mod, modeldef[:body], warn)

    return build_output(modeldef, linenumbernode)
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
    generate_mainbody(mod, expr, warn)

Generate the body of the main evaluation function from expression `expr` and arguments
`args`.

If `warn` is true, a warning is displayed if internal variables are used in the model
definition.
"""
generate_mainbody(mod, expr, warn) = generate_mainbody!(mod, Symbol[], expr, warn)

generate_mainbody!(mod, found, x, warn) = x
function generate_mainbody!(mod, found, sym::Symbol, warn)
    if warn && sym in INTERNALNAMES && sym ∉ found
        @warn "you are using the internal variable `$sym`"
        push!(found, sym)
    end

    return sym
end
function generate_mainbody!(mod, found, expr::Expr, warn)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Do we don't want escaped expressions because we unfortunately
    # escape the entire body afterwards.
    Meta.isexpr(expr, :escape) && return generate_mainbody(mod, found, expr.args[1], warn)

    # If it's a macro, we expand it
    if Meta.isexpr(expr, :macrocall)
        return generate_mainbody!(mod, found, macroexpand(mod, expr; recursive=true), warn)
    end

    # Modify dotted tilde operators.
    args_dottilde = getargs_dottilde(expr)
    if args_dottilde !== nothing
        L, R = args_dottilde
        return generate_mainbody!(
            mod, found, Base.remove_linenums!(generate_dot_tilde(L, R)), warn
        )
    end

    # Modify tilde operators.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        # Check for a ~ b --> c
        args_longrightarrow = getargs_longrightarrow(R)
        if args_longrightarrow !== nothing
            M, R = args_longrightarrow
            return Base.remove_linenums!(
                generate_tilde_longrightarrow(
                    generate_mainbody!(mod, found, L, warn),
                    generate_mainbody!(mod, found, M, warn),
                    generate_mainbody!(mod, found, R, warn),
                ),
            )
        else
            return Base.remove_linenums!(
                generate_tilde(
                    generate_mainbody!(mod, found, L, warn),
                    generate_mainbody!(mod, found, R, warn),
                ),
            )
        end
    end

    # Modify the assignment operators.
    args_assign = getargs_coloneq(expr)
    if args_assign !== nothing
        L, R = args_assign
        return Base.remove_linenums!(
            generate_assign(
                generate_mainbody!(mod, found, L, warn),
                generate_mainbody!(mod, found, R, warn),
            ),
        )
    end

    return Expr(expr.head, map(x -> generate_mainbody!(mod, found, x, warn), expr.args)...)
end

function generate_assign(left, right)
    right_expr = :($(TrackedValue)($right))
    tilde_expr = generate_tilde(left, right_expr)
    return quote
        if $(is_extracting_values)(__context__)
            $tilde_expr
        else
            $left = $right
        end
    end
end

function generate_tilde_literal(left, right)
    # If the LHS is a literal, it is always an observation
    @gensym value
    return quote
        $value, __varinfo__ = $(DynamicPPL.tilde_observe!!)(
            __context__, $(DynamicPPL.check_tilde_rhs)($right), $left, __varinfo__
        )
        $value
    end
end

"""
    generate_tilde(left, right)

Generate an `observe` expression for data variables and `assume` expression for parameter
variables.
"""
function generate_tilde(left, right)
    isliteral(left) && return generate_tilde_literal(left, right)

    # Otherwise it is determined by the model or its value,
    # if the LHS represents an observation
    @gensym vn isassumption value dist

    # HACK: Usage of `drop_escape` is unfortunate. It's a consequence of the fact
    # that in DynamicPPL we the entire function body. Instead we should be
    # more selective with our escape. Until that's the case, we remove them all.
    return quote
        $dist = $right
        $vn = $(DynamicPPL.resolve_varnames)(
            $(AbstractPPL.drop_escape(varname(left, need_concretize(left)))), $dist
        )
        $isassumption = $(DynamicPPL.isassumption(left, vn))
        if $(DynamicPPL.isfixed(left, vn))
            $left = $(DynamicPPL.getfixed_nested)(__context__, $vn)
        elseif $isassumption
            $(generate_tilde_assume(left, dist, vn, nothing))
        else
            # If `vn` is not in `argnames`, we need to make sure that the variable is defined.
            if !$(DynamicPPL.inargnames)($vn, __model__)
                $left = $(DynamicPPL.getconditioned_nested)(__context__, $vn)
            end

            $value, __varinfo__ = $(DynamicPPL.tilde_observe!!)(
                __context__,
                $(DynamicPPL.check_tilde_rhs)($dist),
                $(maybe_view(left)),
                $vn,
                __varinfo__,
            )
            $value
        end
    end
end

"""
    generate_tilde_longrightarrow(left, middle, right)

Generate the expression that replaces `left ~ middle --> right` in the model body.
"""
function generate_tilde_longrightarrow(left, middle, right)
    isliteral(left) && error("Observing `a` is not supported in `a ~ b --> c`") # TODO

    @gensym vn isassumption model retval

    return quote
        $model = $middle
        $vn = $(DynamicPPL.resolve_varnames)(
            $(AbstractPPL.drop_escape(varname(left, need_concretize(left)))), $model
        )
        $isassumption = $(DynamicPPL.isassumption(left, vn))
        if $(DynamicPPL.isfixed(left, vn))
            error("Fixing `a` is not supported in `a ~ b --> c`") # TODO
        elseif $isassumption
            $(generate_tilde_assume(left, model, vn, right))
        else
            error("Observing `a` is not supported in `a ~ b --> c`") # TODO
        end
    end
end

function generate_tilde_assume(left, dist_or_model, vn, maybe_right)
    # HACK: Because the Setfield.jl macro does not support assignment
    # with multiple arguments on the LHS, we need to capture the return-values
    # and then update the LHS variables one by one.

    @gensym value

    has_right = maybe_right !== nothing
    expr = if has_right
        :(($left, $maybe_right) = $value)
    else
        :($left = $value)
    end

    # TODO(penelopeysm): What does this line even do? Not sure if I need to modify it for the
    # a ~ b --> c case.
    if left isa Expr
        expr = AbstractPPL.drop_escape(
            Accessors.setmacro(BangBang.prefermutation, expr; overwrite=true)
        )
    end

    return quote
        $value, __varinfo__ = $(DynamicPPL.tilde_assume!!)(
            __context__,
            $(DynamicPPL.unwrap_right_vn)(
                $(DynamicPPL.check_tilde_rhs)($dist_or_model), $vn
            )...,
            __varinfo__,
            $has_right,
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
        $dist = DynamicPPL.check_dot_tilde_rhs($right)
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

const FloatOrArrayType = Type{<:Union{AbstractFloat,AbstractArray}}
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
function build_output(modeldef, linenumbernode)
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
            :(__varinfo__::$(DynamicPPL.AbstractVarInfo)),
            :(__context__::$(DynamicPPL.AbstractContext)),
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

    # Update the function body of the user-specified model.
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site
    modeldef[:body] = MacroTools.@q begin
        $(linenumbernode)
        return $(DynamicPPL.Model)($name, $args_nt; $(kwargs_inclusion...))
    end

    return MacroTools.@q begin
        $(MacroTools.combinedef(evaluatordef))
        $(Base).@__doc__ $(MacroTools.combinedef(modeldef))
    end
end

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return nothing
end

# TODO(mhauru) matchingvalue has methods that can accept both types and values. Why?
# TODO(mhauru) This function needs a more comprehensive docstring.
"""
    matchingvalue(vi, value)

Convert the `value` to the correct type for the `vi` object.
"""
function matchingvalue(vi, value)
    T = typeof(value)
    if hasmissing(T)
        _value = convert(get_matching_type(vi, T), value)
        # TODO(mhauru) Why do we make a deepcopy, even though in the !hasmissing branch we
        # are happy to return `value` as-is?
        if _value === value
            return deepcopy(_value)
        else
            return _value
        end
    else
        return value
    end
end

function matchingvalue(vi, value::FloatOrArrayType)
    return get_matching_type(vi, value)
end
function matchingvalue(vi, ::TypeWrap{T}) where {T}
    return TypeWrap{get_matching_type(vi, T)}()
end

# TODO(mhauru) This function needs a more comprehensive docstring. What is it for?
"""
    get_matching_type(vi, ::TypeWrap{T}) where {T}

Get the specialized version of type `T` for `vi`.
"""
get_matching_type(_, ::Type{T}) where {T} = T
function get_matching_type(vi, ::Type{<:Union{Missing,AbstractFloat}})
    return Union{Missing,float_type_with_fallback(eltype(vi))}
end
function get_matching_type(vi, ::Type{<:AbstractFloat})
    return float_type_with_fallback(eltype(vi))
end
function get_matching_type(vi, ::Type{<:Array{T,N}}) where {T,N}
    return Array{get_matching_type(vi, T),N}
end
function get_matching_type(vi, ::Type{<:Array{T}}) where {T}
    return Array{get_matching_type(vi, T)}
end
