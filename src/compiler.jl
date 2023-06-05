const INTERNALNAMES = (:__model__, :__context__, :__varinfo__)

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
If `vn` is not specified, `AbstractPPL.drop_escape(varname(expr))` will be
used in its place.
"""
function isassumption(expr::Union{Expr,Symbol}, vn=AbstractPPL.drop_escape(varname(expr)))
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
    if hasvalue(context, vn)
        val = getvalue(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return true
        else
            return false
        end
    end

    # We might have nested contexts, e.g. `ContextionContext{.., <:PrefixContext{..., <:ConditionContext}}`
    # so we defer to `childcontext` if we haven't concluded that anything yet.
    return contextual_isassumption(childcontext(context), vn)
end
function contextual_isassumption(context::PrefixContext, vn)
    return contextual_isassumption(childcontext(context), prefix(context, vn))
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
isliteral(e::Expr) = !isempty(e.args) && all(isliteral, e.args)

"""
    check_tilde_rhs(x)

Check if the right-hand side `x` of a `~` is a `Distribution` or an array of
`Distributions`, then return `x`.
"""
function check_tilde_rhs(@nospecialize(x))
    return throw(
        ArgumentError(
            "the right-hand side of a `~` must be a `Distribution` or an array of `Distribution`s",
        ),
    )
end
check_tilde_rhs(x::Distribution) = x
check_tilde_rhs(x::AbstractArray{<:Distribution}) = x

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
x[:,2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(1, 2), @varname(x)); vns[end]
x[1,2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(1, 2), @varname(x[:])); vns[end]
x[:][1,2]

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
        return vn ∘ Setfield.IndexLens((Colon(), i))
    end
    return unwrap_right_left_vns(right, left, vns)
end
function unwrap_right_left_vns(
    right::Union{Distribution,AbstractArray{<:Distribution}},
    left::AbstractArray,
    vn::VarName,
)
    vns = map(CartesianIndices(left)) do i
        return vn ∘ Setfield.IndexLens(Tuple(i))
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
        return Base.remove_linenums!(
            generate_dot_tilde(
                generate_mainbody!(mod, found, L, warn),
                generate_mainbody!(mod, found, R, warn),
            ),
        )
    end

    # Modify tilde operators.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        return Base.remove_linenums!(
            generate_tilde(
                generate_mainbody!(mod, found, L, warn),
                generate_mainbody!(mod, found, R, warn),
            ),
        )
    end

    return Expr(expr.head, map(x -> generate_mainbody!(mod, found, x, warn), expr.args)...)
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
            $(AbstractPPL.drop_escape(varname(left))), $dist
        )
        $isassumption = $(DynamicPPL.isassumption(left, vn))
        if $isassumption
            $(generate_tilde_assume(left, dist, vn))
        else
            # If `vn` is not in `argnames`, we need to make sure that the variable is defined.
            if !$(DynamicPPL.inargnames)($vn, __model__)
                $left = $(DynamicPPL.getvalue_nested)(__context__, $vn)
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

function generate_tilde_assume(left, right, vn)
    # HACK: Because the Setfield.jl macro does not support assignment
    # with multiple arguments on the LHS, we need to capture the return-values
    # and then update the LHS variables one by one.
    @gensym value
    expr = :($left = $value)
    if left isa Expr
        expr = AbstractPPL.drop_escape(
            Setfield.setmacro(BangBang.prefermutation, expr; overwrite=true)
        )
    end

    return quote
        $value, __varinfo__ = $(DynamicPPL.tilde_assume!!)(
            __context__,
            $(DynamicPPL.unwrap_right_vn)($(DynamicPPL.check_tilde_rhs)($right), $vn)...,
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
    isliteral(left) && return generate_tilde_literal(left, right)

    # Otherwise it is determined by the model or its value,
    # if the LHS represents an observation
    @gensym vn isassumption value
    return quote
        $vn = $(DynamicPPL.resolve_varnames)(
            $(AbstractPPL.drop_escape(varname(left))), $right
        )
        $isassumption = $(DynamicPPL.isassumption(left, vn))
        if $isassumption
            $(generate_dot_tilde_assume(left, right, vn))
        else
            # If `vn` is not in `argnames`, we need to make sure that the variable is defined.
            if !$(DynamicPPL.inargnames)($vn, __model__)
                $left .= $(DynamicPPL.getvalue_nested)(__context__, $vn)
            end

            $value, __varinfo__ = $(DynamicPPL.dot_tilde_observe!!)(
                __context__,
                $(DynamicPPL.check_tilde_rhs)($right),
                $(maybe_view(left)),
                $vn,
                __varinfo__,
            )
            $value
        end
    end
end

function generate_dot_tilde_assume(left, right, vn)
    # We don't need to use `Setfield.@set` here since
    # `.=` is always going to be inplace + needs `left` to
    # be something that supports `.=`.
    @gensym value
    return quote
        $value, __varinfo__ = $(DynamicPPL.dot_tilde_assume!!)(
            __context__,
            $(DynamicPPL.unwrap_right_left_vns)(
                $(DynamicPPL.check_tilde_rhs)($right), $(maybe_view(left)), $vn
            )...,
            __varinfo__,
        )
        $left .= $value
        $value
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
    if isfuncdef(e)
        return e
    end

    if Meta.isexpr(e, :return)
        # We capture the original return-value in `retval` and return
        # a `Tuple{typeof(retval),typeof(__varinfo__)}`.
        # If we don't capture the return-value separately, cases such as
        # `return x = 1` will result in `(x = 1, __varinfo__)` which will
        # mistakenly attempt to construct a `NamedTuple` (which fails on Julia 1.3
        # and is not our intent).
        @gensym retval
        return quote
            $retval = $(e.args...)
            return $retval, __varinfo__
        end
    end

    return Expr(e.head, map(replace_returns, e.args)...)
end

# If it's just a symbol, e.g. `f(x) = 1`, then we make it `f(x) = return 1`.
make_returns_explicit!(body) = Expr(:return, body)
function make_returns_explicit!(body::Expr)
    # If the last statement is a return-statement, we don't do anything.
    # Otherwise we replace the last statement with a `return` statement.
    if !Meta.isexpr(body.args[end], :return)
        body.args[end] = Expr(:return, body.args[end])
    end
    return body
end

const FloatOrArrayType = Type{<:Union{AbstractFloat,AbstractArray}}
hasmissing(::Type) = false
hasmissing(::Type{>:Missing}) = true
hasmissing(::Type{<:AbstractArray{TA}}) where {TA} = hasmissing(TA)
hasmissing(::Type{Union{}}) = false # issue #368

function splitarg_to_expr((arg_name, arg_type, is_splat, default))
    return is_splat ? :($arg_name...) : arg_name
end

function namedtuple_from_splitargs(splitargs)
    names = map(splitargs) do (arg_name, arg_type, is_splat, default)
        is_splat ? Symbol("#splat#$(arg_name)") : arg_name
    end
    names_expr = Expr(:tuple, map(QuoteNode, names)...)
    vals = Expr(:tuple, map(first, splitargs)...)
    return :(NamedTuple{$names_expr}($vals))
end

is_splat_symbol(s::Symbol) = startswith(string(s), "#splat#")

"""
    build_output(modeldef, linenumbernode)

Builds the output expression.
"""
function build_output(modeldef, linenumbernode)
    args = modeldef[:args]
    kwargs = modeldef[:kwargs]

    ## Build the anonymous evaluator from the user-provided model definition.
    evaluatordef = deepcopy(modeldef)

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
        $(replace_returns(make_returns_explicit!(modeldef[:body])))
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

"""
    matchingvalue(sampler, vi, value)
    matchingvalue(context::AbstractContext, vi, value)

Convert the `value` to the correct type for the `sampler` or `context` and the `vi` object.

For a `context` that is _not_ a `SamplingContext`, we fall back to
`matchingvalue(SampleFromPrior(), vi, value)`.
"""
function matchingvalue(sampler, vi, value)
    T = typeof(value)
    if hasmissing(T)
        _value = convert(get_matching_type(sampler, vi, T), value)
        if _value === value
            return deepcopy(_value)
        else
            return _value
        end
    else
        return value
    end
end
function matchingvalue(sampler::AbstractSampler, vi, value::FloatOrArrayType)
    return get_matching_type(sampler, vi, value)
end

function matchingvalue(context::AbstractContext, vi, value)
    return matchingvalue(NodeTrait(matchingvalue, context), context, vi, value)
end
function matchingvalue(::IsLeaf, context::AbstractContext, vi, value)
    return matchingvalue(SampleFromPrior(), vi, value)
end
function matchingvalue(::IsParent, context::AbstractContext, vi, value)
    return matchingvalue(childcontext(context), vi, value)
end
function matchingvalue(context::SamplingContext, vi, value)
    return matchingvalue(context.sampler, vi, value)
end

"""
    get_matching_type(spl::AbstractSampler, vi, ::Type{T}) where {T}

Get the specialized version of type `T` for sampler `spl`.

For example, if `T === Float64` and `spl::Hamiltonian`, the matching type is
`eltype(vi[spl])`.
"""
get_matching_type(spl::AbstractSampler, vi, ::Type{T}) where {T} = T
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:Union{Missing,AbstractFloat}})
    return Union{Missing,float_type_with_fallback(eltype(vi, spl))}
end
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:AbstractFloat})
    return float_type_with_fallback(eltype(vi, spl))
end
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:Array{T,N}}) where {T,N}
    return Array{get_matching_type(spl, vi, T),N}
end
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:Array{T}}) where {T}
    return Array{get_matching_type(spl, vi, T)}
end
