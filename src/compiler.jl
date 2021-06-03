const INTERNALNAMES = (:__model__, :__sampler__, :__context__, :__varinfo__, :__rng__)
const DEPRECATED_INTERNALNAMES = (:_model, :_sampler, :_context, :_varinfo, :_rng)

"""
    isassumption(expr)

Return an expression that can be evaluated to check if `expr` is an assumption in the
model.

Let `expr` be `:(x[1])`. It is an assumption in the following cases:
    1. `x` is not among the input data to the model,
    2. `x` is among the input data to the model but with a value `missing`, or
    3. `x` is among the input data to the model with a value other than missing,
       but `x[1] === missing`.

When `expr` is not an expression or symbol (i.e., a literal), this expands to `false`.
"""
function isassumption(expr::Union{Symbol,Expr})
    vn = gensym(:vn)

    return quote
        let $vn = $(varname(expr))
            # This branch should compile nicely in all cases except for partial missing data
            # For example, when `expr` is `:(x[i])` and `x isa Vector{Union{Missing, Float64}}`
            if !$(DynamicPPL.inargnames)($vn, __model__) ||
               $(DynamicPPL.inmissings)($vn, __model__)
                true
            else
                # Evaluate the LHS
                $expr === missing
            end
        end
    end
end

# failsafe: a literal is never an assumption
isassumption(expr) = :(false)

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
"""
unwrap_right_left_vns(right, left, vns) = right, left, vns
function unwrap_right_left_vns(right::NamedDist, left, vns)
    return unwrap_right_left_vns(right.dist, left, right.name)
end
function unwrap_right_left_vns(
    right::MultivariateDistribution, left::AbstractMatrix, vn::VarName
)
    vns = map(axes(left, 2)) do i
        return VarName(vn, (vn.indexing..., Tuple(i)))
    end
    return unwrap_right_left_vns(right, left, vns)
end
function unwrap_right_left_vns(
    right::Union{Distribution,AbstractArray{<:Distribution}},
    left::AbstractArray,
    vn::VarName,
)
    vns = map(CartesianIndices(left)) do i
        return VarName(vn, (vn.indexing..., Tuple(i)))
    end
    return unwrap_right_left_vns(right, left, vns)
end

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
    modelinfo = build_model_info(expr)

    # Generate main body
    modelinfo[:body] = generate_mainbody(mod, modelinfo[:modeldef][:body], warn)

    return build_output(modelinfo, linenumbernode)
end

"""
    build_model_info(input_expr)

Builds the `model_info` dictionary from the model's expression.
"""
function build_model_info(input_expr)
    # Break up the model definition and extract its name, arguments, and function body
    modeldef = MacroTools.splitdef(input_expr)

    # Print a warning if function body of the model is empty
    warn_empty(modeldef[:body])

    ## Construct model_info dictionary

    # Shortcut if the model does not have any arguments
    if !haskey(modeldef, :args) && !haskey(modeldef, :kwargs)
        modelinfo = Dict(
            :allargs_exprs => [],
            :allargs_syms => [],
            :allargs_namedtuple => NamedTuple(),
            :defaults_namedtuple => NamedTuple(),
            :modeldef => modeldef,
        )
        return modelinfo
    end

    # Extract the positional and keyword arguments from the model definition.
    allargs = vcat(modeldef[:args], modeldef[:kwargs])

    # Split the argument expressions and the default values.
    allargs_exprs_defaults = map(allargs) do arg
        MacroTools.@match arg begin
            (x_ = val_) => (x, val)
            x_ => (x, NO_DEFAULT)
        end
    end

    # Extract the expressions of the arguments, without default values.
    allargs_exprs = first.(allargs_exprs_defaults)

    # Extract the names of the arguments.
    allargs_syms = map(allargs_exprs) do arg
        MacroTools.@match arg begin
            (::Type{T_}) | (name_::Type{T_}) => T
            name_::T_ => name
            x_ => x
        end
    end

    # Build named tuple expression of the argument symbols and variables of the same name.
    allargs_namedtuple = to_namedtuple_expr(allargs_syms)

    # Extract default values of the positional and keyword arguments.
    default_syms = []
    default_vals = []
    for (sym, (expr, val)) in zip(allargs_syms, allargs_exprs_defaults)
        if val !== NO_DEFAULT
            push!(default_syms, sym)
            push!(default_vals, val)
        end
    end

    # Build named tuple expression of the argument symbols with default values.
    defaults_namedtuple = to_namedtuple_expr(default_syms, default_vals)

    modelinfo = Dict(
        :allargs_exprs => allargs_exprs,
        :allargs_syms => allargs_syms,
        :allargs_namedtuple => allargs_namedtuple,
        :defaults_namedtuple => defaults_namedtuple,
        :modeldef => modeldef,
    )

    return modelinfo
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
    if sym in DEPRECATED_INTERNALNAMES
        newsym = Symbol(:_, sym, :__)
        Base.depwarn(
            "internal variable `$sym` is deprecated, use `$newsym` instead.",
            :generate_mainbody!,
        )
        return generate_mainbody!(mod, found, newsym, warn)
    end

    if warn && sym in INTERNALNAMES && sym ∉ found
        @warn "you are using the internal variable `$sym`"
        push!(found, sym)
    end

    return sym
end
function generate_mainbody!(mod, found, expr::Expr, warn)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

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

"""
    generate_tilde(left, right)

Generate an `observe` expression for data variables and `assume` expression for parameter
variables.
"""
function generate_tilde(left, right)
    # If the LHS is a literal, it is always an observation
    if !(left isa Symbol || left isa Expr)
        return quote
            $(DynamicPPL.tilde_observe!)(
                __context__, $(DynamicPPL.check_tilde_rhs)($right), $left, __varinfo__
            )
        end
    end

    # Otherwise it is determined by the model or its value,
    # if the LHS represents an observation
    @gensym vn inds isassumption
    return quote
        $vn = $(varname(left))
        $inds = $(vinds(left))
        $isassumption = $(DynamicPPL.isassumption(left))
        if $isassumption
            $left = $(DynamicPPL.tilde_assume!)(
                __context__,
                $(DynamicPPL.unwrap_right_vn)(
                    $(DynamicPPL.check_tilde_rhs)($right), $vn
                )...,
                $inds,
                __varinfo__,
            )
        else
            $(DynamicPPL.tilde_observe!)(
                __context__,
                $(DynamicPPL.check_tilde_rhs)($right),
                $left,
                $vn,
                $inds,
                __varinfo__,
            )
        end
    end
end

"""
    generate_dot_tilde(left, right)

Generate the expression that replaces `left .~ right` in the model body.
"""
function generate_dot_tilde(left, right)
    # If the LHS is a literal, it is always an observation
    if !(left isa Symbol || left isa Expr)
        return quote
            $(DynamicPPL.dot_tilde_observe!)(
                __context__, $(DynamicPPL.check_tilde_rhs)($right), $left, __varinfo__
            )
        end
    end

    # Otherwise it is determined by the model or its value,
    # if the LHS represents an observation
    @gensym vn inds isassumption
    return quote
        $vn = $(varname(left))
        $inds = $(vinds(left))
        $isassumption = $(DynamicPPL.isassumption(left))
        if $isassumption
            $left .= $(DynamicPPL.dot_tilde_assume!)(
                __context__,
                $(DynamicPPL.unwrap_right_left_vns)(
                    $(DynamicPPL.check_tilde_rhs)($right), $left, $vn
                )...,
                $inds,
                __varinfo__,
            )
        else
            $(DynamicPPL.dot_tilde_observe!)(
                __context__,
                $(DynamicPPL.check_tilde_rhs)($right),
                $left,
                $vn,
                $inds,
                __varinfo__,
            )
        end
    end
end

const FloatOrArrayType = Type{<:Union{AbstractFloat,AbstractArray}}
hasmissing(T::Type{<:AbstractArray{TA}}) where {TA<:AbstractArray} = hasmissing(TA)
hasmissing(T::Type{<:AbstractArray{>:Missing}}) = true
hasmissing(T::Type) = false

"""
    build_output(modelinfo, linenumbernode)

Builds the output expression.
"""
function build_output(modelinfo, linenumbernode)
    ## Build the anonymous evaluator from the user-provided model definition.

    # Remove the name.
    evaluatordef = deepcopy(modelinfo[:modeldef])
    delete!(evaluatordef, :name)

    # Add the internal arguments to the user-specified arguments (positional + keywords).
    evaluatordef[:args] = vcat(
        [
            :(__model__::$(DynamicPPL.Model)),
            :(__varinfo__::$(DynamicPPL.AbstractVarInfo)),
            :(__context__::$(DynamicPPL.AbstractContext)),
        ],
        modelinfo[:allargs_exprs],
    )

    # Delete the keyword arguments.
    evaluatordef[:kwargs] = []

    # Replace the user-provided function body with the version created by DynamicPPL.
    @gensym leafctx
    evaluatordef[:body] = quote
        # in case someone accessed these
        $leafctx = DynamicPPL.unwrap(__context__)
        if $leafctx isa $(DynamicPPL.SamplingContext)
            __rng__ = $leafctx.rng
            __sampler__ = $leafctx.sampler
        end

        $(modelinfo[:body])
    end

    ## Build the model function.

    # Extract the named tuple expression of all arguments and the default values.
    allargs_namedtuple = modelinfo[:allargs_namedtuple]
    defaults_namedtuple = modelinfo[:defaults_namedtuple]

    # Update the function body of the user-specified model.
    # We use a name for the anonymous evaluator that does not conflict with other variables.
    modeldef = modelinfo[:modeldef]
    @gensym evaluator
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site
    modeldef[:body] = MacroTools.@q begin
        $(linenumbernode)
        $evaluator = $(MacroTools.combinedef(evaluatordef))
        return $(DynamicPPL.Model)(
            $(QuoteNode(modeldef[:name])),
            $evaluator,
            $allargs_namedtuple,
            $defaults_namedtuple,
        )
    end

    return :($(Base).@__doc__ $(MacroTools.combinedef(modeldef)))
end

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return nothing
end

"""
    matchingvalue(sampler, vi, value)

Convert the `value` to the correct type for the `sampler` and the `vi` object.
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
matchingvalue(sampler, vi, value::FloatOrArrayType) = get_matching_type(sampler, vi, value)

"""
    get_matching_type(spl::AbstractSampler, vi, ::Type{T}) where {T}

Get the specialized version of type `T` for sampler `spl`.

For example, if `T === Float64` and `spl::Hamiltonian`, the matching type is
`eltype(vi[spl])`.
"""
get_matching_type(spl::AbstractSampler, vi, ::Type{T}) where {T} = T
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:Union{Missing,AbstractFloat}})
    return Union{Missing,floatof(eltype(vi, spl))}
end
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:AbstractFloat})
    return floatof(eltype(vi, spl))
end
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:Array{T,N}}) where {T,N}
    return Array{get_matching_type(spl, vi, T),N}
end
function get_matching_type(spl::AbstractSampler, vi, ::Type{<:Array{T}}) where {T}
    return Array{get_matching_type(spl, vi, T)}
end

floatof(::Type{T}) where {T<:Real} = typeof(one(T) / one(T))
floatof(::Type) = Real # fallback if type inference failed
