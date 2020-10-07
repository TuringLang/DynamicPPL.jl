const DISTMSG = "Right-hand side of a ~ must be subtype of Distribution or a vector of " *
    "Distributions."

const INTERNALNAMES = (:_model, :_sampler, :_context, :_varinfo, :_rng)

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
function isassumption(expr::Union{Symbol, Expr})
    vn = gensym(:vn)

    return quote
        let $vn = $(varname(expr))
            # This branch should compile nicely in all cases except for partial missing data
            # For example, when `expr` is `:(x[i])` and `x isa Vector{Union{Missing, Float64}}`
            if !$(DynamicPPL.inargnames)($vn, _model) || $(DynamicPPL.inmissings)($vn, _model)
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

#################
# Main Compiler #
#################

"""
    @model(expr[, warn = true])

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
macro model(expr, warn=true)
    esc(model(expr, warn))
end

function model(expr, warn)
    modelinfo = build_model_info(expr)

    # Generate main body
    modelinfo[:body] = generate_mainbody(
        modelinfo[:modeldef][:body], modelinfo[:allargs_exprs], warn
    )

    return build_output(modelinfo)
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
    allargs_syms = map(allargs_exprs_defaults) do (arg, _)
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
    generate_mainbody(expr, args, warn)

Generate the body of the main evaluation function from expression `expr` and arguments
`args`.

If `warn` is true, a warning is displayed if internal variables are used in the model
definition.
"""
generate_mainbody(expr, args, warn) = generate_mainbody!(Symbol[], expr, args, warn)

generate_mainbody!(found, x, args, warn) = x
function generate_mainbody!(found, sym::Symbol, args, warn)
    if warn && sym in INTERNALNAMES && sym ∉ found
        @warn "you are using the internal variable `$(sym)`"
        push!(found, sym)
    end
    return sym
end
function generate_mainbody!(found, expr::Expr, args, warn)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Apply the `@.` macro first.
    if Meta.isexpr(expr, :macrocall) && length(expr.args) > 1 &&
        expr.args[1] === Symbol("@__dot__")
        return generate_mainbody!(found, Base.Broadcast.__dot__(expr.args[end]), args, warn)
    end

    # Modify dotted tilde operators.
    args_dottilde = getargs_dottilde(expr)
    if args_dottilde !== nothing
        L, R = args_dottilde
        return generate_dot_tilde(generate_mainbody!(found, L, args, warn),
                                  generate_mainbody!(found, R, args, warn),
                                  args) |> Base.remove_linenums!
    end

    # Modify tilde operators.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        return generate_tilde(generate_mainbody!(found, L, args, warn),
                              generate_mainbody!(found, R, args, warn),
                              args) |> Base.remove_linenums!
    end

    return Expr(expr.head, map(x -> generate_mainbody!(found, x, args, warn), expr.args)...)
end



"""
    generate_tilde(left, right, args)

Generate an `observe` expression for data variables and `assume` expression for parameter
variables.
"""
function generate_tilde(left, right, args)
    @gensym tmpright
    top = [:($tmpright = $right),
           :($tmpright isa Union{$Distribution,AbstractVector{<:$Distribution}}
             || throw(ArgumentError($DISTMSG)))]

    if left isa Symbol || left isa Expr
        @gensym out vn inds
        push!(top, :($vn = $(varname(left))), :($inds = $(vinds(left))))

        # It can only be an observation if the LHS is an argument of the model
        if vsym(left) in args
            @gensym isassumption
            return quote
                $(top...)
                $isassumption = $(DynamicPPL.isassumption(left))
                if $isassumption
                    $left = $(DynamicPPL.tilde_assume)(
                        _rng, _context, _sampler, $tmpright, $vn, $inds, _varinfo)
                else
                    $(DynamicPPL.tilde_observe)(
                        _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
                end
            end
        end

        return quote
            $(top...)
            $left = $(DynamicPPL.tilde_assume)(_rng, _context, _sampler, $tmpright, $vn,
                                               $inds, _varinfo)
        end
    end

    # If the LHS is a literal, it is always an observation
    return quote
        $(top...)
        $(DynamicPPL.tilde_observe)(_context, _sampler, $tmpright, $left, _varinfo)
    end
end

"""
    generate_dot_tilde(left, right, args)

Generate the expression that replaces `left .~ right` in the model body.
"""
function generate_dot_tilde(left, right, args)
    @gensym tmpright
    top = [:($tmpright = $right),
           :($tmpright isa Union{$Distribution,AbstractVector{<:$Distribution}}
             || throw(ArgumentError($DISTMSG)))]

    if left isa Symbol || left isa Expr
        @gensym out vn inds
        push!(top, :($vn = $(varname(left))), :($inds = $(vinds(left))))

        # It can only be an observation if the LHS is an argument of the model
        if vsym(left) in args
            @gensym isassumption
            return quote
                $(top...)
                $isassumption = $(DynamicPPL.isassumption(left))
                if $isassumption
                    $left .= $(DynamicPPL.dot_tilde_assume)(
                        _rng, _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
                else
                    $(DynamicPPL.dot_tilde_observe)(
                        _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
                end
            end
        end

        return quote
            $(top...)
            $left .= $(DynamicPPL.dot_tilde_assume)(
                _rng, _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
        end
    end

    # If the LHS is a literal, it is always an observation
    return quote
        $(top...)
        $(DynamicPPL.dot_tilde_observe)(_context, _sampler, $tmpright, $left, _varinfo)
    end
end

const FloatOrArrayType = Type{<:Union{AbstractFloat, AbstractArray}}
hasmissing(T::Type{<:AbstractArray{TA}}) where {TA <: AbstractArray} = hasmissing(TA)
hasmissing(T::Type{<:AbstractArray{>:Missing}}) = true
hasmissing(T::Type) = false

"""
    build_output(modelinfo)

Builds the output expression.
"""
function build_output(modelinfo)
    ## Build the anonymous evaluator from the user-provided model definition.

    # Remove the name.
    evaluatordef = deepcopy(modelinfo[:modeldef])
    delete!(evaluatordef, :name)

    # Add the internal arguments to the user-specified arguments (positional + keywords).
    evaluatordef[:args] = vcat(
        [
            :(_rng::$(Random.AbstractRNG)),
            :(_model::$(DynamicPPL.Model)),
            :(_varinfo::$(DynamicPPL.AbstractVarInfo)),
            :(_sampler::$(DynamicPPL.AbstractSampler)),
            :(_context::$(DynamicPPL.AbstractContext)),
        ],
        modelinfo[:allargs_exprs],
    )

    # Delete the keyword arguments.
    evaluatordef[:kwargs] = []

    # Replace the user-provided function body with the version created by DynamicPPL.
    evaluatordef[:body] = modelinfo[:body]

    ## Build the model function.

    # Extract the named tuple expression of all arguments and the default values.
    allargs_namedtuple = modelinfo[:allargs_namedtuple]
    defaults_namedtuple = modelinfo[:defaults_namedtuple]

    # Update the function body of the user-specified model.
    # We use a name for the anonymous evaluator that does not conflict with other variables.
    modeldef = modelinfo[:modeldef]
    @gensym evaluator
    modeldef[:body] = quote
        $evaluator = $(combinedef_anonymous(evaluatordef))
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
    return
end

"""
    matchingvalue(sampler, vi, value)

Convert the `value` to the correct type for the `sampler` and the `vi` object.
"""
function matchingvalue(sampler, vi, value)
    T = typeof(value)
    if hasmissing(T)
        return get_matching_type(sampler, vi, T)(value)
    else
        return value
    end
end
matchingvalue(sampler, vi, value::FloatOrArrayType) = get_matching_type(sampler, vi, value)

"""
    get_matching_type(spl, vi, ::Type{T}) where {T}
Get the specialized version of type `T` for sampler `spl`. For example,
if `T === Float64` and `spl::Hamiltonian`, the matching type is `eltype(vi[spl])`.
"""
function get_matching_type end
