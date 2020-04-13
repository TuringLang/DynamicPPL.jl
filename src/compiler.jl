macro varinfo()
    :(throw(_error_msg()))
end
macro logpdf()
    :(throw(_error_msg()))
end
macro sampler()
    :(throw(_error_msg()))
end
function _error_msg()
    return "This macro is only for use in the `@model` macro and not for external use."
end

const DISTMSG = "Right-hand side of a ~ must be subtype of Distribution or a vector of " *
    "Distributions."

"""
    isassumption(model, expr)

Return an expression that can be evaluated to check if `expr` is an assumption in the
`model`.

Let `expr` be `:(x[1])`. It is an assumption in the following cases:
    1. `x` is not among the input data to the `model`,
    2. `x` is among the input data to the `model` but with a value `missing`, or
    3. `x` is among the input data to the `model` with a value other than missing,
       but `x[1] === missing`.

When `expr` is not an expression or symbol (i.e., a literal), this expands to `false`.
"""
function isassumption(model, expr::Union{Symbol, Expr})
    vn = gensym(:vn)

    return quote
        let $vn = $(varname(expr))
            # This branch should compile nicely in all cases except for partial missing data
            # For example, when `expr` is `:(x[i])` and `x isa Vector{Union{Missing, Float64}}`
            if !$(DynamicPPL.inargnames)($vn, $model) || $(DynamicPPL.inmissings)($vn, $model)
                true
            else
                # Evaluate the LHS
                $expr === missing
            end
        end
    end
end

# failsafe: a literal is never an assumption
isassumption(model, expr) = :(false)

#################
# Main Compiler #
#################

"""
    @model(body)

Macro to specify a probabilistic model.

Example:

Model definition:

```julia
@model model_generator(x = default_x, y) = begin
    ...
end
```

To generate a `Model`, call `model_generator(x_value)`.
"""
macro model(expr)
    esc(model(expr))
end

function model(expr)
    modelinfo = build_model_info(expr)

    # Generate main body
    modelinfo[:main_body] = generate_mainbody(modelinfo)

    return build_output(modelinfo)
end

"""
    build_model_info(input_expr)

Builds the `model_info` dictionary from the model's expression.
"""
function build_model_info(input_expr)
    # Extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(input_expr)
    # Function body of the model is empty
    warn_empty(modeldef[:body])
    # Construct model_info dictionary

    # Extracting the argument symbols from the model definition
    arg_syms = map(modeldef[:args]) do arg
        # @model demo(x)
        if (arg isa Symbol)
            arg
        # @model demo(::Type{T}) where {T}
        elseif MacroTools.@capture(arg, ::Type{T_} = Tval_)
            T
        # @model demo(x::T = 1)
        elseif MacroTools.@capture(arg, x_::T_ = val_)
            x
        # @model demo(x = 1)
        elseif MacroTools.@capture(arg, x_ = val_)
            x
        else
            throw(ArgumentError("Unsupported argument $arg to the `@model` macro."))
        end
    end
    if length(arg_syms) == 0
        args_nt = :(NamedTuple())
    else
        nt_type = Expr(:curly, :NamedTuple, 
            Expr(:tuple, QuoteNode.(arg_syms)...), 
            Expr(:curly, :Tuple, [:(Core.Typeof($x)) for x in arg_syms]...)
        )
        args_nt = Expr(:call, :($namedtuple), nt_type, Expr(:tuple, arg_syms...))
    end
    args = map(modeldef[:args]) do arg
        if (arg isa Symbol)
            arg
        elseif MacroTools.@capture(arg, ::Type{T_} = Tval_)
            if in(T, modeldef[:whereparams])
                S = :Any
            else
                ind = findfirst(modeldef[:whereparams]) do x
                    MacroTools.@capture(x, T1_ <: S_) && T1 == T
                end
                ind !== nothing || throw(ArgumentError("Please make sure type parameters are properly used. Every `Type{T}` argument need to have `T` in the a `where` clause"))
            end
            Expr(:kw, :($T::Type{<:$S}), Tval)
        else
            arg
        end
    end
    args_nt = to_namedtuple_expr(arg_syms)

    default_syms = []
    default_vals = [] 
    foreach(modeldef[:args]) do arg
        # @model demo(::Type{T}) where {T}
        if MacroTools.@capture(arg, ::Type{T_} = Tval_)
            push!(default_syms, T)
            push!(default_vals, Tval)
        # @model demo(x::T = 1)
        elseif MacroTools.@capture(arg, x_::T_ = val_)
            push!(default_syms, x)
            push!(default_vals, val)
        # @model demo(x = 1)
        elseif MacroTools.@capture(arg, x_ = val_)
            push!(default_syms, x)
            push!(default_vals, val)
        end
    end
    defaults_nt = to_namedtuple_expr(default_syms, default_vals)

    model_info = Dict(
        :name => modeldef[:name],
        :main_body => modeldef[:body],
        :arg_syms => arg_syms,
        :args_nt => args_nt,
        :defaults_nt => defaults_nt,
        :args => args,
        :whereparams => modeldef[:whereparams],
        :main_body_names => Dict(
            :ctx => gensym(:ctx),
            :vi => gensym(:vi),
            :sampler => gensym(:sampler),
            :model => gensym(:model)
        )
    )

    return model_info
end

"""
    generate_mainbody([expr, ]modelinfo)

Generate the body of the main evaluation function.
"""
generate_mainbody(modelinfo) = generate_mainbody(modelinfo[:main_body], modelinfo)

generate_mainbody(x, modelinfo) = x
function generate_mainbody(expr::Expr, modelinfo)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Apply the `@.` macro first.
    if Meta.isexpr(expr, :macrocall) && length(expr.args) > 1 &&
        expr.args[1] === Symbol("@__dot__")
        return generate_mainbody(Base.Broadcast.__dot__(expr.args[end]), modelinfo)
    end

    # Modify macro calls.
    if Meta.isexpr(expr, :macrocall) && !isempty(expr.args)
        name = expr.args[1]
        if name === Symbol("@varinfo")
            return modelinfo[:main_body_names][:vi]
        elseif name === Symbol("@logpdf")
            return :($(modelinfo[:main_body_names][:vi]).logp[])
        elseif name === Symbol("@sampler")
            return :($(modelinfo[:main_body_names][:sampler]))
        end
    end

    # Modify dotted tilde operators.
    args_dottilde = getargs_dottilde(expr)
    if args_dottilde !== nothing
        L, R = args_dottilde
        return Base.remove_linenums!(generate_dot_tilde(generate_mainbody(L, modelinfo),
                                                        generate_mainbody(R, modelinfo),
                                                        modelinfo))
    end

    # Modify tilde operators.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        return Base.remove_linenums!(generate_tilde(generate_mainbody(L, modelinfo),
                                                    generate_mainbody(R, modelinfo),
                                                    modelinfo))
    end

    return Expr(expr.head, map(x -> generate_mainbody(x, modelinfo), expr.args)...)
end

"""
    replace_tilde!(model_info)

Replace `~` and `.~` expressions with observation or assumption expressions, updating `model_info`.
"""
function replace_tilde!(model_info)
    # Apply the `@.` macro first.
    expr = model_info[:main_body]
    dottedexpr = MacroTools.postwalk(apply_dotted, expr)

    # Check for tilde operators.
    tildeexpr = MacroTools.postwalk(dottedexpr) do x
        # Check dot tilde first.
        dotargs = getargs_dottilde(x)
        if dotargs !== nothing
            L, R = dotargs
            return Base.remove_linenums!(generate_dot_tilde(L, R, model_info))
        end

        # Check tilde.
        args = getargs_tilde(x)
        if args !== nothing
            L, R = args
            return Base.remove_linenums!(generate_tilde(L, R, model_info))
        end

        return x
    end

    # Update the function body.
    model_info[:main_body] = tildeexpr

    return model_info
end

# """ Unbreak code highlighting in Emacs julia-mode


"""
    generate_tilde(left, right, model_info)

The `tilde` function generates `observe` expression for data variables and `assume` 
expressions for parameter variables, updating `model_info` in the process.
"""
function generate_tilde(left, right, model_info)
    model = model_info[:main_body_names][:model]
    vi = model_info[:main_body_names][:vi]
    ctx = model_info[:main_body_names][:ctx]
    sampler = model_info[:main_body_names][:sampler]

    @gensym tmpright tmpleft
    top = [:($tmpright = $right),
           :($tmpright isa Union{$Distribution,AbstractVector{<:$Distribution}}
             || throw(ArgumentError($DISTMSG)))]

    if left isa Symbol || left isa Expr
        @gensym out vn inds
        push!(top, :($vn = $(varname(left))), :($inds = $(vinds(left))))

        assumption = [
            :($out = $(DynamicPPL.tilde_assume)($ctx, $sampler, $tmpright, $vn, $inds,
                                                $vi)),
            :($(DynamicPPL.acclogp!)($vi, $out[2])),
            :($left = $out[1])
        ]

        # It can only be an observation if the LHS is an argument of the model
        if vsym(left) in model_info[:args]
            @gensym isassumption
            return quote
                $(top...)
                $isassumption = $(DynamicPPL.isassumption(model, left))
                if $isassumption
                    $(assumption...)
                else
                    $tmpleft = $left
                    $(DynamicPPL.acclogp!)(
                        $vi,
                        $(DynamicPPL.tilde_observe)($ctx, $sampler, $tmpright, $tmpleft, $vn,
                                                    $inds, $vi)
                    )
                    $tmpleft
                end
            end
        end

        return quote
            $(top...)
            $(assumption...)
        end
    end

    # If the LHS is a literal, it is always an observation
    return quote
        $(top...)
        $tmpleft = $left
        $(DynamicPPL.acclogp!)(
            $vi,
            $(DynamicPPL.tilde_observe)($ctx, $sampler, $tmpright, $tmpleft, $vi)
        )
        $tmpleft
    end
end

"""
    generate_dot_tilde(left, right, model_info)

This function returns the expression that replaces `left .~ right` in the model body. If
`preprocessed isa VarName`, then a `dot_assume` block will be run. Otherwise, a `dot_observe` block
will be run.
"""
function generate_dot_tilde(left, right, model_info)
    model = model_info[:main_body_names][:model]
    vi = model_info[:main_body_names][:vi]
    ctx = model_info[:main_body_names][:ctx]
    sampler = model_info[:main_body_names][:sampler]

    @gensym tmpright tmpleft
    top = [:($tmpright = $right),
           :($tmpright isa Union{$Distribution,AbstractVector{<:$Distribution}}
             || throw(ArgumentError($DISTMSG)))]

    if left isa Symbol || left isa Expr
        @gensym out vn inds
        push!(top, :($vn = $(varname(left))), :($inds = $(vinds(left))))

        assumption = [
            :($out = $(DynamicPPL.dot_tilde_assume)($ctx, $sampler, $tmpright, $left,
                                                    $vn, $inds, $vi)),
            :($(DynamicPPL.acclogp!)($vi, $out[2])),
            :($left .= $out[1])
        ]

        # It can only be an observation if the LHS is an argument of the model
        if vsym(left) in model_info[:args]
            @gensym isassumption
            return quote
                $(top...)
                $isassumption = $(DynamicPPL.isassumption(model, left))
                if $isassumption
                    $(assumption...)
                else
                    $tmpleft = $left
                    $(DynamicPPL.acclogp!)(
                        $vi,
                        $(DynamicPPL.dot_tilde_observe)($ctx, $sampler, $tmpright, $tmpleft,
                                                        $vn, $inds, $vi)
                    )
                    $tmpleft
                end
            end
        end

        return quote
            $(top...)
            $(assumption...)
        end
    end

    # If the LHS is a literal, it is always an observation
    return quote
        $(top...)
        $tmpleft = $left
        $(DynamicPPL.acclogp!)(
            $vi,
            $(DynamicPPL.dot_tilde_observe)($ctx, $sampler, $tmpright, $tmpleft, $vi)
        )
        $tmpleft
    end
end

const FloatOrArrayType = Type{<:Union{AbstractFloat, AbstractArray}}
hasmissing(T::Type{<:AbstractArray{TA}}) where {TA <: AbstractArray} = hasmissing(TA)
hasmissing(T::Type{<:AbstractArray{>:Missing}}) = true
hasmissing(T::Type) = false

"""
    build_output(model_info)

Builds the output expression.
"""
function build_output(model_info)
    # Construct user-facing function
    main_body_names = model_info[:main_body_names]
    ctx = main_body_names[:ctx]
    vi = main_body_names[:vi]
    model = main_body_names[:model]
    sampler = main_body_names[:sampler]

    # Arguments with default values
    args = model_info[:args]
    # Argument symbols without default values
    arg_syms = model_info[:arg_syms]
    # Arguments namedtuple
    args_nt = model_info[:args_nt]
    # Default values of the arguments
    # Arguments namedtuple
    defaults_nt = model_info[:defaults_nt]
    # Where parameters
    whereparams = model_info[:whereparams]
    # Model generator name
    model_gen = model_info[:name]
    # Main body of the model
    main_body = model_info[:main_body]

    unwrap_data_expr = Expr(:block)
    for var in arg_syms
        push!(unwrap_data_expr.args,
              :($var = $(DynamicPPL.matchingvalue)($sampler, $vi, $(model).args.$var)))
    end

    @gensym(evaluator, generator)
    generator_kw_form = isempty(args) ? () : (:($generator(;$(args...)) = $generator($(arg_syms...))),)
    model_gen_constructor = :($(DynamicPPL.ModelGen){$(Tuple(arg_syms))}($generator, $defaults_nt))

    return quote
        function $evaluator(
            $model::$(DynamicPPL.Model),
            $vi::$(DynamicPPL.VarInfo),
            $sampler::$(DynamicPPL.AbstractSampler),
            $ctx::$(DynamicPPL.AbstractContext),
        )
            $unwrap_data_expr
            $main_body
        end

        $generator($(args...)) = $(DynamicPPL.Model)($evaluator, $args_nt, $model_gen_constructor)
        $(generator_kw_form...)

        $model_gen = $model_gen_constructor
    end
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
