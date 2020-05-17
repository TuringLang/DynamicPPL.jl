const DISTMSG = "Right-hand side of a ~ must be subtype of Distribution or a vector of " *
    "Distributions."

const INTERNALNAMES = (:_model, :_sampler, :_context, :_varinfo)

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

# Example

Model definition:

```julia
@model function model_generator(x = default_x, y)
    ...
end
```

To generate a `Model`, call `model_generator(x_value)`.
"""
macro model(expr, warn=true)
    esc(model(expr, warn))
end

function model(expr, warn)
    modelinfo = build_model_info(expr)

    # Generate main body
    modelinfo[:main_body] = generate_mainbody(modelinfo[:main_body], modelinfo[:args], warn)

    return build_output(modelinfo)
end

"""
    build_model_info(input_expr)

Builds the `model_info` dictionary from the model's expression.
"""
function build_model_info(input_expr)
    # Extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = ExprTools.splitdef(input_expr)
    modeldef[:whereparams] = get(modeldef,:whereparams, [])
    modeldef[:args] = get(modeldef,:args, [])
    # Function body of the model is empty
    warn_empty(modeldef[:body])

    # Extracting the argument symbols from the model definition
    arg_syms = map(modeldef[:args]) do arg
        # @model demo(x)
        if arg isa Symbol
            arg
        elseif arg isa Expr && arg.head == :kw
            sym, default = arg.args
            # TODO support x::Type{T}
            # @model demo(::Type{T}=Float64) where {T}
            if !isnothing(get_type(sym))
                get_type(sym)
            # @model demo(x::Int = 1)
            elseif sym == Expr(:(::), IsEqual(issymbol), IsEqual(issymbol))
                sym.args[1]
            # @model demo(x = 1)
            elseif sym isa Symbol
                sym
            else
                throw(ArgumentError("Unsupported default argument $arg to the `@model` macro."))
            end
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
    args = map(get(modeldef, :args, [])) do arg
        if (arg isa Symbol)
            arg
        elseif !isnothing(get_type(arg.args[1]))
            Texpr, Tval = arg.args
            T = get_type(Texpr)
            S = nothing
            for x in modeldef[:whereparams]
                if x == T 
                    S = :Any
                elseif x isa Expr && x.args[1] == T
                    S = x.args[2]
                end
            end
            !isnothing(S) || throw(ArgumentError("Please make sure type parameters are properly used. Every `Type{T}` argument need to have `T` in the a `where` clause"))
            Expr(:kw, :($T::Type{<:$S}), Tval)
        else
            arg
        end
    end
    args_nt = to_namedtuple_expr(arg_syms)

    default_syms = []
    default_vals = [] 
    foreach(get(modeldef, :args, [])) do arg
        if arg == Expr(:kw, IsEqual(x->true), IsEqual(x->true))
            var, val = arg.args
            sym = if var isa Symbol
                var
            elseif var == Expr(:(::), IsEqual(issymbol), IsEqual(issymbol))
                var.args[1]
            elseif !isnothing(get_type(var))
                get_type(var)
            else
                error("Unsupported keyword argument given $arg")
            end
            push!(default_syms, sym)
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
        :whereparams => modeldef[:whereparams]
    )

    return model_info
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
                        _context, _sampler, $tmpright, $vn, $inds, _varinfo)
                else
                    $(DynamicPPL.tilde_observe)(
                        _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
                end
            end
        end

        return quote
            $(top...)
            $left = $(DynamicPPL.tilde_assume)(_context, _sampler, $tmpright, $vn, $inds,
                                               _varinfo)
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
                        _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
                else
                    $(DynamicPPL.dot_tilde_observe)(
                        _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
                end
            end
        end

        return quote
            $(top...)
            $left .= $(DynamicPPL.dot_tilde_assume)(
                _context, _sampler, $tmpright, $left, $vn, $inds, _varinfo)
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
    build_output(model_info)

Builds the output expression.
"""
function build_output(model_info)
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
              :($var = $(DynamicPPL.matchingvalue)(_sampler, _varinfo, _model.args.$var)))
    end

    @gensym(evaluator, generator)
    generator_kw_form = isempty(args) ? () : (:($generator(;$(args...)) = $generator($(arg_syms...))),)
    model_gen_constructor = :($(DynamicPPL.ModelGen){$(Tuple(arg_syms))}($generator, $defaults_nt))

    return quote
        function $evaluator(
            _model::$(DynamicPPL.Model),
            _varinfo::$(DynamicPPL.AbstractVarInfo),
            _sampler::$(DynamicPPL.AbstractSampler),
            _context::$(DynamicPPL.AbstractContext),
        )
            $unwrap_data_expr
            $main_body
        end

        $generator($(args...)) = $(DynamicPPL.Model)($evaluator, $args_nt, $model_gen_constructor)
        $(generator_kw_form...)

        $(Base).@__doc__ $model_gen = $model_gen_constructor
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
