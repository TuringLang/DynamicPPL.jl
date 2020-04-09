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

const RESERVEDNAMES = (:_model, :_sampler, :_context, :_varinfo)

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
    return esc(model(expr))
end

function model(expr)
    modelinfo = build_model_info(expr)

    ex = generate_main_body(modelinfo[:main_body], modelinfo[:args])
    modelinfo[:main_body] = ex

    return build_output(modelinfo)
end

generate_main_body(x, args) = generate_main_body(x, args, Symbol[])
generate_main_body(x, args, checked) = x
function generate_main_body(sym::Symbol, args, checked)
    if sym in RESERVEDNAMES && sym ∉ checked
        @warn "you are using the reserved name `$(sym)`"
        push!(checked, sym)
    end
    return sym
end
function generate_main_body(expr::Expr, args, checked)
    # do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Apply the `@.` macro first.
    if Meta.isexpr(expr, :macrocall) && length(expr.args) > 1 &&
        expr.args[1] === Symbol("@__dot__")
        return generate_main_body(Base.Broadcast.__dot__(expr.args[end]), args, checked)
    end

    # Check dot tilde.
    args_dottilde = getargs_dottilde(expr)
    if args_dottilde !== nothing
        L, R = args_dottilde
        return Base.remove_linenums!(generate_dot_tilde(generate_main_body(L, args, checked),
                                                        generate_main_body(R, args, checked),
                                                        args))
    end

    # Check tilde.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        return Base.remove_linenums!(generate_tilde(generate_main_body(L, args, checked),
                                                    generate_main_body(R, args, checked),
                                                    args))
    end

    return Expr(expr.head, map(x -> generate_main_body(x, args, checked), expr.args)...)
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
        :whereparams => modeldef[:whereparams]
    )

    return model_info
end

"""
    replace_tilde!(model_info)

Replace `~` and `.~` expressions with observation or assumption expressions, updating `model_info`.
"""
function replace_tilde!(model_info)
    # Apply the `@.` macro first.
    expr = model_info[:main_body]
    dottedexpr = MacroTools.postwalk(apply_dotted, expr)


    # Update the function body.
    model_info[:main_body] = tildeexpr

    return model_info
end

# """ Unbreak code highlighting in Emacs julia-mode


"""
    generate_tilde(left, right, args)

Generate `observe` expressions for data variables and `assume` expressions for parameter
variables for a model with the given `args`.
"""
function generate_tilde(left, right, args)
    @gensym tmpright
    top = [:($tmpright = $right),
           :($tmpright isa Union{$Distribution,AbstractVector{<:$Distribution}}
             || throw(ArgumentError($DISTMSG)))]

    if left isa Symbol || left isa Expr
        @gensym out vn inds
        push!(top, :($vn = $(varname(left))), :($inds = $(vinds(left))))

        assumption = [
            :($out = $(DynamicPPL.tilde_assume)(_context, _sampler, $tmpright, $vn, $inds,
                                                _varinfo)),
            :($left = $out[1]),
            :($(DynamicPPL.acclogp!)(_varinfo, $out[2]))
        ]

        # It can only be an observation if the LHS is an argument of the model
        if vsym(left) in args
            @gensym isassumption
            return quote
                $(top...)
                $isassumption = $(DynamicPPL.isassumption(:_model, left))
                if $isassumption
                    $(assumption...)
                else
                    $(DynamicPPL.acclogp!)(
                        _varinfo,
                        $(DynamicPPL.tilde_observe)(_context, _sampler, $tmpright, $left, $vn,
                                                    $inds, _varinfo)
                    )
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
        $(DynamicPPL.acclogp!)(
            _varinfo,
            $(DynamicPPL.tilde_observe)(_context, _sampler, $tmpright, $left, _varinfo)
        )
    end
end

"""
    generate_dot_tilde(left, right, args)

Generate broadcasted `observe` expressions for data variables and `assume` expressions for parameter
variables for a model with the given `args`.
"""
function generate_dot_tilde(left, right, args)
    @gensym tmpright
    top = [:($tmpright = $right),
           :($tmpright isa Union{$Distribution,AbstractVector{<:$Distribution}}
             || throw(ArgumentError($DISTMSG)))]

    if left isa Symbol || left isa Expr
        @gensym out vn inds
        push!(top, :($vn = $(varname(left))), :($inds = $(vinds(left))))

        assumption = [
            :($out = $(DynamicPPL.dot_tilde_assume)(_context, _sampler, $tmpright, $left,
                                                    $vn, $inds, _varinfo)),
            :($left .= $out[1]),
            :($(DynamicPPL.acclogp!)(_varinfo, $out[2]))
        ]

        # It can only be an observation if the LHS is an argument of the model
        if vsym(left) in args
            @gensym isassumption
            return quote
                $(top...)
                $isassumption = $(DynamicPPL.isassumption(:_model, left))
                if $isassumption
                    $(assumption...)
                else
                    $(DynamicPPL.acclogp!)(
                        _varinfo,
                        $(DynamicPPL.dot_tilde_observe)(_context, _sampler, $tmpright, $left,
                                                        $vn, $inds, _varinfo)
                    )
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
        $(DynamicPPL.acclogp!)(
            _varinfo,
            $(DynamicPPL.dot_tilde_observe)(_context, _sampler, $tmpright, $left, _varinfo)
        )
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

    ex = quote
        function $evaluator(
            _model::$(DynamicPPL.Model),
            _varinfo::$(DynamicPPL.VarInfo),
            _sampler::$(DynamicPPL.AbstractSampler),
            _context::$(DynamicPPL.AbstractContext),
        )
            $unwrap_data_expr
            $(DynamicPPL.resetlogp!)(_varinfo)
            $main_body
        end

        $generator($(args...)) = $(DynamicPPL.Model)($evaluator, $args_nt, $model_gen_constructor)
        $(generator_kw_form...)

        $model_gen = $model_gen_constructor
    end

    return esc(ex)
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
