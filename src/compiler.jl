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



# Check if the right-hand side is a distribution.
function assert_dist(dist; msg)
    isa(dist, Distribution) || throw(ArgumentError(msg))
end
function assert_dist(dist::AbstractVector; msg)
    all(d -> isa(d, Distribution), dist) || throw(ArgumentError(msg))
end

function wrong_dist_errormsg(l)
    return "Right-hand side of a ~ must be subtype of Distribution or a vector of " *
        "Distributions on line $(l)."
end

"""
    @isassumption(model, expr)

Let `expr` be `x[1]`. `vn` is an assumption in the following cases:
    1. `x` was not among the input data to the model,
    2. `x` was among the input data to the model but with a value `missing`, or
    3. `x` was among the input data to the model with a value other than missing, 
       but `x[1] === missing`.
When `expr` is not an expression or symbol (i.e., a literal), this expands to `false`.
"""
macro isassumption(model, expr::Union{Symbol, Expr})
    # Note: never put a return in this... don't forget it's a macro!
    vn = gensym(:vn)
    
    return quote
        $vn = @varname($expr)
        
        # This branch should compile nicely in all cases except for partial missing data
        # For example, when `expr` is `x[i]` and `x isa Vector{Union{Missing, Float64}}`
        if !DynamicPPL.inargnames($vn, $model) || DynamicPPL.inmissings($vn, $model)
            true
        else
            if DynamicPPL.inargnames($vn, $model)
                # Evaluate the lhs
                $expr === missing
            else
                throw("This point should not be reached. Please report this error.")
            end
        end
    end |> esc
end

macro isassumption(model, expr)
    # failsafe: a literal is never an assumption
    false
end



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
    model_info = build_model_info(expr)

    # Replace macros in the function body.
    vi = gensym(:vi)
    sampler = gensym(:sampler)
    ex = replacemacro(model_info[:main_body],
                      Symbol("@varinfo") => vi,
                      Symbol("@logpdf") => :($(DynamicPPL.getlogp)($vi)),
                      Symbol("@sampler") => sampler)

    # Replace tildes in the function body.
    model = gensym(:model)
    context = gensym(:ctx)
    mainbody = replacetilde(ex, model, vi, sampler, context)

    model_info[:main_body] = mainbody

    return build_output(model_info, model, vi, sampler, context)
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
        args_nt = Expr(:call, :(DynamicPPL.namedtuple), nt_type, Expr(:tuple, arg_syms...))
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
    replacetilde(expr, model, vi, sampler, context)

Replace `~` and `.~` expressions in `expr` with observation or assumption expressions for
the given `model`, `vi` object, `sampler`, and `context`.
"""
function replacetilde(expr, model, vi, sampler, context)
    # Apply the `@.` macro first.
    dottedexpr = MacroTools.postwalk(apply_dotted, expr)

    # Check for tilde operators.
    return MacroTools.postwalk(dottedexpr) do x
        # Check dot tilde first.
        dotargs = getargs_dottilde(x)
        if dotargs !== nothing
            L, R = dotargs
            return generate_dot_tilde(L, R, model, vi, sampler, context)
        end

        # Check tilde.
        args = getargs_tilde(x)
        if args !== nothing
            L, R = args
            return generate_tilde(L, R, model, vi, sampler, context)
        end

        return x
    end
end

# """ Unbreak code highlighting in Emacs julia-mode


"""
    generate_tilde(left, right, model, vi, sampler, context)

Return the expression that replaces `left ~ right` in the function body for the provided
`model`, `vi` object, `sampler`, and `context`.
"""
function generate_tilde(left, right, model, vi, sampler, ctx)
    temp_right = gensym(:temp_right)
    out = gensym(:out)
    lp = gensym(:lp)
    vn = gensym(:vn)
    inds = gensym(:inds)
    isassumption = gensym(:isassumption)
    assert_ex = :(DynamicPPL.assert_dist($temp_right, msg = $(wrong_dist_errormsg(@__LINE__))))
    
    if left isa Symbol || left isa Expr
        ex = quote
            $temp_right = $right
            $assert_ex
            
            $vn, $inds = $(varname(left)), $(vinds(left))
            $isassumption = DynamicPPL.@isassumption($model, $left)
            if $isassumption 
                $out = DynamicPPL.tilde_assume($ctx, $sampler, $temp_right, $vn, $inds, $vi)
                $left = $out[1]
                DynamicPPL.acclogp!($vi, $out[2])
            else
                DynamicPPL.acclogp!(
                    $vi,
                    DynamicPPL.tilde_observe($ctx, $sampler, $temp_right, $left, $vn, $inds, $vi),
                )
            end
        end
    else
        # we have a literal, which is automatically an observation
        ex = quote
            $temp_right = $right
            $assert_ex
            
            DynamicPPL.acclogp!(
                $vi,
                DynamicPPL.tilde_observe($ctx, $sampler, $temp_right, $left, $vi),
            )
        end
    end
    return ex
end

"""
    generate_dot_tilde(left, right, model, vi, sampler, context)

Return the expression that replaces `left .~ right` in the function body for the provided
`model`, `vi` object, `sampler`, and `context`.
"""
function generate_dot_tilde(left, right, model, vi, sampler, ctx)
    out = gensym(:out)
    temp_right = gensym(:temp_right)
    isassumption = gensym(:isassumption)
    lp = gensym(:lp)
    vn = gensym(:vn)
    inds = gensym(:inds)
    assert_ex = :(DynamicPPL.assert_dist($temp_right, msg = $(wrong_dist_errormsg(@__LINE__))))
    
    if left isa Symbol || left isa Expr
        ex = quote
            $temp_right = $right
            $assert_ex

            $vn, $inds = $(varname(left)), $(vinds(left))
            $isassumption = DynamicPPL.@isassumption($model, $left)
            
            if $isassumption
                $out = DynamicPPL.dot_tilde_assume($ctx, $sampler, $temp_right, $left, $vn, $inds, $vi)
                $left .= $out[1]
                DynamicPPL.acclogp!($vi, $out[2])
            else
                DynamicPPL.acclogp!(
                    $vi,
                    DynamicPPL.dot_tilde_observe($ctx, $sampler, $temp_right, $left, $vn, $inds, $vi),
                )
            end
        end
    else
        # we have a literal, which is automatically an observation
        ex = quote
            $temp_right = $right
            $assert_ex
            
            DynamicPPL.acclogp!(
                $vi,
                DynamicPPL.dot_tilde_observe($ctx, $sampler, $temp_right, $left, $vi),
            )
        end
    end
    return ex
end

const FloatOrArrayType = Type{<:Union{AbstractFloat, AbstractArray}}
hasmissing(T::Type{<:AbstractArray{TA}}) where {TA <: AbstractArray} = hasmissing(TA)
hasmissing(T::Type{<:AbstractArray{>:Missing}}) = true
hasmissing(T::Type) = false

"""
    build_output(model_info, model, vi, sampler, context)

Build the output expression.
"""
function build_output(model_info, model, vi, sampler, ctx)
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
        temp_var = gensym(:temp_var)
        varT = gensym(:varT)
        push!(unwrap_data_expr.args, quote
            local $var
            $temp_var = $model.args.$var
            $varT = typeof($temp_var)
            if $temp_var isa DynamicPPL.FloatOrArrayType
                $var = DynamicPPL.get_matching_type($sampler, $vi, $temp_var)
            elseif DynamicPPL.hasmissing($varT)
                $var = DynamicPPL.get_matching_type($sampler, $vi, $varT)($temp_var)
            else
                $var = $temp_var
            end
        end)
    end

    @gensym(evaluator, generator)
    generator_kw_form = isempty(args) ? () : (:($generator(;$(args...)) = $generator($(arg_syms...))),)
    model_gen_constructor = :(DynamicPPL.ModelGen{$(Tuple(arg_syms))}($generator, $defaults_nt))
    
    return quote
        function $evaluator(
            $model::Model,
            $vi::DynamicPPL.VarInfo,
            $sampler::DynamicPPL.AbstractSampler,
            $ctx::DynamicPPL.AbstractContext,
        )
            $unwrap_data_expr
            DynamicPPL.resetlogp!($vi)
            $main_body
        end
        

        $generator($(args...)) = DynamicPPL.Model($evaluator, $args_nt, $model_gen_constructor)
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
    get_matching_type(spl, vi, ::Type{T}) where {T}
Get the specialized version of type `T` for sampler `spl`. For example,
if `T === Float64` and `spl::Hamiltonian`, the matching type is `eltype(vi[spl])`.
"""
function get_matching_type end
