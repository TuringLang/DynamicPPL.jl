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
macro model(input_expr)
    build_model_info(input_expr) |> replace_tilde! |> replace_vi! |> 
        replace_logpdf! |> replace_sampler! |> build_output
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
    replace_vi!(model_info)

Replaces `@varinfo()` expressions with a handle to the `VarInfo` struct.
"""
function replace_vi!(model_info)
    ex = model_info[:main_body]
    vi = model_info[:main_body_names][:vi]
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @varinfo())
            vi
        else
            x
        end
    end
    model_info[:main_body] = ex
    return model_info
end

"""
    replace_logpdf!(model_info)

Replaces `@logpdf()` expressions with the value of the accumulated `logpdf` in the `VarInfo` struct.
"""
function replace_logpdf!(model_info)
    ex = model_info[:main_body]
    vi = model_info[:main_body_names][:vi]
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @logpdf())
            :($vi.logp[])
        else
            x
        end
    end
    model_info[:main_body] = ex
    return model_info
end

"""
    replace_sampler!(model_info)

Replaces `@sampler()` expressions with a handle to the sampler struct.
"""
function replace_sampler!(model_info)
    ex = model_info[:main_body]
    spl = model_info[:main_body_names][:sampler]
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @sampler())
            spl
        else
            x
        end
    end
    model_info[:main_body] = ex
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

    # Check for tilde operators.
    tildeexpr = MacroTools.postwalk(dottedexpr) do x
        # Check dot tilde first.
        dotargs = getargs_dottilde(x)
        if dotargs !== nothing
            L, R = dotargs
            return generate_dot_tilde(L, R, model_info)
        end

        # Check tilde.
        args = getargs_tilde(x)
        if args !== nothing
            L, R = args
            return generate_tilde(L, R, model_info)
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
    
    ex = quote
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

    return esc(ex)
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
