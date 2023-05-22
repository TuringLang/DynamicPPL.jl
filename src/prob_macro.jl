macro logprob_str(str)
    expr1, expr2 = get_exprs(str)
    return :(logprob($(esc(expr1)), $(esc(expr2))))
end
macro prob_str(str)
    expr1, expr2 = get_exprs(str)
    return :(exp.(logprob($(esc(expr1)), $(esc(expr2)))))
end

function get_exprs(str::String)
    substrings = split(str, '|'; limit=2)
    length(substrings) == 2 || error("Invalid expression.")
    str1, str2 = substrings

    expr1 = Meta.parse("($str1,)")
    expr1 = Expr(:tuple, expr1.args...)

    expr2 = Meta.parse("($str2,)")
    expr2 = Expr(:tuple, expr2.args...)

    return expr1, expr2
end

function logprob(ex1, ex2)
    ptype, model, vi = probtype(ex1, ex2)
    if ptype isa Val{:prior}
        return logprior(ex1, ex2, model, vi)
    elseif ptype isa Val{:likelihood}
        return loglikelihood(ex1, ex2, model, vi)
    end
end

function probtype(ntl::NamedTuple{namesl}, ntr::NamedTuple{namesr}) where {namesl,namesr}
    if :chain in namesr
        if isdefined(ntr.chain.info, :model)
            model = ntr.chain.info.model
        elseif isdefined(ntr, :model)
            model = ntr.model
        else
            throw(
                "The model is not defined. Please make sure the model is either saved in the chain or passed on the RHS of |.",
            )
        end
        @assert model isa Model
        if isdefined(ntr.chain.info, :vi)
            _vi = ntr.chain.info.vi
            @assert _vi isa VarInfo
            vi = TypedVarInfo(_vi)
        elseif isdefined(ntr, :varinfo)
            _vi = ntr.varinfo
            @assert _vi isa VarInfo
            vi = TypedVarInfo(_vi)
        else
            vi = nothing
        end
        defaults = model.defaults
        @assert all(getargnames(model)) do arg
            isdefined(ntl, arg) ||
                isdefined(ntr, arg) ||
                isdefined(defaults, arg) && getfield(defaults, arg) !== missing
        end
        return Val(:likelihood), model, vi
    else
        @assert isdefined(ntr, :model)
        model = ntr.model
        @assert model isa Model
        if isdefined(ntr, :varinfo)
            _vi = ntr.varinfo
            @assert _vi isa VarInfo
            vi = TypedVarInfo(_vi)
        else
            vi = nothing
        end
        return probtype(ntl, ntr, model), model, vi
    end
end

function probtype(
    left::NamedTuple{leftnames},
    right::NamedTuple{rightnames},
    model::Model{_F,argnames,defaultnames},
) where {leftnames,rightnames,argnames,defaultnames,_F}
    defaults = model.defaults
    prior_rhs = all(
        n -> n in (:model, :varinfo) || n in argnames && getfield(right, n) !== missing,
        rightnames,
    )
    function get_arg(arg)
        if arg in leftnames
            return getfield(left, arg)
        elseif arg in rightnames
            return getfield(right, arg)
        elseif arg in defaultnames
            return getfield(defaults, arg)
        elseif arg in argnames
            return getfield(model.args, arg)
        else
            return nothing
        end
    end
    function valid_arg(arg)
        a = get_arg(arg)
        return a !== nothing && a !== missing
    end
    valid_args = all(valid_arg, argnames)

    # Uses the default values for model arguments not provided.
    # If no default value exists, use `nothing`.
    if prior_rhs
        return Val(:prior)
        # Uses the default values for model arguments not provided.
        # If no default value exists or the default value is missing, then error.
    elseif valid_args
        return Val(:likelihood)
    else
        for argname in argnames
            if !valid_arg(argname)
                throw(ArgumentError(missing_arg_error_msg(argname, get_arg(argname))))
            end
        end
    end
end

function missing_arg_error_msg(arg, ::Missing)
    return """Variable $arg has a value of `missing`, or is not defined and its default value is `missing`. Please make sure all the variables are either defined with a value other than `missing` or have a default value other than `missing`."""
end
function missing_arg_error_msg(arg, ::Nothing)
    return """Variable $arg is not defined and has no default value. Please make sure all the variables are either defined with a value other than `missing` or have a default value other than `missing`."""
end

function logprior(
    left::NamedTuple, right::NamedTuple, _model::Model, _vi::Union{Nothing,VarInfo}
)
    # For model args on the LHS of |, use their passed value but add the symbol to 
    # model.missings. This will lead to an `assume`/`dot_assume` call for those variables.
    # Let `p::PriorContext`. If `p.vars` is `nothing`, `assume` and `dot_assume` will use 
    # the values of the random variables in the `VarInfo`. If `p.vars` is a `NamedTuple` 
    # or a `Chain`, the value in `p.vars` is input into the `VarInfo` and used instead.

    # For model args not on the LHS of |, if they have a default value, use that, 
    # otherwise use `nothing`. This will lead to an `observe`/`dot_observe`call for 
    # those variables.
    # All `observe` and `dot_observe` calls are no-op in the PriorContext

    # When all of model args are on the lhs of |, this is also equal to the logjoint.
    model = make_prior_model(left, right, _model)
    vi = _vi === nothing ? VarInfo(deepcopy(model), PriorContext()) : _vi
    foreach(keys(vi.metadata)) do n
        @assert n in keys(left) "Variable $n is not defined."
    end
    return getlogp(
        last(DynamicPPL.evaluate!!(model, vi, SampleFromPrior(), PriorContext(left)))
    )
end

@generated function make_prior_model(
    left::NamedTuple{leftnames},
    right::NamedTuple{rightnames},
    model::Model{_F,argnames,defaultnames},
) where {leftnames,rightnames,argnames,defaultnames,_F}
    argvals = []
    missings = []
    warnings = []

    for argname in argnames
        if argname in leftnames
            push!(argvals, :(deepcopy(left.$argname)))
            push!(missings, argname)
        elseif argname in rightnames
            push!(argvals, :(right.$argname))
        elseif argname in defaultnames
            push!(argvals, :(model.defaults.$argname))
        else
            push!(warnings, :(@warn($(warn_msg(argname)))))
            push!(argvals, :(model.args.$argname))
        end
    end

    # `args` is inserted as properly typed NamedTuple expression;
    # `missings` is splatted into a tuple at compile time and inserted as literal
    return quote
        $(warnings...)
        Model{$(Tuple(missings))}(
            model.f, $(to_namedtuple_expr(argnames, argvals)), model.defaults
        )
    end
end

warn_msg(arg) = "Argument $arg is not defined. Using the value from the model."

function Distributions.loglikelihood(
    left::NamedTuple, right::NamedTuple, _model::Model, _vi::Union{Nothing,VarInfo}
)
    model = make_likelihood_model(left, right, _model)
    vi = _vi === nothing ? VarInfo(deepcopy(model)) : _vi
    if isdefined(right, :chain)
        # Element-wise likelihood for each value in chain
        chain = right.chain
        ctx = LikelihoodContext(right)
        iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
        logps = map(iters) do (sample_idx, chain_idx)
            setval!(vi, chain, sample_idx, chain_idx)
            model(vi, SampleFromPrior(), ctx)
            return getlogp(vi)
        end
        return reshape(logps, size(chain, 1), size(chain, 3))
    else
        # Likelihood without chain
        # Rhs values are used in the context
        ctx = LikelihoodContext(right)
        model(vi, SampleFromPrior(), ctx)
        return getlogp(vi)
    end
end

@generated function make_likelihood_model(
    left::NamedTuple{leftnames},
    right::NamedTuple{rightnames},
    model::Model{_F,argnames,defaultnames},
) where {leftnames,rightnames,argnames,defaultnames,_F}
    argvals = []
    missings = []

    for argname in argnames
        if argname in leftnames
            push!(argvals, :(left.$argname))
        elseif argname in rightnames
            push!(argvals, :(right.$argname))
            push!(missings, argname)
        elseif argname in defaultnames
            push!(argvals, :(model.defaults.$argname))
        elseif argname in argnames
            push!(argvals, :(model.args.$argname))
        else
            throw(
                "This point should not be reached. Please open an issue in the DynamicPPL.jl repository.",
            )
        end
    end

    # `args` is inserted as properly typed NamedTuple expression;
    # `missings` is splatted into a tuple at compile time and inserted as literal
    return :(Model{$(Tuple(missings))}(
        model.f, $(to_namedtuple_expr(argnames, argvals)), model.defaults
    ))
end
