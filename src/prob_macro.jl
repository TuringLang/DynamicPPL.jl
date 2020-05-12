macro logprob_str(str)
    expr1, expr2 = get_exprs(str)
    return :(logprob($(esc(expr1)), $(esc(expr2))))
end
macro prob_str(str)
    expr1, expr2 = get_exprs(str)
    return :(exp.(logprob($(esc(expr1)), $(esc(expr2)))))
end

function get_exprs(str::String)
    substrings = split(str, '|'; limit = 2)
    length(substrings) == 2 || error("Invalid expression.")
    str1, str2 = substrings

    expr1 = Meta.parse("($str1,)")
    expr1 = Expr(:tuple, expr1.args...)

    expr2 = Meta.parse("($str2,)")
    expr2 = Expr(:tuple, expr2.args...)

    return expr1, expr2
end

function logprob(ex1, ex2)
    ptype, modelgen, vi = probtype(ex1, ex2)
    if ptype isa Val{:prior}
        return logprior(ex1, ex2, modelgen, vi)
    elseif ptype isa Val{:likelihood}
        return loglikelihood(ex1, ex2, modelgen, vi)
    end
end

function probtype(ntl::NamedTuple{namesl}, ntr::NamedTuple{namesr}) where {namesl, namesr}
    if :chain in namesr
        if isdefined(ntr.chain.info, :model)
            model = ntr.chain.info.model
            @assert model isa Model
            modelgen = getgenerator(model)
        elseif isdefined(ntr, :model)
            modelgen = ntr.model
        else
            throw("The model is not defined. Please make sure the model is either saved in the chain or passed on the RHS of |.")
        end
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
        defaults = getdefaults(modelgen)
        valid_arg(arg) = isdefined(ntl, arg) || isdefined(ntr, arg) || 
            isdefined(defaults, arg) && getfield(defaults, arg) !== missing
        @assert all(valid_arg, getargnames(modelgen))
        return Val(:likelihood), modelgen, vi
    else
        @assert isdefined(ntr, :model)
        modelgen = ntr.model
        if isdefined(ntr, :varinfo)
            _vi = ntr.varinfo
            @assert _vi isa VarInfo
            vi = TypedVarInfo(_vi)
        else
            vi = nothing
        end
        return probtype(ntl, ntr, modelgen), modelgen, vi
    end
end
function probtype(
    left::NamedTuple{leftnames},
    right::NamedTuple{rightnames},
    modelgen::ModelGen{_G, argnames, defaultnames}
) where {leftnames, rightnames, argnames, defaultnames, _G}
    defaults = getdefaults(modelgen)
    prior_rhs = all(n -> n in (:model, :varinfo) || 
        n in argnames && getfield(right, n) !== missing, rightnames)
    function get_arg(arg)
        if arg in leftnames
            return getfield(left, arg)
        elseif arg in rightnames
            return getfield(right, arg)
        elseif arg in defaultnames
            return getfield(defaults, arg)
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

missing_arg_error_msg(arg, ::Missing) = """Variable $arg has a value of `missing`, or is not defined and its default value is `missing`. Please make sure all the variables are either defined with a value other than `missing` or have a default value other than `missing`."""
missing_arg_error_msg(arg, ::Nothing) = """Variable $arg is not defined and has no default value. Please make sure all the variables are either defined with a value other than `missing` or have a default value other than `missing`."""

function logprior(
    left::NamedTuple,
    right::NamedTuple,
    modelgen::ModelGen,
    _vi::Union{Nothing, VarInfo}
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
    model = make_prior_model(left, right, modelgen)
    vi = _vi === nothing ? VarInfo(deepcopy(model), PriorContext()) : _vi
    foreach(keys(vi.metadata)) do n
        @assert n in keys(left) "Variable $n is not defined."
    end
    model(vi, SampleFromPrior(), PriorContext(left))
    return getlogp(vi)
end

@generated function make_prior_model(
    left::NamedTuple{leftnames},
    right::NamedTuple{rightnames},
    modelgen::ModelGen{_G, argnames, defaultnames}
) where {leftnames, rightnames, argnames, defaultnames, _G}
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
            push!(argvals, :(getdefaults(modelgen).$argname))
        else
            push!(warnings, :(@warn($(warn_msg(argname)))))
            push!(argvals, :(nothing))
        end
    end

    # `args` is inserted as properly typed NamedTuple expression; 
    # `missings` is splatted into a tuple at compile time and inserted as literal
    return quote
        $(warnings...)
        Model{$(Tuple(missings))}(modelgen, $(to_namedtuple_expr(argnames, argvals)))
    end
end

warn_msg(arg) = "Argument $arg is not defined. A value of `nothing` is used."

function Distributions.loglikelihood(
    left::NamedTuple,
    right::NamedTuple,
    modelgen::ModelGen,
    _vi::Union{Nothing, VarInfo},
)
    model = make_likelihood_model(left, right, modelgen)
    vi = _vi === nothing ? VarInfo(deepcopy(model)) : _vi
    if isdefined(right, :chain)
        # Element-wise likelihood for each value in chain
        chain = right.chain
        ctx = LikelihoodContext()
        return map(1:length(chain)) do i
            c = chain[i]
            _setval!(vi, c)
            model(vi, SampleFromPrior(), ctx)
            return getlogp(vi)
        end
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
    modelgen::ModelGen{_G, argnames, defaultnames}
) where {leftnames, rightnames, argnames, defaultnames, _G}
    argvals = []
    missings = []
    
    for argname in argnames
        if argname in leftnames
            push!(argvals, :(left.$argname))
        elseif argname in rightnames
            push!(argvals, :(right.$argname))
            push!(missings, argname)
        elseif argname in defaultnames
            push!(argvals, :(getdefaults(modelgen).$argname))
        else
            throw("This point should not be reached. Please open an issue in the DynamicPPL.jl repository.")
        end
    end

    # `args` is inserted as properly typed NamedTuple expression; 
    # `missings` is splatted into a tuple at compile time and inserted as literal
    return :(Model{$(Tuple(missings))}(modelgen, $(to_namedtuple_expr(argnames, argvals))))
end

_setval!(vi::TypedVarInfo, c::AbstractChains) = _setval!(vi.metadata, vi, c)
@generated function _setval!(md::NamedTuple{names}, vi, c) where {names}
    return Expr(:block, map(names) do n
        quote
            for vn in md.$n.vns
                val = copy.(vec(c[Symbol(string(vn))].value))
                setval!(vi, val, vn)
                settrans!(vi, false, vn)
            end
        end
    end...)
end
