# default model
@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
const gdemo_default = gdemo_d()

function test_model_ad(model, logp_manual)
    vi = VarInfo(model)
    x = DynamicPPL.getall(vi)

    # Log probabilities using the model.
    ℓ = DynamicPPL.LogDensityFunction(model, vi)
    logp_model = Base.Fix1(LogDensityProblems.logdensity, ℓ)

    # Check that both functions return the same values.
    lp = logp_manual(x)
    @test logp_model(x) ≈ lp

    # Gradients based on the manual implementation.
    grad = ForwardDiff.gradient(logp_manual, x)

    y, back = Tracker.forward(logp_manual, x)
    @test Tracker.data(y) ≈ lp
    @test Tracker.data(back(1)[1]) ≈ grad

    y, back = Zygote.pullback(logp_manual, x)
    @test y ≈ lp
    @test back(1)[1] ≈ grad

    # Gradients based on the model.
    @test ForwardDiff.gradient(logp_model, x) ≈ grad

    y, back = Tracker.forward(logp_model, x)
    @test Tracker.data(y) ≈ lp
    @test Tracker.data(back(1)[1]) ≈ grad

    y, back = Zygote.pullback(logp_model, x)
    @test y ≈ lp
    @test back(1)[1] ≈ grad
end

"""
    short_varinfo_name(vi::AbstractVarInfo)

Return string representing a short description of `vi`.
"""
short_varinfo_name(vi::DynamicPPL.ThreadSafeVarInfo) =
    "threadsafe($(short_varinfo_name(vi.varinfo)))"
function short_varinfo_name(vi::TypedVarInfo)
    DynamicPPL.has_varnamedvector(vi) && return "TypedVarInfo with VarNamedVector"
    return "TypedVarInfo"
end
short_varinfo_name(::UntypedVarInfo) = "UntypedVarInfo"
short_varinfo_name(::DynamicPPL.VectorVarInfo) = "VectorVarInfo"
function short_varinfo_name(::SimpleVarInfo{<:NamedTuple,<:Ref})
    return "SimpleVarInfo{<:NamedTuple,<:Ref}"
end
function short_varinfo_name(::SimpleVarInfo{<:OrderedDict,<:Ref})
    return "SimpleVarInfo{<:OrderedDict,<:Ref}"
end
function short_varinfo_name(::SimpleVarInfo{<:DynamicPPL.VarNamedVector,<:Ref})
    return "SimpleVarInfo{<:VarNamedVector,<:Ref}"
end
short_varinfo_name(::SimpleVarInfo{<:NamedTuple}) = "SimpleVarInfo{<:NamedTuple}"
short_varinfo_name(::SimpleVarInfo{<:OrderedDict}) = "SimpleVarInfo{<:OrderedDict}"
function short_varinfo_name(::SimpleVarInfo{<:DynamicPPL.VarNamedVector})
    return "SimpleVarInfo{<:VarNamedVector}"
end

# convenient functions for testing model.jl
# function to modify the representation of values based on their length
function modify_value_representation(nt::NamedTuple)
    modified_nt = NamedTuple()
    for (key, value) in zip(keys(nt), values(nt))
        if length(value) == 1  # Scalar value
            modified_value = value[1]
        else  # Non-scalar value
            modified_value = value
        end
        modified_nt = merge(modified_nt, (key => modified_value,))
    end
    return modified_nt
end

"""
    make_chain_from_prior([rng,] model, n_iters)

Construct an MCMCChains.Chains object by sampling from the prior of `model` for
`n_iters` iterations.
"""
function make_chain_from_prior(rng::Random.AbstractRNG, model::Model, n_iters::Int)
    # Sample from the prior
    varinfos = [VarInfo(rng, model) for _ in 1:n_iters]
    # Extract all varnames found in any dictionary. Doing it this way guards
    # against the possibility of having different varnames in different
    # dictionaries, e.g. for models that have dynamic variables / array sizes
    varnames = OrderedSet{VarName}()
    # Convert each varinfo into an OrderedDict of vns => params.
    # We have to use varname_and_value_leaves so that each parameter is a scalar
    dicts = map(varinfos) do t
        vals = DynamicPPL.values_as(t, OrderedDict)
        iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))
        tuples = mapreduce(collect, vcat, iters)
        push!(varnames, map(first, tuples)...)
        OrderedDict(tuples)
    end
    # Convert back to list
    varnames = collect(varnames)
    # Construct matrix of values
    vals = [get(dict, vn, missing) for dict in dicts, vn in varnames]
    # Construct and return the Chains object
    return Chains(vals, varnames)
end
function make_chain_from_prior(model::Model, n_iters::Int)
    return make_chain_from_prior(Random.default_rng(), model, n_iters)
end
