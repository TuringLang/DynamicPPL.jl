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
    test_setval!(model, chain; sample_idx = 1, chain_idx = 1)

Test `setval!` on `model` and `chain`.

Worth noting that this only supports models containing symbols of the forms
`m`, `m[1]`, `m[1, 2]`, not `m[1][1]`, etc.
"""
function test_setval!(model, chain; sample_idx=1, chain_idx=1)
    var_info = VarInfo(model)
    spl = SampleFromPrior()
    θ_old = var_info[spl]
    DynamicPPL.setval!(var_info, chain, sample_idx, chain_idx)
    θ_new = var_info[spl]
    @test θ_old != θ_new
    vals = DynamicPPL.values_as(var_info, OrderedDict)
    iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))
    for (n, v) in mapreduce(collect, vcat, iters)
        n = string(n)
        if Symbol(n) ∉ keys(chain)
            # Assume it's a group
            chain_val = vec(
                MCMCChains.group(chain, Symbol(n)).value[sample_idx, :, chain_idx]
            )
            v_true = vec(v)
        else
            chain_val = chain[sample_idx, n, chain_idx]
            v_true = v
        end

        @test v_true == chain_val
    end
end

"""
    short_varinfo_name(vi::AbstractVarInfo)

Return string representing a short description of `vi`.
"""
short_varinfo_name(vi::DynamicPPL.ThreadSafeVarInfo) =
    "threadsafe($(short_varinfo_name(vi.varinfo)))"
function short_varinfo_name(vi::TypedVarInfo)
    DynamicPPL.has_varnamevector(vi) && return "TypedVarInfo with VarNameVector"
    return "TypedVarInfo"
end
short_varinfo_name(::UntypedVarInfo) = "UntypedVarInfo"
short_varinfo_name(::SimpleVarInfo{<:NamedTuple}) = "SimpleVarInfo{<:NamedTuple}"
short_varinfo_name(::SimpleVarInfo{<:OrderedDict}) = "SimpleVarInfo{<:OrderedDict}"

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
