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
    nt = DynamicPPL.tonamedtuple(var_info)
    for (k, (vals, names)) in pairs(nt)
        for (n, v) in zip(names, vals)
            chain_val = if Symbol(n) ∉ keys(chain)
                # Assume it's a group
                vec(MCMCChains.group(chain, Symbol(n)).value[sample_idx, :, chain_idx])
            else
                chain[sample_idx, n, chain_idx]
            end
            @test v == chain_val
        end
    end
end

"""
    short_varinfo_name(vi::AbstractVarInfo)

Return string representing a short description of `vi`.
"""
short_varinfo_name(vi::DynamicPPL.ThreadSafeVarInfo) = short_varinfo_name(vi.varinfo)
short_varinfo_name(::TypedVarInfo) = "TypedVarInfo"
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

# helper functions for testing (both varname_leaves and values_from_chain are in utils.jl, for some reason they are not loaded)
varname_leaves(vn::VarName, ::Real) = [vn]
function varname_leaves(vn::VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (
        VarName(vn, getlens(vn) ∘ Setfield.IndexLens(Tuple(I))) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_leaves(VarName(vn, getlens(vn) ∘ Setfield.IndexLens(Tuple(I))), val[I]) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::DynamicPPL.VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        lens = Setfield.PropertyLens{sym}()
        varname_leaves(vn ∘ lens, get(val, lens))
    end
    return Iterators.flatten(iter)
end

# values_from_chain
# TODO: remove after PR#481 is merged
function values_from_chain(x, vn_parent, chain, chain_idx, iteration_idx)
    # HACK: If it's not an array, we fall back to just returning the first value.
    return only(chain[iteration_idx, Symbol(vn_parent), chain_idx])
end
function values_from_chain(
    x::AbstractArray, vn_parent::VarName{sym}, chain, chain_idx, iteration_idx
) where {sym}
    # We use `VarName{sym}()` so that the resulting leaf `vn` only contains the tail of the lens.
    # This way we can use `getlens(vn)` to extract the value from `x` and use `vn_parent ∘ getlens(vn)`
    # to extract the value from the `chain`.
    return reduce(varname_leaves(VarName{sym}(), x); init=similar(x)) do x, vn
        # Update `x`, possibly in place, and return.
        l = AbstractPPL.getlens(vn)
        Setfield.set(
            x,
            BangBang.prefermutation(l),
            chain[iteration_idx, Symbol(vn_parent ∘ l), chain_idx],
        )
    end
end
function values_from_chain(vi::AbstractVarInfo, vn_parent, chain, chain_idx, iteration_idx)
    # Use the value `vi[vn_parent]` to obtain a buffer.
    return values_from_chain(vi[vn_parent], vn_parent, chain, chain_idx, iteration_idx)
end