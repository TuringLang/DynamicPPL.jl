# sampler.jl
# ----------
#
# Utilities to test samplers on models.

"""
    marginal_mean_of_samples(chain, varname)

Return the mean of variable represented by `varname` in `chain`.
"""
DynamicPPL.TestUtils.marginal_mean_of_samples(chain, varname) =
    mean(Array(chain[Symbol(varname)]))

"""
    test_sampler(models, sampler, args...; kwargs...)

Test that `sampler` produces correct marginal posterior means on each model in `models`.

In short, this method iterates through `models`, calls `AbstractMCMC.sample` on the
`model` and `sampler` to produce a `chain`, and then checks `marginal_mean_of_samples(chain, vn)`
for every (leaf) varname `vn` against the corresponding value returned by
[`posterior_mean`](@ref) for each model.

To change how comparison is done for a particular `chain` type, one can overload
[`marginal_mean_of_samples`](@ref) for the corresponding type.

# Arguments
- `models`: A collection of instaces of [`DynamicPPL.Model`](@ref) to test on.
- `sampler`: The `AbstractMCMC.AbstractSampler` to test.
- `args...`: Arguments forwarded to `sample`.

# Keyword arguments
- `varnames_filter`: A filter to apply to `varnames(model)`, allowing comparison for only
    a subset of the varnames.
- `atol=1e-1`: Absolute tolerance used in `@test`.
- `rtol=1e-3`: Relative tolerance used in `@test`.
- `kwargs...`: Keyword arguments forwarded to `sample`.
"""
function DynamicPPL.TestUtils.test_sampler(
    models,
    sampler::AbstractMCMC.AbstractSampler,
    args...;
    varnames_filter=Returns(true),
    atol=1e-1,
    rtol=1e-3,
    sampler_name=typeof(sampler),
    kwargs...,
)
    @testset "$(sampler_name) on $(nameof(model))" for model in models
        chain = AbstractMCMC.sample(model, sampler, args...; kwargs...)
        target_values = posterior_mean(model)
        for vn in filter(varnames_filter, varnames(model))
            # We want to compare elementwise which can be achieved by
            # extracting the leaves of the `VarName` and the corresponding value.
            for vn_leaf in DynamicPPL.varname_leaves(vn, get(target_values, vn))
                target_value = get(target_values, vn_leaf)
                chain_mean_value = marginal_mean_of_samples(chain, vn_leaf)
                @test chain_mean_value ≈ target_value atol = atol rtol = rtol
            end
        end
    end
end

"""
    test_sampler_on_demo_models(meanfunction, sampler, args...; kwargs...)

Test `sampler` on every model in [`DEMO_MODELS`](@ref).

This is just a proxy for `test_sampler(meanfunction, DEMO_MODELS, sampler, args...; kwargs...)`.
"""
function DynamicPPL.TestUtils.test_sampler_on_demo_models(
    sampler::AbstractMCMC.AbstractSampler, args...; kwargs...
)
    return test_sampler(DynamicPPL.TestUtils.DEMO_MODELS, sampler, args...; kwargs...)
end

"""
    test_sampler_continuous(sampler, args...; kwargs...)

Test that `sampler` produces the correct marginal posterior means on all models in `demo_models`.

As of right now, this is just an alias for [`test_sampler_on_demo_models`](@ref).
"""
function DynamicPPL.TestUtils.test_sampler_continuous(
    sampler::AbstractMCMC.AbstractSampler, args...; kwargs...
)
    return test_sampler_on_demo_models(sampler, args...; kwargs...)
end
