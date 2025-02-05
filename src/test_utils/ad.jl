module AD

import ADTypes: AbstractADType
import DifferentiationInterface as DI
import ..DynamicPPL: DynamicPPL, Model, LogDensityFunction, VarInfo, AbstractVarInfo
import LogDensityProblems: logdensity, logdensity_and_gradient
import LogDensityProblemsAD: ADgradient
import Random: Random, AbstractRNG
import Test: @test

export make_function, make_params, ad_ldp, ad_di, test_correctness

"""
    flipped_logdensity(θ, ldf)

Flips the order of arguments for `logdensity` to match the signature needed
for DifferentiationInterface.jl.
"""
flipped_logdensity(θ, ldf) = logdensity(ldf, θ)

"""
    ad_ldp(
        model::Model,
        params::Vector{<:Real},
        adtype::AbstractADType,
        varinfo::AbstractVarInfo=VarInfo(model)
    )

Calculate the logdensity of `model` and its gradient using the AD backend
`adtype`, evaluated at the parameters `params`, using the implementation of
`logdensity_and_gradient` in the LogDensityProblemsAD.jl package.

The `varinfo` argument is optional and is used to provide the container
structure for the parameters. Note that the _parameters_ inside the `varinfo`
argument itself are overridden by the `params` argument. This argument defaults
to [`DynamicPPL.VarInfo`](@ref), which is the default container structure used
throughout the Turing ecosystem; however, you can provide e.g.
[`DynamicPPL.SimpleVarInfo`](@ref) if you want to use a different container
structure.

Returns a tuple `(value, gradient)` where `value <: Real` is the logdensity
of the model evaluated at `params`, and `gradient <: Vector{<:Real}` is the
gradient of the logdensity with respect to `params`.

Note that DynamicPPL.jl and Turing.jl currently use LogDensityProblemsAD.jl
throughout, and hence this function most closely mimics the usage of AD within
the Turing ecosystem.

For some AD backends such as Mooncake.jl, LogDensityProblemsAD.jl simply defers
to the DifferentiationInterface.jl package. In such a case, `ad_ldp` simplifies
to `ad_di` (in that if `ad_di` passes, one should expect `ad_ldp` to pass as
well).

However, there are other AD backends which still have custom code in
LogDensityProblemsAD.jl (such as ForwardDiff.jl). For these backends, `ad_di`
may yield different results compared to `ad_ldp`, and the behaviour of `ad_di`
is in such cases not guaranteed to be consistent with the behaviour of
Turing.jl.

See also: [`ad_di`](@ref).
"""
function ad_ldp(
    model::Model,
    params::Vector{<:Real},
    adtype::AbstractADType,
    vi::AbstractVarInfo=VarInfo(model),
)
    ldf = LogDensityFunction(model, vi)
    # Note that the implementation of logdensity takes care of setting the 
    # parameters in vi to the correct values (using unflatten)
    return logdensity_and_gradient(ADgradient(adtype, ldf), params)
end

"""
    ad_di(
        model::Model,
        params::Vector{<:Real},
        adtype::AbstractADType,
        varinfo::AbstractVarInfo=VarInfo(model)
    )

Calculate the logdensity of `model` and its gradient using the AD backend
`adtype`, evaluated at the parameters `params`, directly using
DifferentiationInterface.jl.

See the notes in [`ad_ldp`](@ref) for more details on the differences between
`ad_di` and `ad_ldp`.
"""
function ad_di(
    model::Model,
    params::Vector{<:Real},
    adtype::AbstractADType,
    vi::AbstractVarInfo=VarInfo(model),
)
    ldf = LogDensityFunction(model, vi)
    # Note that the implementation of logdensity takes care of setting the 
    # parameters in vi to the correct values (using unflatten)
    prep = DI.prepare_gradient(flipped_logdensity, adtype, params, DI.Constant(ldf))
    return DI.value_and_gradient(flipped_logdensity, prep, adtype, params, DI.Constant(ldf))
end

"""
    make_function(model, varinfo::AbstractVarInfo=VarInfo(model))

Generate the function to be differentiated. Specifically,
`make_function(model)` returns a function which takes a single argument
`params` and returns the logdensity of `model` evaluated at `params`.

The `varinfo` parameter is optional and is used to determine the structure of
the varinfo used during evaluation. See the [`ad_ldp`](@ref) function for more
details on the `varinfo` argument.

If you have an AD package that does not have integrations with either
LogDensityProblemsAD.jl (in which case you can use [`ad_ldp`](@ref)) or
DifferentiationInterface.jl (in which case you can use [`ad_di`](@ref)), you
can test whether your AD package works with Turing.jl models using:

```julia
f = make_function(model)
params = make_params(model)
value, grad = YourADPackage.gradient(f, params)
```

and compare the results against that obtained from either `ad_ldp` or `ad_di` for
an existing AD package that _is_ supported.

See also: [`make_params`](@ref).
"""
function make_function(model::Model, vi::AbstractVarInfo=VarInfo(model))
    # TODO: Can we simplify this even further by inlining the definition of
    # logdensity?
    return Base.Fix1(logdensity, LogDensityFunction(model, vi))
end

"""
    make_params(model, rng::Random.AbstractRNG=Random.default_rng())

Generate a vector of parameters sampled from the prior distribution of `model`.
This can be used as the input to the function to be differentiated. See
[`make_function`](@ref) for more details.
"""
function make_params(model::Model, rng::AbstractRNG=Random.default_rng())
    return VarInfo(rng, model)[:]
end

"""
    test_correctness(
        ad_function,
        model::Model,
        adtypes::Vector{<:ADTypes.AbstractADType},
        reference_adtype::ADTypes.AbstractADType,
        rng::Random.AbstractRNG=Random.default_rng(),
        params::Vector{<:Real}=VarInfo(rng, model)[:];
        value_atol=1e-6,
        grad_atol=1e-6
    )

Test the correctness of all the AD backend `adtypes` for the model `model`
using the implementation `ad_function`. `ad_function` should be either
[`ad_ldp`](@ref) or [`ad_di`](@ref), or a custom function that has the same
signature.

The test is performed by calculating the logdensity and its gradient using all
the AD backends, and comparing the results against that obtained with the
reference AD backend `reference_adtype`.

The parameters can either be passed explicitly using the `params` argument, or can
be sampled from the prior distribution of the model using the `rng` argument.
"""
function test_correctness(
    ad_function,
    model::Model,
    adtypes::Vector{<:AbstractADType},
    reference_adtype::AbstractADType,
    rng::AbstractRNG=Random.default_rng(),
    params::Vector{<:Real}=VarInfo(rng, model)[:];
    value_atol=1e-6,
    grad_atol=1e-6,
)
    value_true, grad_true = ad_function(model, params, reference_adtype)
    for adtype in adtypes
        value, grad = ad_function(model, params, adtype)
        info_str = join(
            [
                "Testing AD correctness",
                " AD function : $(ad_function)",
                "     backend : $(adtype)",
                "       model : $(model.f)",
                "      params : $(params)",
                "      actual : $((value, grad))",
                "    expected : $((value_true, grad_true))",
            ],
            "\n",
        )
        @info info_str
        @test value ≈ value_true atol = value_atol
        @test grad ≈ grad_true atol = grad_atol
    end
end

end # module DynamicPPL.TestUtils.AD
