module DynamicPPLMarginalLogDensitiesExt

using DynamicPPL: DynamicPPL, LogDensityProblems, VarName
using MarginalLogDensities: MarginalLogDensities

_to_varname(n::Symbol) = VarName{n}()
_to_varname(n::VarName) = n

"""
    marginalize(
        model::DynamicPPL.Model,
        varnames::AbstractVector{<:Union{Symbol,<:VarName}},
        getlogprob=DynamicPPL.getlogjoint,
        method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox();
        kwargs...,
    )

Construct a `MarginalLogDensities.MarginalLogDensity` object that represents the marginal
log-density of the given `model`, after marginalizing out the variables specified in
`varnames`.

The resulting object can be called with a vector of parameter values to compute the marginal
log-density.

The `getlogprob` argument can be used to specify which kind of marginal log-density to
compute. Its default value is `DynamicPPL.getlogjoint` which returns the marginal log-joint
probability.

By default the marginalization is performed with a Laplace approximation. Please see [the
MarginalLogDensities.jl package](https://github.com/ElOceanografo/MarginalLogDensities.jl/)
for other options.

## Example

```jldoctest
julia> using DynamicPPL, Distributions, MarginalLogDensities

julia> @model function demo()
           x ~ Normal(1.0)
           y ~ Normal(2.0)
       end
demo (generic function with 2 methods)

julia> marginalized = marginalize(demo(), [:x]);

julia> # The resulting callable computes the marginal log-density of `y`.
       marginalized([1.0])
-1.4189385332046727

julia> logpdf(Normal(2.0), 1.0)
-1.4189385332046727
```
"""
function DynamicPPL.marginalize(
    model::DynamicPPL.Model,
    varnames::AbstractVector{<:Union{Symbol,<:VarName}},
    getlogprob=DynamicPPL.getlogjoint,
    method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox();
    kwargs...,
)
    # Determine the indices for the variables to marginalise out.
    varinfo = DynamicPPL.typed_varinfo(model)
    vns = map(_to_varname, varnames)
    varindices = reduce(vcat, DynamicPPL.vector_getranges(varinfo, vns))
    # Construct the marginal log-density model.
    # Use linked `varinfo` to that we're working in unconstrained space
    varinfo_linked = DynamicPPL.link(varinfo, model)

    f = DynamicPPL.LogDensityFunction(model, getlogprob, varinfo_linked)
    mdl = MarginalLogDensities.MarginalLogDensity(
        (x, _) -> LogDensityProblems.logdensity(f, x),
        varinfo_linked[:],
        varindices,
        (),
        method;
        kwargs...,
    )
    return mdl
end

end
