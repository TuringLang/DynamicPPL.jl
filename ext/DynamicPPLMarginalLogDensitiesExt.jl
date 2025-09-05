module DynamicPPLMarginalLogDensitiesExt

using DynamicPPL: DynamicPPL, LogDensityProblems, VarName
using MarginalLogDensities: MarginalLogDensities

_to_varname(n::Symbol) = VarName{n}()
_to_varname(n::VarName) = n

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
