module DynamicPPLMarginalLogDensitiesExt

using DynamicPPL: DynamicPPL, LogDensityProblems, VarName, RangeAndLinked
using MarginalLogDensities: MarginalLogDensities

# Make LogDensityFunction directly callable with the two-argument interface expected by
# MarginalLogDensities. The second argument is the gradient and is unused here because
# MarginalLogDensities handles differentiation separately.
function (ldf::DynamicPPL.LogDensityFunction)(x, _)
    return LogDensityProblems.logdensity(ldf, x)
end

"""
    marginalize(
        model::DynamicPPL.Model,
        marginalized_varnames::AbstractVector{<:VarName};
        varinfo::DynamicPPL.AbstractVarInfo=link(VarInfo(model), model),
        getlogprob=DynamicPPL.getlogjoint,
        method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox();
        kwargs...,
    )

Construct a `MarginalLogDensities.MarginalLogDensity` object that represents the marginal
log-density of the given `model`, after marginalizing out the variables specified in
`varnames`.

The resulting object can be called with a vector of parameter values to compute the marginal
log-density.

## Keyword arguments

- `varinfo`: The `varinfo` to use for the model. By default we use a linked `VarInfo`,
   meaning that the resulting log-density function accepts parameters that have been
   transformed to unconstrained space.

- `getlogprob`: A function which specifies which kind of marginal log-density to compute.
   Its default value is `DynamicPPL.getlogjoint` which returns the marginal log-joint
   probability.

- `method`: The marginalization method; defaults to a Laplace approximation. Please see [the
   MarginalLogDensities.jl package](https://github.com/ElOceanografo/MarginalLogDensities.jl/)
   for other options.

- Other keyword arguments are passed to the `MarginalLogDensities.MarginalLogDensity`
  constructor.

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


!!! warning

    The default usage of linked VarInfo means that, for example, optimization of the
    marginal log-density can be performed in unconstrained space. However, care must be
    taken if the model contains variables where the link transformation depends on a
    marginalized variable. For example:
    ```julia
        @model function f()
            x ~ Normal()
            y ~ truncated(Normal(); lower=x)
        end
    ```

    Here, the support of `y`, and hence the link transformation used, depends on the value
    of `x`. If we now marginalize over `x`, we obtain a function mapping linked values of
    `y` to log-probabilities. However, it will not be possible to use DynamicPPL to
    correctly retrieve _unlinked_ values of `y`.
"""
function DynamicPPL.marginalize(
    model::DynamicPPL.Model,
    marginalized_varnames::AbstractVector{<:VarName};
    varinfo::DynamicPPL.AbstractVarInfo=DynamicPPL.link(DynamicPPL.VarInfo(model), model),
    getlogprob::Function=DynamicPPL.getlogjoint,
    method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox(),
    kwargs...,
)
    # Construct the log-density function directly from the model and varinfo.
    ldf = DynamicPPL.LogDensityFunction(model, getlogprob, varinfo)
    # Determine the indices for the variables to marginalise out.
    varindices = mapreduce(vcat, marginalized_varnames) do vn
        # The type assertion helps in cases where the model is type unstable and thus
        # `varname_ranges` may have an abstract element type.
        (ldf._varname_ranges[vn]::RangeAndLinked).range
    end
    mld = MarginalLogDensities.MarginalLogDensity(
        ldf, varinfo[:], varindices, (), method; kwargs...
    )
    return mld
end

"""
    InitFromVector(
        mld::MarginalLogDensities.MarginalLogDensity{<:DynamicPPL.LogDensityFunction},
        unmarginalized_params::Union{AbstractVector,Nothing}=nothing
    )

Return an [`InitFromVector`](@ref DynamicPPL.InitFromVector) initialisation strategy whose
parameter vector reflects the state of `mld`.

If a Laplace approximation was used for marginalisation, the marginalized parameters are set
to their modal values (note that this requires `mld` to have been evaluated at least once,
so that the mode has been found).

If `unmarginalized_params` is provided, those values are used for the non-marginalized
parameters. This vector may be obtained e.g. by optimizing the marginal log-density.

To obtain a fully consistent `VarInfo` — with updated log-probabilities and correct link
status — use the returned strategy to re-evaluate the model:
```julia
init_strategy = DynamicPPL.InitFromVector(mld, opt_solution.u)
ldf = mld.logdensity
_, vi = DynamicPPL.init!!(ldf.model, DynamicPPL.VarInfo(), init_strategy, ldf.transform_strategy)
```
"""
function DynamicPPL.InitFromVector(
    mld::MarginalLogDensities.MarginalLogDensity{<:DynamicPPL.LogDensityFunction},
    unmarginalized_params::Union{AbstractVector,Nothing}=nothing,
)
    # Retrieve the full cached parameter vector (includes modal values for marginalized
    # parameters if a Laplace approximation has been run).
    full_params = MarginalLogDensities.cached_params(mld)
    # Overwrite the non-marginalized entries if the caller supplied them.
    if unmarginalized_params !== nothing
        full_params[MarginalLogDensities.ijoint(mld)] = unmarginalized_params
    end
    # Use the convenience constructor that reads varname_ranges and transform_strategy
    # directly from the LogDensityFunction stored inside mld.
    return DynamicPPL.InitFromVector(full_params, mld.logdensity)
end

end
