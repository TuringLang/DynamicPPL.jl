module DynamicPPLMarginalLogDensitiesExt

using DynamicPPL: DynamicPPL, VarName, RangeAndTransform
using LogDensityProblems: LogDensityProblems
using MarginalLogDensities: MarginalLogDensities

# A thin wrapper to adapt a DynamicPPL.LogDensityFunction to the interface expected by
# MarginalLogDensities. It's helpful to have a struct so that we can dispatch on its type
# below.
struct LogDensityFunctionWrapper{L<:DynamicPPL.LogDensityFunction}
    ldf::L
end
function (lw::LogDensityFunctionWrapper)(x, _)
    return LogDensityProblems.logdensity(lw.ldf, x)
end

"""
    DynamicPPL.marginalize(
        model::DynamicPPL.Model,
        marginalized_varnames::AbstractVector{<:VarName};
        transform_strategy::DynamicPPL.AbstractTransformStrategy=DynamicPPL.LinkAll(),
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

- `transform_strategy`: The transform strategy to use for the model, which determines
   whether the marginalisation is performed in the original (possibly constrained) space
   or in a transformed (unconstrained) space. By default, this is `DynamicPPL.LinkAll()`,
   which transforms all variables to unconstrained space.

   To avoid this transformation and perform the marginalisation in the original space, use
   `DynamicPPL.UnlinkAll()`. You can also use fixed transforms which can in specific
   circumstances improve performance: see the DynamicPPL documentation for more details.

- `getlogprob`: A function which specifies which kind of marginal log-density to compute.
   Its default value is `DynamicPPL.getlogjoint` which returns the marginal log-joint
   probability in the original space (i.e., log-Jacobians from the transformation to
   unconstrained space are ignored).

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

julia> marginalized = marginalize(demo(), [@varname(x)]);

julia> # The resulting callable computes the marginal log-density of `y`.
       marginalized([1.0])
-1.4189385332046727

julia> logpdf(Normal(2.0), 1.0)
-1.4189385332046727
```


!!! warning

    The default usage of `DynamicPPL.LinkAll()` means that, for example, optimization of the
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
    transform_strategy::DynamicPPL.AbstractTransformStrategy=DynamicPPL.LinkAll(),
    getlogprob::Function=DynamicPPL.getlogjoint,
    method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox(),
    kwargs...,
)
    # Construct the marginal log-density model.
    ldf = DynamicPPL.LogDensityFunction(model, getlogprob, transform_strategy)
    initial_params = rand(ldf)
    # Determine the indices for the variables to marginalise out.
    varindices = mapreduce(vcat, marginalized_varnames) do vn
        DynamicPPL.get_range_and_transform(ldf, vn).range
    end
    mld = MarginalLogDensities.MarginalLogDensity(
        LogDensityFunctionWrapper(ldf), initial_params, varindices, (), method; kwargs...
    )
    return mld
end

"""
    DynamicPPL.InitFromVector(
        mld::MarginalLogDensities.MarginalLogDensity{<:LogDensityFunctionWrapper},
        unmarginalized_params::Union{AbstractVector,Nothing}=nothing,
    )

Create a new initialisation strategy using the parameter values stored in `mld` (and
optionally `unmarginalized_params`).

If a Laplace approximation was used for the marginalisation, the values of the marginalized
parameters are set to their mode (note that this only happens if the `mld` object has been
used to compute the marginal log-density at least once, so that the mode has been computed).

If a vector of `unmarginalized_params` is specified, the values for the corresponding
parameters will also be available as part of the initialisation strategy. This vector may be
obtained e.g. by performing an optimization of the marginal log-density.

To use this initialisation strategy to obtain e.g. updated log-probabilities, you should
re-evaluate the model with the values inside the returned VarInfo, for example using:

```julia
init_strategy = DynamicPPL.InitFromVector(mld, unmarginalized_params)
accs = DynamicPPL.OnlyAccsVarInfo((
    DynamicPPL.LogPriorAccumulator(),
    DynamicPPL.LogLikelihoodAccumulator(),
    DynamicPPL.RawValueAccumulator(false),
    # ... whatever else you need
))
_, accs = DynamicPPL.init!!(rng, model, accs, init_strategy, DynamicPPL.UnlinkAll())
```

You can then extract all the updated data from `accs` using DynamicPPL's existing API (see
the DynamicPPL documentation for more details).

## Example

```jldoctest
julia> using DynamicPPL, Distributions, MarginalLogDensities

julia> @model function demo()
           x ~ Normal()
           y ~ Beta(2, 2)
       end
demo (generic function with 2 methods)

julia> # Note that by default `marginalize` uses a linked VarInfo.
       mld = marginalize(demo(), [@varname(x)]);

julia> using MarginalLogDensities: Optimization, OptimizationOptimJL

julia> # Find the mode of the marginal log-density of `y`, with an initial point of `y0`.
       y0 = 2.0; opt_problem = Optimization.OptimizationProblem(mld, [y0])
OptimizationProblem. In-place: true
u0: 1-element Vector{Float64}:
 2.0

julia> # This tells us the optimal (linked) value of `y` is around 0.
       opt_solution = Optimization.solve(opt_problem, OptimizationOptimJL.NelderMead())
retcode: Success
u: 1-element Vector{Float64}:
 4.88281250001733e-5

julia> # Get an initialisation strategy representing the mode of `y`.
       init_strategy = InitFromVector(mld, opt_solution.u);

julia> # Evaluate the model with this initialisation strategy.
       accs = DynamicPPL.OnlyAccsVarInfo((
           DynamicPPL.LogPriorAccumulator(),
           DynamicPPL.LogLikelihoodAccumulator(),
           DynamicPPL.RawValueAccumulator(false),
       ));
       _, accs = DynamicPPL.init!!(demo(), accs, init_strategy, DynamicPPL.UnlinkAll());

julia> # Extract the raw (i.e. untransformed) values for all variables.
       # `x` is set to its mode (which for `Normal()` is zero).
       # Furthermore, `y` is set to the optimal value we found above.
       vals = DynamicPPL.get_raw_values(accs)
VarNamedTuple
├─ x => 0.0
└─ y => 0.5000122070312476
```
"""
function DynamicPPL.InitFromVector(
    mld::MarginalLogDensities.MarginalLogDensity{<:LogDensityFunctionWrapper},
    unmarginalized_params::Union{AbstractVector,Nothing}=nothing,
)
    # Extract the stored parameters, which includes the modes for any marginalized
    # parameters
    full_params = MarginalLogDensities.cached_params(mld)
    # We can then (if needed) set the values for any non-marginalized parameters
    if unmarginalized_params !== nothing
        full_params[MarginalLogDensities.ijoint(mld)] = unmarginalized_params
    end
    return DynamicPPL.InitFromVector(full_params, mld.logdensity.ldf)
end

end
