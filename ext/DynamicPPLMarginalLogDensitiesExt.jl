module DynamicPPLMarginalLogDensitiesExt

using DynamicPPL: DynamicPPL, LogDensityProblems, VarName
using MarginalLogDensities: MarginalLogDensities

_to_varname(n::Symbol) = VarName{n}()
_to_varname(n::VarName) = n

# A thin wrapper to adapt a DynamicPPL.LogDensityFunction to the interface expected by
# MarginalLogDensities. It's helpful to have a struct so that we can dispatch on its type
# below.
struct LogDensityFunctionWrapper{L<:DynamicPPL.LogDensityFunction}
    logdensity::L
end
function (lw::LogDensityFunctionWrapper)(x, _)
    return LogDensityProblems.logdensity(lw.logdensity, x)
end

"""
    marginalize(
        model::DynamicPPL.Model,
        varnames::AbstractVector{<:Union{Symbol,<:VarName}};
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
    varnames::AbstractVector{<:Union{Symbol,<:VarName}};
    varinfo::DynamicPPL.AbstractVarInfo=DynamicPPL.link(DynamicPPL.VarInfo(model), model),
    getlogprob::Function=DynamicPPL.getlogjoint,
    method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox(),
    kwargs...,
)
    # Determine the indices for the variables to marginalise out.
    vns = map(_to_varname, varnames)
    varindices = reduce(vcat, DynamicPPL.vector_getranges(varinfo, vns))
    # Construct the marginal log-density model.
    f = DynamicPPL.LogDensityFunction(model, getlogprob, varinfo)
    mld = MarginalLogDensities.MarginalLogDensity(
        LogDensityFunctionWrapper(f), varinfo[:], varindices, (), method; kwargs...
    )
    return mld
end

"""
    VarInfo(
        mld::MarginalLogDensities.MarginalLogDensity{<:LogDensityFunctionWrapper},
        unmarginalized_params::Union{AbstractVector,Nothing}=nothing
    )

Retrieve the `VarInfo` object used in the marginalisation process.

If a Laplace approximation was used for the marginalisation, the values of the marginalized
parameters are also set to their mode (note that this only happens if the `mld` object has
been used to compute the marginal log-density at least once, so that the mode has been
computed).

If a vector of `unmarginalized_params` is specified, the values for the corresponding
parameters will also be updated in the returned VarInfo. This vector may be obtained e.g. by
performing an optimization of the marginal log-density.

All other aspects of the VarInfo, such as link status, are preserved from the original
VarInfo used in the marginalisation.

!!! note

    The other fields of the VarInfo, e.g. accumulated log-probabilities, will not be
    updated. If you wish to have a fully consistent VarInfo, you should re-evaluate the
    model with the returned VarInfo (e.g. using `vi = last(DynamicPPL.evaluate!!(model,
    vi))`).

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

julia> # Get the VarInfo corresponding to the mode of `y`.
       vi = VarInfo(mld, opt_solution.u);

julia> # `x` is set to its mode (which for `Normal()` is zero).
       vi[@varname(x)]
0.0

julia> # `y` is set to the optimal value we found above.
       DynamicPPL.getindex_internal(vi, @varname(y))
1-element Vector{Float64}:
 4.88281250001733e-5

julia> # To obtain values in the original constrained space, we can either
       # use `getindex`:
       vi[@varname(y)]
0.5000122070312476

julia> # Or invlink the entire VarInfo object using the model:
       vi_unlinked = DynamicPPL.invlink(vi, demo()); vi_unlinked[:]
2-element Vector{Float64}:
 0.0
 0.5000122070312476
```
"""
function DynamicPPL.VarInfo(
    mld::MarginalLogDensities.MarginalLogDensity{<:LogDensityFunctionWrapper},
    unmarginalized_params::Union{AbstractVector,Nothing}=nothing,
)
    # Extract the original VarInfo. Its contents will in general be junk.
    original_vi = mld.logdensity.logdensity.varinfo
    # Extract the stored parameters, which includes the modes for any marginalized
    # parameters
    full_params = MarginalLogDensities.cached_params(mld)
    # We can then (if needed) set the values for any non-marginalized parameters
    if unmarginalized_params !== nothing
        full_params[MarginalLogDensities.ijoint(mld)] = unmarginalized_params
    end
    return DynamicPPL.unflatten(original_vi, full_params)
end

end
