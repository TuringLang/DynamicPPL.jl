# UniformInit random numbers with range 4 for robust initializations
# Reference: https://mc-stan.org/docs/2_19/reference-manual/initialization.html
randrealuni(rng::Random.AbstractRNG) = 4 * rand(rng) - 2
randrealuni(rng::Random.AbstractRNG, args...) = 4 .* rand(rng, args...) .- 2

"""
    AbstractInitStrategy

Abstract type representing the possible ways of initialising new values for
the random variables in a model (e.g., when creating a new VarInfo).
"""
abstract type AbstractInitStrategy end

"""
    init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, strategy::AbstractInitStrategy)

Generate a new value for a random variable with the given distribution.

!!! warning "Values must be unlinked"
    The values returned by `init` are always in the untransformed space, i.e.,
    they must be within the support of the original distribution. That means that,
    for example, `init(rng, dist, u::UniformInit)` will in general return values that
    are outside the range [u.lower, u.upper].
"""
function init end

"""
    PriorInit()

Obtain new values by sampling from the prior distribution.
"""
struct PriorInit <: AbstractInitStrategy end
init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, ::PriorInit) = rand(rng, dist)

"""
    UniformInit()
    UniformInit(lower, upper)

Obtain new values by first transforming the distribution of the random variable
to unconstrained space, and then sampling a value uniformly between `lower` and
`upper`.

If unspecified, defaults to `(lower, upper) = (-2, 2)`, which mimics Stan's
default initialisation strategy.

# References

[Stan reference manual page on initialization](https://mc-stan.org/docs/reference-manual/execution.html#initialization)
"""
struct UniformInit{T<:AbstractFloat} <: AbstractInitStrategy
    lower::T
    upper::T
    function UniformInit(lower::T, upper::T) where {T<:AbstractFloat}
        lower > upper &&
            throw(ArgumentError("`lower` must be less than or equal to `upper`"))
        return new{T}(lower, upper)
    end
    UniformInit() = UniformInit(-2.0, 2.0)
end
function init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, u::UniformInit)
    b = Bijectors.bijector(dist)
    sz = Bijectors.output_size(b, size(dist))
    y = rand(rng, Uniform(u.lower, u.upper), sz)
    b_inv = Bijectors.inverse(b)
    return b_inv(y)
end

"""
    ParamsInit(params::AbstractDict{<:VarName}, default::AbstractInitStrategy=PriorInit())
    ParamsInit(params::NamedTuple, default::AbstractInitStrategy=PriorInit())

Obtain new values by extracting them from the given dictionary or NamedTuple.
The parameter `default` specifies how new values are to be obtained if they
cannot be found in `params`, or they are specified as `missing`. The default
for `default` is `PriorInit()`.

!!! note
    These values must be provided in the space of the untransformed distribution.
"""
struct ParamsInit{P,S<:AbstractInitStrategy} <: AbstractInitStrategy
    params::P
    default::S
    function ParamsInit(params::AbstractDict{<:VarName}, default::AbstractInitStrategy)
        return new{typeof(params),typeof(default)}(params, default)
    end
    ParamsInit(params::AbstractDict{<:VarName}) = ParamsInit(params, PriorInit())
    function ParamsInit(params::NamedTuple, default::AbstractInitStrategy=PriorInit())
        return ParamsInit(to_varname_dict(params), default)
    end
end
function init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, p::ParamsInit)
    return if hasvalue(p.params, vn)
        x = getvalue(p.params, vn)
        if x === missing
            init(rng, vn, dist, p.default)
        else
            # TODO: Check that the type of x matches the dist?
            x
        end
    else
        init(rng, vn, dist, p.default)
    end
end

"""
    InitContext(
            [rng::Random.AbstractRNG=Random.default_rng()],
            [strategy::AbstractInitStrategy=PriorInit()],
    )

A leaf context that indicates that new values for random variables are
currently being obtained through sampling. Used e.g. when initialising a fresh
VarInfo. Note that, if `leafcontext(model.context) isa InitContext`, then
`evaluate!!(model, varinfo)` will override all values in the VarInfo.
"""
struct InitContext{R<:Random.AbstractRNG,S<:AbstractInitStrategy} <: AbstractContext
    rng::R
    strategy::S
    function InitContext(
        rng::Random.AbstractRNG, strategy::AbstractInitStrategy=PriorInit()
    )
        return new{typeof(rng),typeof(strategy)}(rng, strategy)
    end
    function InitContext(strategy::AbstractInitStrategy=PriorInit())
        return InitContext(Random.default_rng(), strategy)
    end
end
NodeTrait(::InitContext) = IsLeaf()

function tilde_assume(
    ctx::InitContext, dist::Distribution, vn::VarName, vi::AbstractVarInfo
)
    in_varinfo = haskey(vi, vn)
    # `init()` always returns values in original space, i.e. possibly
    # constrained
    x = init(ctx.rng, vn, dist, ctx.strategy)
    # There is a function `to_maybe_linked_internal_transform` that does this,
    # but unfortunately it uses `istrans(vi, vn)` which fails if vn is not in
    # vi, so we have to manually check. By default we will insert an unlinked
    # value into the varinfo.
    is_transformed = in_varinfo ? istrans(vi, vn) : false
    f = if is_transformed
        to_linked_internal_transform(vi, vn, dist)
    else
        to_internal_transform(vi, vn, dist)
    end
    # TODO(penelopeysm): We would really like to do:
    #     y, logjac = with_logabsdet_jacobian(f, x)
    # Unfortunately, `to_{linked_}internal_transform` returns a function that
    # always converts x to a vector, i.e., if dist is univariate, f(x) will be
    # a vector of length 1. It would be nice if we could unify these.
    y = f(x)
    logjac = logabsdetjac(is_transformed ? Bijectors.bijector(dist) : identity, x)
    # Add the new value to the VarInfo. `push!!` errors if the value already
    # exists, hence the need for setindex!!
    if in_varinfo
        vi = setindex!!(vi, y, vn)
    else
        vi = push!!(vi, vn, y, dist)
    end
    # `accumulate_assume!!` wants untransformed values as the second argument.
    vi = accumulate_assume!!(vi, x, -logjac, vn, dist)
    # We always return the untransformed value here, as that will determine
    # what the lhs of the tilde-statement is set to.
    return x, vi
end

function tilde_observe!!(::InitContext, right, left, vn, vi)
    return tilde_observe!!(DefaultContext(), right, left, vn, vi)
end

# """
#     set_initial_values(varinfo::AbstractVarInfo, initial_params::AbstractVector)
#     set_initial_values(varinfo::AbstractVarInfo, initial_params::NamedTuple)
#
# Take the values inside `initial_params`, replace the corresponding values in
# the given VarInfo object, and return a new VarInfo object with the updated values.
#
# This differs from `DynamicPPL.unflatten` in two ways:
#
# 1. It works with `NamedTuple` arguments.
# 2. For the `AbstractVector` method, if any of the elements are missing, it will not
# overwrite the original value in the VarInfo (it will just use the original
# value instead).
# """
# function set_initial_values(varinfo::AbstractVarInfo, initial_params::AbstractVector)
#     throw(
#         ArgumentError(
#             "`initial_params` must be a vector of type `Union{Real,Missing}`. " *
#             "If `initial_params` is a vector of vectors, please flatten it (e.g. using `vcat`) first.",
#         ),
#     )
# end
#
# function set_initial_values(
#     varinfo::AbstractVarInfo, initial_params::AbstractVector{<:Union{Real,Missing}}
# )
#     flattened_param_vals = varinfo[:]
#     length(flattened_param_vals) == length(initial_params) || throw(
#         DimensionMismatch(
#             "Provided initial value size ($(length(initial_params))) doesn't match " *
#             "the model size ($(length(flattened_param_vals))).",
#         ),
#     )
#
#     # Update values that are provided.
#     for i in eachindex(initial_params)
#         x = initial_params[i]
#         if x !== missing
#             flattened_param_vals[i] = x
#         end
#     end
#
#     # Update in `varinfo`.
#     new_varinfo = unflatten(varinfo, flattened_param_vals)
#     return new_varinfo
# end
#
# function set_initial_values(varinfo::AbstractVarInfo, initial_params::NamedTuple)
#     varinfo = deepcopy(varinfo)
#     vars_in_varinfo = keys(varinfo)
#     for v in keys(initial_params)
#         vn = VarName{v}()
#         if !(vn in vars_in_varinfo)
#             for vv in vars_in_varinfo
#                 if subsumes(vn, vv)
#                     throw(
#                         ArgumentError(
#                             "The current model contains sub-variables of $v, such as ($vv). " *
#                             "Using NamedTuple for initial_params is not supported in such a case. " *
#                             "Please use AbstractVector for initial_params instead of NamedTuple.",
#                         ),
#                     )
#                 end
#             end
#             throw(ArgumentError("Variable $v not found in the model."))
#         end
#     end
#     initial_params = NamedTuple(k => v for (k, v) in pairs(initial_params) if v !== missing)
#     return update_values!!(
#         varinfo, initial_params, map(k -> VarName{k}(), keys(initial_params))
#     )
# end
#
# function initialize_parameters!!(vi::AbstractVarInfo, initial_params, model::Model)
#     @debug "Using passed-in initial variable values" initial_params
#
#     # `link` the varinfo if needed.
#     linked = islinked(vi)
#     if linked
#         vi = invlink!!(vi, model)
#     end
#
#     # Set the values in `vi`.
#     vi = set_initial_values(vi, initial_params)
#
#     # `invlink` if needed.
#     if linked
#         vi = link!!(vi, model)
#     end
#
#     return vi
# end
