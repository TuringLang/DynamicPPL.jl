"""
    AbstractInitStrategy

Abstract type representing the possible ways of initialising new values for
the random variables in a model (e.g., when creating a new VarInfo).

Any subtype of `AbstractInitStrategy` must implement the
[`DynamicPPL.init`](@ref) method.
"""
abstract type AbstractInitStrategy end

"""
    init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, strategy::AbstractInitStrategy)

Generate a new value for a random variable with the given distribution.

!!! warning "Return values must be unlinked"
    The values returned by `init` must always be in the untransformed space, i.e.,
    they must be within the support of the original distribution. That means that,
    for example, `init(rng, dist, u::InitFromUniform)` will in general return values that
    are outside the range [u.lower, u.upper].
"""
function init end

"""
    InitFromPrior()

Obtain new values by sampling from the prior distribution.
"""
struct InitFromPrior <: AbstractInitStrategy end
function init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, ::InitFromPrior)
    return rand(rng, dist)
end

"""
    InitFromUniform()
    InitFromUniform(lower, upper)

Obtain new values by first transforming the distribution of the random variable
to unconstrained space, then sampling a value uniformly between `lower` and
`upper`, and transforming that value back to the original space.

If `lower` and `upper` are unspecified, they default to `(-2, 2)`, which mimics
Stan's default initialisation strategy.

Requires that `lower <= upper`.

# References

[Stan reference manual page on initialization](https://mc-stan.org/docs/reference-manual/execution.html#initialization)
"""
struct InitFromUniform{T<:AbstractFloat} <: AbstractInitStrategy
    lower::T
    upper::T
    function InitFromUniform(lower::T, upper::T) where {T<:AbstractFloat}
        lower > upper &&
            throw(ArgumentError("`lower` must be less than or equal to `upper`"))
        return new{T}(lower, upper)
    end
    InitFromUniform() = InitFromUniform(-2.0, 2.0)
end
function init(rng::Random.AbstractRNG, ::VarName, dist::Distribution, u::InitFromUniform)
    b = Bijectors.bijector(dist)
    sz = Bijectors.output_size(b, size(dist))
    y = u.lower .+ ((u.upper - u.lower) .* rand(rng, sz...))
    b_inv = Bijectors.inverse(b)
    x = b_inv(y)
    # 0-dim arrays: https://github.com/TuringLang/Bijectors.jl/issues/398
    if x isa Array{<:Any,0}
        x = x[]
    end
    return x
end

"""
    InitFromParams(
        params::Union{AbstractDict{<:VarName},NamedTuple},
        fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )

Obtain new values by extracting them from the given dictionary or NamedTuple.

The parameter `fallback` specifies how new values are to be obtained if they
cannot be found in `params`, or they are specified as `missing`. `fallback`
can either be an initialisation strategy itself, in which case it will be
used to obtain new values, or it can be `nothing`, in which case an error
will be thrown. The default for `fallback` is `InitFromPrior()`.

!!! note
    The values in `params` must be provided in the space of the untransformed
    distribution.
"""
struct InitFromParams{P,S<:Union{AbstractInitStrategy,Nothing}} <: AbstractInitStrategy
    params::P
    fallback::S
    function InitFromParams(
        params::AbstractDict{<:VarName}, fallback::Union{AbstractInitStrategy,Nothing}
    )
        return new{typeof(params),typeof(fallback)}(params, fallback)
    end
    function InitFromParams(params::AbstractDict{<:VarName})
        return InitFromParams(params, InitFromPrior())
    end
    function InitFromParams(
        params::NamedTuple, fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )
        return InitFromParams(to_varname_dict(params), fallback)
    end
end
function init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, p::InitFromParams)
    # TODO(penelopeysm): It would be nice to do a check to make sure that all
    # of the parameters in `p.params` were actually used, and either warn or
    # error if they aren't. This is actually quite non-trivial though because
    # the structure of Dicts in particular can have arbitrary nesting.
    return if hasvalue(p.params, vn, dist)
        x = getvalue(p.params, vn, dist)
        if x === missing
            p.fallback === nothing &&
                error("A `missing` value was provided for the variable `$(vn)`.")
            init(rng, vn, dist, p.fallback)
        else
            # TODO(penelopeysm): Since x is user-supplied, maybe we could also
            # check here that the type / size of x matches the dist?
            x
        end
    else
        p.fallback === nothing && error("No value was provided for the variable `$(vn)`.")
        init(rng, vn, dist, p.fallback)
    end
end

"""
    InitContext(
            [rng::Random.AbstractRNG=Random.default_rng()],
            [strategy::AbstractInitStrategy=InitFromPrior()],
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
        rng::Random.AbstractRNG, strategy::AbstractInitStrategy=InitFromPrior()
    )
        return new{typeof(rng),typeof(strategy)}(rng, strategy)
    end
    function InitContext(strategy::AbstractInitStrategy=InitFromPrior())
        return InitContext(Random.default_rng(), strategy)
    end
end
NodeTrait(::InitContext) = IsLeaf()

function tilde_assume!!(
    ctx::InitContext, dist::Distribution, vn::VarName, vi::AbstractVarInfo
)
    in_varinfo = haskey(vi, vn)
    # `init()` always returns values in original space, i.e. possibly
    # constrained
    x = init(ctx.rng, vn, dist, ctx.strategy)
    # Determine whether to insert a transformed value into the VarInfo.
    # If the VarInfo alrady had a value for this variable, we will
    # keep the same linked status as in the original VarInfo. If not, we
    # check the rest of the VarInfo to see if other variables are linked.
    # istrans(vi) returns true if vi is nonempty and all variables in vi
    # are linked.
    insert_transformed_value = in_varinfo ? istrans(vi, vn) : istrans(vi)
    f = if insert_transformed_value
        link_transform(dist)
    else
        identity
    end
    y, logjac = with_logabsdet_jacobian(f, x)
    # Add the new value to the VarInfo. `push!!` errors if the value already
    # exists, hence the need for setindex!!.
    if in_varinfo
        vi = setindex!!(vi, y, vn)
    else
        vi = push!!(vi, vn, y, dist)
    end
    # Neither of these set the `trans` flag so we have to do it manually if
    # necessary.
    insert_transformed_value && settrans!!(vi, true, vn)
    # `accumulate_assume!!` wants untransformed values as the second argument.
    vi = accumulate_assume!!(vi, x, logjac, vn, dist)
    # We always return the untransformed value here, as that will determine
    # what the lhs of the tilde-statement is set to.
    return x, vi
end

function tilde_observe!!(
    ::InitContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    return tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
