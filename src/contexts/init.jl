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
to unconstrained space, then sampling a value uniformly between `lower` and
`upper`, and transforming that value back to the original space.

If `lower` and `upper` are unspecified, they default to `(-2, 2)`, which mimics
Stan's default initialisation strategy.

Requires that `lower <= upper`.

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
    ParamsInit(
        params::Union{AbstractDict{<:VarName},NamedTuple},
        default::Union{AbstractInitStrategy,Nothing}=PriorInit()
    )

Obtain new values by extracting them from the given dictionary or NamedTuple.

The parameter `default` specifies how new values are to be obtained if they
cannot be found in `params`, or they are specified as `missing`. `default`
can either be an initialisation strategy itself, in which case it will be
used to obtain new values, or it can be `nothing`, in which case an error
will be thrown. The default for `default` is `PriorInit()`.

!!! note
    The values in `params` must be provided in the space of the untransformed
distribution.
"""
struct ParamsInit{P,S<:Union{AbstractInitStrategy,Nothing}} <: AbstractInitStrategy
    params::P
    default::S
    function ParamsInit(
        params::AbstractDict{<:VarName}, default::Union{AbstractInitStrategy,Nothing}
    )
        return new{typeof(params),typeof(default)}(params, default)
    end
    ParamsInit(params::AbstractDict{<:VarName}) = ParamsInit(params, PriorInit())
    function ParamsInit(
        params::NamedTuple, default::Union{AbstractInitStrategy,Nothing}=PriorInit()
    )
        return ParamsInit(to_varname_dict(params), default)
    end
end
function init(rng::Random.AbstractRNG, vn::VarName, dist::Distribution, p::ParamsInit)
    # TODO(penelopeysm): It would be nice to do a check to make sure that all
    # of the parameters in `p.params` were actually used, and either warn or
    # error if they aren't. This is actually quite non-trivial though because
    # the structure of Dicts in particular can have arbitrary nesting.
    return if hasvalue(p.params, vn, dist)
        x = getvalue(p.params, vn, dist)
        if x === missing
            p.default === nothing &&
                error("A `missing` value was provided for the variable `$(vn)`.")
            init(rng, vn, dist, p.default)
        else
            # TODO(penelopeysm): Since x is user-supplied, maybe we could also
            # check here that the type / size of x matches the dist?
            x
        end
    else
        p.default === nothing && error("No value was provided for the variable `$(vn)`.")
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

function tilde_observe!!(::InitContext, right, left, vn, vi)
    return tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
