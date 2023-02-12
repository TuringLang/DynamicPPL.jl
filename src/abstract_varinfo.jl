# Transformation related.
"""
    $(TYPEDEF)

Represents a transformation to be used in `link!!` and `invlink!!`, amongst others.

A concrete implementation of this should implement the following methods:
- [`link!!`](@ref): transforms the [`AbstractVarInfo`](@ref) to the unconstrained space.
- [`invlink!!`](@ref): transforms the [`AbstractVarInfo`](@ref) to the constrained space.

And potentially:
- [`maybe_invlink_before_eval!!`](@ref): hook to decide whether to transform _before_
  evaluating the model.

See also: [`link!!`](@ref), [`invlink!!`](@ref), [`maybe_invlink_before_eval!!`](@ref).
"""
abstract type AbstractTransformation end

"""
    $(TYPEDEF)

Transformation which applies the identity function.
"""
struct NoTransformation <: AbstractTransformation end

"""
    $(TYPEDEF)

Transformation which transforms the variables on a per-need-basis
in the execution of a given `Model`.

This is in constrast to `StaticTransformation` which transforms all variables
_before_ the execution of a given `Model`.

See also: [`StaticTransformation`](@ref).
"""
struct DynamicTransformation <: AbstractTransformation end

"""
    $(TYPEDEF)

Transformation which transforms all variables _before_ the execution of a given `Model`.

This is done through the `maybe_invlink_before_eval!!` method.

See also: [`DynamicTransformation`](@ref), [`maybe_invlink_before_eval!!`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct StaticTransformation{F} <: AbstractTransformation
    "The function, assumed to implement the `Bijectors` interface, to be applied to the variables"
    bijector::F
end

"""
    default_transformation(model::Model[, vi::AbstractVarInfo])

Return the `AbstractTransformation` currently related to `model` and, potentially, `vi`.
"""
default_transformation(model::Model, ::AbstractVarInfo) = default_transformation(model)
default_transformation(::Model) = DynamicTransformation()

"""
    transformation(vi::AbstractVarInfo)

Return the `AbstractTransformation` related to `vi`.
"""
function transformation end

# Accumulation of log-probabilities.
"""
    getlogp(vi::AbstractVarInfo)

Return the log of the joint probability of the observed data and parameters sampled in
`vi`.
"""
function getlogp end

"""
    setlogp!!(vi::AbstractVarInfo, logp)

Set the log of the joint probability of the observed data and parameters sampled in
`vi` to `logp`, mutating if it makes sense.
"""
function setlogp!! end

"""
    acclogp!!(vi::AbstractVarInfo, logp)

Add `logp` to the value of the log of the joint probability of the observed data and
parameters sampled in `vi`, mutating if it makes sense.
"""
function acclogp!! end

"""
    resetlogp!!(vi::AbstractVarInfo)

Reset the value of the log of the joint probability of the observed data and parameters
sampled in `vi` to 0, mutating if it makes sense.
"""
resetlogp!!(vi::AbstractVarInfo) = setlogp!!(vi, zero(getlogp(vi)))

# Variables and their realizations.
@doc """
    keys(vi::AbstractVarInfo)

Return an iterator over all `vns` in `vi`.
""" Base.keys

@doc """
    getindex(vi::AbstractVarInfo, vn::VarName[, dist::Distribution])
    getindex(vi::AbstractVarInfo, vns::Vector{<:VarName}[, dist::Distribution])

Return the current value(s) of `vn` (`vns`) in `vi` in the support of its (their)
distribution(s).

If `dist` is specified, the value(s) will be reshaped accordingly.

See also: [`getindex_raw(vi::AbstractVarInfo, vn::VarName, dist::Distribution)`](@ref)
""" Base.getindex

"""
    getindex(vi::AbstractVarInfo, ::Colon)
    getindex(vi::AbstractVarInfo, ::AbstractSampler)

Return the current value(s) of `vn` (`vns`) in `vi` in the support of its (their)
distribution(s) as a flattened `Vector`.

The default implementation is to call [`values_as`](@ref) with `Vector` as the type-argument.

See also: [`getindex(vi::AbstractVarInfo, vn::VarName, dist::Distribution)`](@ref)
"""
Base.getindex(vi::AbstractVarInfo, ::Colon) = values_as(vi, Vector)
Base.getindex(vi::AbstractVarInfo, ::AbstractSampler) = vi[:]

"""
    getindex_raw(vi::AbstractVarInfo, vn::VarName[, dist::Distribution])
    getindex_raw(vi::AbstractVarInfo, vns::Vector{<:VarName}[, dist::Distribution])

Return the current value(s) of `vn` (`vns`) in `vi`.

If `dist` is specified, the value(s) will be reshaped accordingly.

See also: [`getindex(vi::AbstractVarInfo, vn::VarName, dist::Distribution)`](@ref)

!!! note
    The difference between `getindex(vi, vn, dist)` and `getindex_raw` is that 
    `getindex` will also transform the value(s) to the support of the distribution(s). 
    This is _not_ the case for `getindex_raw`.

"""
function getindex_raw end

"""
    push!!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution)

Push a new random variable `vn` with a sampled value `r` from a distribution `dist` to
the `VarInfo` `vi`, mutating if it makes sense.
"""
function BangBang.push!!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution)
    return BangBang.push!!(vi, vn, r, dist, Set{Selector}([]))
end

"""
    push!!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, spl::AbstractSampler)

Push a new random variable `vn` with a sampled value `r` sampled with a sampler `spl`
from a distribution `dist` to `VarInfo` `vi`, if it makes sense.

The sampler is passed here to invalidate its cache where defined.

$(LEGACY_WARNING)
"""
function BangBang.push!!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, spl::Sampler
)
    return BangBang.push!!(vi, vn, r, dist, spl.selector)
end
function BangBang.push!!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, spl::AbstractSampler
)
    return BangBang.push!!(vi, vn, r, dist)
end

"""
    push!!(vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Selector)

Push a new random variable `vn` with a sampled value `r` sampled with a sampler of
selector `gid` from a distribution `dist` to `VarInfo` `vi`.

$(LEGACY_WARNING)
"""
function BangBang.push!!(
    vi::AbstractVarInfo, vn::VarName, r, dist::Distribution, gid::Selector
)
    return BangBang.push!!(vi, vn, r, dist, Set([gid]))
end

@doc """
    empty!!(vi::AbstractVarInfo)

Empty the fields of `vi.metadata` and reset `vi.logp[]` and `vi.num_produce[]` to
zeros.

This is useful when using a sampling algorithm that assumes an empty `vi`, e.g. `SMC`.
""" BangBang.empty!!

@doc """
    isempty(vi::AbstractVarInfo)

Return true if `vi` is empty and false otherwise.
""" Base.isempty

"""
    values_as(varinfo[, Type])

Return the values/realizations in `varinfo` as `Type`, if implemented.

If no `Type` is provided, return values as stored in `varinfo`.

# Examples

`SimpleVarInfo` with `NamedTuple`:

```jldoctest
julia> data = (x = 1.0, m = [2.0]);

julia> values_as(SimpleVarInfo(data))
(x = 1.0, m = [2.0])

julia> values_as(SimpleVarInfo(data), NamedTuple)
(x = 1.0, m = [2.0])

julia> values_as(SimpleVarInfo(data), OrderedDict)
OrderedDict{VarName{sym, Setfield.IdentityLens} where sym, Any} with 2 entries:
  x => 1.0
  m => [2.0]

julia> values_as(SimpleVarInfo(data), Vector)
2-element Vector{Float64}:
 1.0
 2.0
```

`SimpleVarInfo` with `OrderedDict`:

```jldoctest
julia> data = OrderedDict{Any,Any}(@varname(x) => 1.0, @varname(m) => [2.0]);

julia> values_as(SimpleVarInfo(data))
OrderedDict{Any, Any} with 2 entries:
  x => 1.0
  m => [2.0]

julia> values_as(SimpleVarInfo(data), NamedTuple)
(x = 1.0, m = [2.0])

julia> values_as(SimpleVarInfo(data), OrderedDict)
OrderedDict{Any, Any} with 2 entries:
  x => 1.0
  m => [2.0]

julia> values_as(SimpleVarInfo(data), Vector)
2-element Vector{Float64}:
 1.0
 2.0
```

`TypedVarInfo`:

```jldoctest
julia> # Just use an example model to construct the `VarInfo` because we're lazy.
       vi = VarInfo(DynamicPPL.TestUtils.demo_assume_dot_observe());

julia> vi[@varname(s)] = 1.0; vi[@varname(m)] = 2.0;

julia> # For the sake of brevity, let's just check the type.
       md = values_as(vi); md.s isa DynamicPPL.Metadata
true

julia> values_as(vi, NamedTuple)
(s = 1.0, m = 2.0)

julia> values_as(vi, OrderedDict)
OrderedDict{VarName{sym, Setfield.IdentityLens} where sym, Float64} with 2 entries:
  s => 1.0
  m => 2.0

julia> values_as(vi, Vector)
2-element Vector{Float64}:
 1.0
 2.0
```

`UntypedVarInfo`:

```jldoctest
julia> # Just use an example model to construct the `VarInfo` because we're lazy.
       vi = VarInfo(); DynamicPPL.TestUtils.demo_assume_dot_observe()(vi);

julia> vi[@varname(s)] = 1.0; vi[@varname(m)] = 2.0;

julia> # For the sake of brevity, let's just check the type.
       values_as(vi) isa DynamicPPL.Metadata
true

julia> values_as(vi, NamedTuple)
(s = 1.0, m = 2.0)

julia> values_as(vi, OrderedDict)
OrderedDict{VarName{sym, Setfield.IdentityLens} where sym, Float64} with 2 entries:
  s => 1.0
  m => 2.0

julia> values_as(vi, Vector)
2-element Vector{Real}:
 1.0
 2.0
```
"""
function values_as end

"""
    eltype(vi::AbstractVarInfo, spl::Union{AbstractSampler,SampleFromPrior}

Determine the default `eltype` of the values returned by `vi[spl]`.

!!! warning
    This should generally not be called explicitly, as it's only used in
    [`matchingvalue`](@ref) to determine the default type to use in place of
    type-parameters passed to the model.
    
    This method is considered legacy, and is likely to be deprecated in the future.
"""
function Base.eltype(vi::AbstractVarInfo, spl::Union{AbstractSampler,SampleFromPrior})
    return eltype(Core.Compiler.return_type(getindex, Tuple{typeof(vi),typeof(spl)}))
end

# Transformations
"""
    istrans(vi::AbstractVarInfo[, vns::Union{VarName, AbstractVector{<:Varname}}])

Return `true` if `vi` is working in unconstrained space, and `false`
if `vi` is assuming realizations to be in support of the corresponding distributions.

If `vns` is provided, then only check if this/these varname(s) are transformed.

!!! warning
    Not all implementations of `AbstractVarInfo` support transforming only a subset of
    the variables.
"""
istrans(vi::AbstractVarInfo) = istrans(vi, collect(keys(vi)))
function istrans(vi::AbstractVarInfo, vns::AbstractVector)
    return !isempty(vns) && all(Base.Fix1(istrans, vi), vns)
end

"""
    settrans!!(vi::AbstractVarInfo, trans::Bool[, vn::VarName])

Return `vi` with `istrans(vi, vn)` evaluating to `true`.

If `vn` is not specified, then `istrans(vi)` evaluates to `true` for all variables.
"""
function settrans!! end

"""
    link!!([t::AbstractTransformation, ]vi::AbstractVarInfo, model::Model)
    link!!([t::AbstractTransformation, ]vi::AbstractVarInfo, spl::AbstractSampler, model::Model)

Transforms the variables in `vi` to their linked space, using the transformation `t`.

If `t` is not provided, `default_transformation(model, vi)` will be used.

See also: [`default_transformation`](@ref), [`invlink!!`](@ref).
"""
link!!(vi::AbstractVarInfo, model::Model) = link!!(vi, SampleFromPrior(), model)
function link!!(t::AbstractTransformation, vi::AbstractVarInfo, model::Model)
    return link!!(t, vi, SampleFromPrior(), model)
end
function link!!(vi::AbstractVarInfo, spl::AbstractSampler, model::Model)
    # Use `default_transformation` to decide which transformation to use if none is specified.
    return link!!(default_transformation(model, vi), vi, spl, model)
end

"""
    invlink!!([t::AbstractTransformation, ]vi::AbstractVarInfo, model::Model)
    invlink!!([t::AbstractTransformation, ]vi::AbstractVarInfo, spl::AbstractSampler, model::Model)

Transform the variables in `vi` to their constrained space, using the (inverse of) 
transformation `t`.

If `t` is not provided, `default_transformation(model, vi)` will be used.

See also: [`default_transformation`](@ref), [`link!!`](@ref).
"""
invlink!!(vi::AbstractVarInfo, model::Model) = invlink!!(vi, SampleFromPrior(), model)
function invlink!!(t::AbstractTransformation, vi::AbstractVarInfo, model::Model)
    return invlink!!(t, vi, SampleFromPrior(), model)
end
function invlink!!(vi::AbstractVarInfo, spl::AbstractSampler, model::Model)
    # Here we extract the `transformation` from `vi` rather than using the default one.
    return invlink!!(transformation(vi), vi, spl, model)
end

# Vector-based ones.
function link!!(
    t::StaticTransformation{<:Bijectors.Transform},
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    model::Model,
)
    b = inverse(t.bijector)
    x = vi[spl]
    y, logjac = with_logabsdet_jacobian(b, x)

    lp_new = getlogp(vi) - logjac
    vi_new = setlogp!!(unflatten(vi, spl, y), lp_new)
    return settrans!!(vi_new, t)
end

function invlink!!(
    t::StaticTransformation{<:Bijectors.Transform},
    vi::AbstractVarInfo,
    spl::AbstractSampler,
    model::Model,
)
    b = t.bijector
    y = vi[spl]
    x, logjac = with_logabsdet_jacobian(b, y)

    lp_new = getlogp(vi) + logjac
    vi_new = setlogp!!(unflatten(vi, spl, x), lp_new)
    return settrans!!(vi_new, NoTransformation())
end

"""
    maybe_invlink_before_eval!!([t::Transformation,] vi, context, model)

Return a possibly invlinked version of `vi`.

This will be called prior to `model` evaluation, allowing one to perform a single
`invlink!!` _before_ evaluation rather than lazyily evaluating the transforms on as-we-need
basis as is done with [`DynamicTransformation`](@ref).

See also: [`StaticTransformation`](@ref), [`DynamicTransformation`](@ref).

# Examples
```julia-repl
julia> using DynamicPPL, Distributions, Bijectors

julia> @model demo() = x ~ Normal()
demo (generic function with 2 methods)

julia> # By subtyping `Transform`, we inherit the `(inv)link!!`.
       struct MyBijector <: Bijectors.Transform end

julia> # Define some dummy `inverse` which will be used in the `link!!` call.
       Bijectors.inverse(f::MyBijector) = identity

julia> # We need to define `with_logabsdet_jacobian` for `MyBijector`
       # (`identity` already has `with_logabsdet_jacobian` defined)
       function Bijectors.with_logabsdet_jacobian(::MyBijector, x)
           # Just using a large number of the logabsdet-jacobian term
           # for demonstration purposes.
           return (x, 1000)
       end

julia> # Change the `default_transformation` for our model to be a
       # `StaticTransformation` using `MyBijector`.
       function DynamicPPL.default_transformation(::Model{typeof(demo)})
           return DynamicPPL.StaticTransformation(MyBijector())
       end

julia> model = demo();

julia> vi = SimpleVarInfo(x=1.0)
SimpleVarInfo((x = 1.0,), 0.0)

julia> # Uses the `inverse` of `MyBijector`, which we have defined as `identity`
       vi_linked = link!!(vi, model)
Transformed SimpleVarInfo((x = 1.0,), 0.0)

julia> # Now performs a single `invlink!!` before model evaluation.
       logjoint(model, vi_linked)
-1001.4189385332047
```
"""
function maybe_invlink_before_eval!!(
    vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    return maybe_invlink_before_eval!!(transformation(vi), vi, context, model)
end
function maybe_invlink_before_eval!!(
    ::NoTransformation, vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    return vi
end
function maybe_invlink_before_eval!!(
    ::DynamicTransformation, vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    # `DynamicTransformation` is meant to _not_ do the transformation statically, hence we do nothing.
    return vi
end
function maybe_invlink_before_eval!!(
    t::StaticTransformation, vi::AbstractVarInfo, context::AbstractContext, model::Model
)
    return invlink!!(t, vi, _default_sampler(context), model)
end

function _default_sampler(context::AbstractContext)
    return _default_sampler(NodeTrait(_default_sampler, context), context)
end
_default_sampler(::IsLeaf, context::AbstractContext) = SampleFromPrior()
function _default_sampler(::IsParent, context::AbstractContext)
    return _default_sampler(childcontext(context))
end

# Utilities
"""
    unflatten(vi::AbstractVarInfo[, context::AbstractContext], x::AbstractVector)

Return a new instance of `vi` with the values of `x` assigned to the variables.

If `context` is provided, `x` is assumed to be realizations only for variables not
filtered out by `context`.
"""
function unflatten(varinfo::AbstractVarInfo, context::AbstractContext, θ)
    if hassampler(context)
        unflatten(getsampler(context), varinfo, context, θ)
    else
        DynamicPPL.unflatten(varinfo, θ)
    end
end

# TODO: deprecate this once `sampler` is no longer the main way of filtering out variables.
function unflatten(sampler::AbstractSampler, varinfo::AbstractVarInfo, ::AbstractContext, θ)
    return unflatten(varinfo, sampler, θ)
end

"""
    tonamedtuple(vi::AbstractVarInfo)

Convert a `vi` into a `NamedTuple` where each variable symbol maps to the values and 
indexing string of the variable.

For example, a model that had a vector of vector-valued
variables `x` would return

```julia
(x = ([1.5, 2.0], [3.0, 1.0], ["x[1]", "x[2]"]), )
```
"""
function tonamedtuple end

# TODO: Clean up all this linking stuff once and for all!
link_transform(dist) = bijector(dist)
link_transform(::LKJ) = Bijectors.VecCorrBijector()

invlink_transform(dist) = inverse(bijector(dist))
invlink_transform(::LKJ) = inverse(Bijectors.VecCorrBijector())

"""
    with_logabsdet_jacobian_and_reconstruct([f, ]dist, x)

Like `Bijectors.with_logabsdet_jacobian(f, x)`, but also ensures the resulting
value is reconstructed to the correct type and shape according to `dist`.
"""
function with_logabsdet_jacobian_and_reconstruct(f, dist, x)
    x_recon = reconstruct(dist, x)
    return with_logabsdet_jacobian(f, x_recon)
end

function with_logabsdet_jacobian_and_reconstruct(
    f::Bijectors.Inverse{Bijectors.VecCorrBijector}, ::LKJ, x::AbstractVector
)
    # "Reconstruction" occurs in the `LKJ` bijector.
    return with_logabsdet_jacobian(f, x)
end

# TODO: Once we `(inv)link` isn't used heavily in `getindex(vi, vn)`, we can
# just use `first ∘ with_logabsdet_jacobian` to reduce the maintenance burden.
# NOTE: `reconstruct` is no-op if `val` is already of correct shape.
"""
    link_and_reconstruct(dist, val)
    link_and_reconstruct(vi::AbstractVarInfo, vi::VarName, dist, val)

Return linked and reconstructed `val`.
"""
link_and_reconstruct(f, dist, val) = f(reconstruct(dist, val))
link_and_reconstruct(dist, val) = link_and_reconstruct(link_transform(dist), dist, val)
function link_and_reconstruct(::AbstractVarInfo, ::VarName, dist, val)
    return link_and_reconstruct(dist, val)
end

"""
    invlink_and_reconstruct(dist, val)
    invlink_and_reconstruct(vi::AbstractVarInfo, vn::VarName, dist, val)

Return invlinked and reconstructed `val`.

See also: [`reconstruct`](@ref).
"""
invlink_and_reconstruct(f, dist, val) = f(reconstruct(dist, val))
invlink_and_reconstruct(dist, val) = invlink_and_reconstruct(invlink_transform(dist), dist, val)
function invlink_and_reconstruct(::AbstractVarInfo, ::VarName, dist, val)
    return invlink_and_reconstruct(dist, val)
end

"""
    maybe_link_and_reconstruct(vi::AbstractVarInfo, vn::VarName, dist, val)

Return reconstructed `val`, possibly linked if `istrans(vi, vn)` is `true`.
"""
function maybe_link_and_reconstruct(vi::AbstractVarInfo, vn::VarName, dist, val)
    return if istrans(vi, vn)
        link_and_reconstruct(vi, vn, dist, val)
    else
        reconstruct(dist, val)
    end
end

"""
    maybe_invlink_and_reconstruct(vi::AbstractVarInfo, vn::VarName, dist, val)

Return reconstructed `val`, possibly invlinked if `istrans(vi, vn)` is `true`.
"""
function maybe_invlink_and_reconstruct(vi::AbstractVarInfo, vn::VarName, dist, val)
    return if istrans(vi, vn)
        invlink_and_reconstruct(vi, vn, dist, val)
    else
        reconstruct(dist, val)
    end
end

# Special cases.
function invlink_and_reconstruct(f::Bijectors.Inverse{Bijectors.VecCorrBijector}, ::LKJ, val::AbstractVector{<:Real})
    # Reconstruction already occurs in `invlink` here.
    return f(val)
end

# Legacy code that is currently overloaded for the sake of simplicity.
# TODO: Remove when possible.
increment_num_produce!(::AbstractVarInfo) = nothing
setgid!(vi::AbstractVarInfo, gid::Selector, vn::VarName) = nothing
