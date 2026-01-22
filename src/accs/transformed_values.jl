"""
    abstract type AbstractLinkStrategy end

An abstract type for strategies specifying which variables to link or unlink.

Current subtypes are [`LinkAll`](@ref), [`UnlinkAll`](@ref), [`LinkSome`](@ref), and
[`UnlinkSome`](@ref).

!!! warning
    Even though the subtypes listed above are public, this abstract type is not part of
    DynamicPPL's public API and end users should not subtype this. (There should really not
    be any reason to!)

For subtypes of `AbstractLinkStrategy`, the only method that needs to be overloaded is
`DynamicPPL.generate_linked_value`. Note that this is also an internal function.
"""
abstract type AbstractLinkStrategy end

"""
    generate_linked_value(linker::AbstractLinkStrategy, vn::VarName)

Determine whether a variable with name `vn` should be linked according to the
`linker` strategy.
"""
function generate_linked_value end

"""
    UnlinkAll() <: AbstractLinkStrategy

Indicate that all variables should be unlinked.
"""
struct UnlinkAll <: AbstractLinkStrategy end
generate_linked_value(::UnlinkAll, ::VarName, ::AbstractTransformedValue) = false

"""
    LinkAll() <: AbstractLinkStrategy

Indicate that all variables should be linked.
"""
struct LinkAll <: AbstractLinkStrategy end
generate_linked_value(::LinkAll, ::VarName, ::AbstractTransformedValue) = true

"""
    LinkSome(vns) <: AbstractLinkStrategy

Indicate that the variables in `vns` must be linked. The link statuses of other variables
are preserved. `vns` should be some iterable collection of `VarName`s, although there is no
strict type requirement.
"""
struct LinkSome{V} <: AbstractLinkStrategy
    vns::V
end
generate_linked_value(::LinkSome, ::VarName, ::LinkedVectorValue) = true
function generate_linked_value(
    linker::LinkSome, vn::VarName, ::Union{VectorValue,UntransformedValue}
)
    return any(linker_vn -> subsumes(linker_vn, vn), linker.vns)
end

"""
    UnlinkSome(vns}) <: AbstractLinkStrategy

Indicate that the variables in `vns` must not Be linked. The link statuses of other
variables are preserved. `vns` should be some iterable collection of `VarName`s, although
there is no strict type requirement.
"""
struct UnlinkSome{V} <: AbstractLinkStrategy
    vns::V
end
function generate_linked_value(
    ::UnlinkSome, ::VarName, ::Union{VectorValue,UntransformedValue}
)
    return false
end
function generate_linked_value(linker::UnlinkSome, vn::VarName, ::LinkedVectorValue)
    return !any(linker_vn -> subsumes(linker_vn, vn), linker.vns)
end

# A transformed value accumulator is just a `VNTAccumulator` that collects
# `AbstractTransformedValues`. It has a double role:
#
#   (1) It stores transformed values; which makes it similar in principle to VarInfo itself;
#   (2) It also converts those transformed values between linked (`LinkedVectorValue`) and
#       unlinked (`VectorValue`) according to an `AbstractLinkStrategy`. In the process it
#       also keeps track of the log-Jacobian adjustments that need to be made.
#
# For example, if the strategy is `LinkAll()`, then all transformed values stored will be
# linked. That gives us a way to essentially 'link' a VarInfo, although the idea here is 
# different: instead of reaching into the VarInfo and modifying its contents, we instead
# run the model and accumulate transformed values that are linked.
#
# When executing a model with a VarInfo, the `tval` passed through will always be either
# a `VectorValue` or a `LinkedVectorValue` (never an `UntransformedValue`), since that
# is what VarInfo stores. So you might ask: why do we need to handle `UntransformedValue`s?
#
# The answer is that initialisation strategies like `InitFromPrior()` will generate
# `UntransformedValue`s, which can be passed through all the way into these accumulators.
# What this means is that we can _immediately_ generate a linked VarInfo by sampling from
# the prior, without having to first create an unlinked VarInfo and then link it! See the
# `VarInfo` constructors in `src/varinfo.jl` for examples.
#
# This accumulator is used in the implementation of `link!!` and `invlink!!`; however, we
# can't define them in this file as we haven't defined the `VarInfo` struct yet. See
# `src/varinfo.jl` for the definitions.

const LINK_ACCNAME = :LinkAccumulator
mutable struct Link!{V<:AbstractLinkStrategy}
    strategy::V
    logjac::LogProbType
    Link!(vns::V) where {V} = new{V}(vns, zero(LogProbType))
end

function (linker::Link!)(val::Any, tval::LinkedVectorValue, logjac::Any, vn::Any, dist::Any)
    original_val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    return if generate_linked_value(linker.strategy, vn, tval)
        # No need to do anything.
        tval
    else
        # tval contains a linked value, we need to invlink it.
        # Note that logjac of from_linked_vec_transform will already be included in
        # the `logjac` argument (!) so we only need to add the logjac of to_vec_transform
        # here, which in principle should be zero, but...
        f = to_vec_transform(dist)
        new_val, vect_logjac = with_logabsdet_jacobian(f, val)
        # In this case, the LogJacobianAccumulator will have counted logjac. We want to
        # cancel out that contribution here since we are removing the linking.
        linker.logjac += vect_logjac - logjac
        VectorValue(new_val, inverse(f), original_val_size)
    end
end

function (linker::Link!)(val::Any, tval::VectorValue, logjac::Any, vn::Any, dist::Any)
    # Note that we don't need to care about the logjac passed in, since
    # LogJacobianAccumulator takes care of _that_.
    original_val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    return if generate_linked_value(linker.strategy, vn, tval)
        # tval contains an unlinked value, we need to generate a new linked value.
        f = to_linked_vec_transform(dist)
        new_val, link_logjac = with_logabsdet_jacobian(f, val)
        linker.logjac += link_logjac
        LinkedVectorValue(new_val, inverse(f), original_val_size)
    else
        # No need to do anything.
        tval
    end
end

function (linker::Link!)(
    val::Any, tval::UntransformedValue, logjac::Any, vn::Any, dist::Any
)
    original_val_size = hasmethod(size, Tuple{typeof(val)}) ? size(val) : ()
    # Inside here we can just use `val` directly since that is the same thing as unwrapping
    # `tval`.
    return if generate_linked_value(linker.strategy, vn, tval)
        f = to_linked_vec_transform(dist)
        new_val, link_logjac = with_logabsdet_jacobian(f, val)
        linker.logjac += link_logjac
        LinkedVectorValue(new_val, inverse(f), original_val_size)
    else
        f = to_vec_transform(dist)
        # logjac should really be zero, but well. Just check, I guess.
        new_val, logjac = with_logabsdet_jacobian(f, val)
        linker.logjac += logjac
        VectorValue(new_val, inverse(f), original_val_size)
    end
end
