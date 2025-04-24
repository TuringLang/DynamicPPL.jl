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
    merge_transformations(transformation_left, transformation_right)

Merge two transformations.

The main use of this is in [`merge(::AbstractVarInfo, ::AbstractVarInfo)`](@ref).
"""
function merge_transformations(::NoTransformation, ::NoTransformation)
    return NoTransformation()
end
function merge_transformations(::DynamicTransformation, ::DynamicTransformation)
    return DynamicTransformation()
end
function merge_transformations(left::StaticTransformation, right::StaticTransformation)
    return StaticTransformation(merge_bijectors(left.bijector, right.bijector))
end

function merge_bijectors(left::Bijectors.NamedTransform, right::Bijectors.NamedTransform)
    return Bijectors.NamedTransform(merge_bijector(left.bs, right.bs))
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
    getlogjoint(vi::AbstractVarInfo)

Return the log of the joint probability of the observed data and parameters in `vi`.

See also: [`getlogprior`](@ref), [`getloglikelihood`](@ref).
"""
getlogjoint(vi::AbstractVarInfo) = getlogprior(vi) + getloglikelihood(vi)
function getlogp(vi::AbstractVarInfo)
    Base.depwarn("getlogp is deprecated, use getlogjoint instead", :getlogp)
    return getlogjoint(vi)
end

"""
    setaccs!!(vi::AbstractVarInfo, accs::AccumulatorTuple)
    setaccs!!(vi::AbstractVarInfo, accs::NTuple{N,AbstractAccumulator} where {N})

Update the `AccumulatorTuple` of `vi` to `accs`, mutating if it makes sense.

`setaccs!!(vi:AbstractVarInfo, accs::AccumulatorTuple) should be implemented by each subtype
of `AbstractVarInfo`.
"""
function setaccs!!(vi::AbstractVarInfo, accs::NTuple{N,AbstractAccumulator}) where {N}
    return setaccs!!(vi, AccumulatorTuple(accs))
end

"""
    getaccs(vi::AbstractVarInfo)

Return the `AccumulatorTuple` of `vi`.

This should be implemented by each subtype of `AbstractVarInfo`. `getaccs` is not
user-facing, but used in the implementation of many other functions.
"""
function getaccs end

"""
    hasacc(vi::AbstractVarInfo, ::Val{accname}) where {accname}

Return a boolean for whether `vi` has an accumulator with name `accname`.
"""
hasacc(vi::AbstractVarInfo, accname::Val) = haskey(getaccs(vi), accname)
function hassacc(vi::AbstractVarInfo, accname::Symbol)
    return error(
        """
        The method hasacc(vi::AbstractVarInfo, accname::Symbol) does not exist. For type
        stability reasons use hasacc(vi::AbstractVarInfo, Val(accname)) instead.
        """
    )
end

"""
    acckeys(vi::AbstractVarInfo)

Return the names of the accumulators in `vi`.
"""
acckeys(vi::AbstractVarInfo) = keys(getaccs(vi))

"""
    getlogprior(vi::AbstractVarInfo)

Return the log of the prior probability of the parameters in `vi`.

See also: [`getlogjoint`](@ref), [`getloglikelihood`](@ref), [`setlogprior!!`](@ref).
"""
getlogprior(vi::AbstractVarInfo) = getacc(vi, Val(:LogPrior)).logp

"""
    getloglikelihood(vi::AbstractVarInfo)

Return the log of the likelihood probability of the observed data in `vi`.

See also: [`getlogjoint`](@ref), [`getlogprior`](@ref), [`setloglikelihood!!`](@ref).
"""
getloglikelihood(vi::AbstractVarInfo) = getacc(vi, Val(:LogLikelihood)).logp

"""
    setacc!!(vi::AbstractVarInfo, acc::AbstractAccumulator)

Add `acc` to the `AccumulatorTuple` of `vi`, mutating if it makes sense.

If an accumulator with the same [`accumulator_name`](@ref) already exists, it will be
replaced.

See also: [`getaccs`](@ref).
"""
function setacc!!(vi::AbstractVarInfo, acc::AbstractAccumulator)
    return setaccs!!(vi, setacc!!(getaccs(vi), acc))
end

"""
    setlogprior!!(vi::AbstractVarInfo, logp)

Set the log of the prior probability of the parameters sampled in `vi` to `logp`.

See also: [`setloglikelihood!!`](@ref), [`setlogp!!`](@ref), [`getlogprior`](@ref).
"""
setlogprior!!(vi::AbstractVarInfo, logp) = setacc!!(vi, LogPrior(logp))

"""
    setloglikelihood!!(vi::AbstractVarInfo, logp)

Set the log of the likelihood probability of the observed data sampled in `vi` to `logp`.

See also: [`setlogprior!!`](@ref), [`setlogp!!`](@ref), [`getloglikelihood`](@ref).
"""
setloglikelihood!!(vi::AbstractVarInfo, logp) = setacc!!(vi, LogLikelihood(logp))

"""
    setlogp!!(vi::AbstractVarInfo, logp)

Set the log of the joint probability of the observed data and parameters sampled in
`vi` to `logp`, mutating if it makes sense.

See also: [`setlogprior!!`](@ref), [`setloglikelihood!!`](@ref), [`getlogp`](@ref).
"""
function setlogp!!(vi::AbstractVarInfo, logp)
    Base.depwarn(
        "setlogp!! is deprecated, use setlogprior!! or setloglikelihood!! instead",
        :setlogp!!,
    )
    vi = setlogprior!!(vi, zero(logp))
    vi = setloglikelihood!!(vi, logp)
    return vi
end

"""
    getacc(vi::AbstractVarInfo, accname)

Return the `AbstractAccumulator` of `vi` with name `accname`.
"""
function getacc(vi::AbstractVarInfo, accname::Val)
    return getacc(getaccs(vi), accname)
end
function getacc(vi::AbstractVarInfo, accname::Symbol)
    return error(
        """
        The method getacc(vi::AbstractVarInfo, accname::Symbol) does not exist. For type
        stability reasons use getacc(vi::AbstractVarInfo, Val(accname)) instead.
        """
    )
end

"""
    accumulate_assume!!(vi::AbstractVarInfo, val, logjac, vn, right)

Update all the accumulators of `vi` by calling `accumulate_assume!!` on them.
"""
function accumulate_assume!!(vi::AbstractVarInfo, val, logjac, vn, right)
    return map_accumulators!!(acc -> accumulate_assume!!(acc, val, logjac, vn, right), vi)
end

"""
    accumulate_observe!!(vi::AbstractVarInfo, right, left, vn)

Update all the accumulators of `vi` by calling `accumulate_observe!!` on them.
"""
function accumulate_observe!!(vi::AbstractVarInfo, right, left, vn)
    return map_accumulators!!(acc -> accumulate_observe!!(acc, right, left, vn), vi)
end

"""
    map_accumulators(vi::AbstractVarInfo, func::Function)

Update all accumulators of `vi` by calling `func` on them and replacing them with the return
values.
"""
function map_accumulators!!(func::Function, vi::AbstractVarInfo)
    return setaccs!!(vi, map(func, getaccs(vi)))
end

"""
    map_accumulator!!(func::Function, vi::AbstractVarInfo, ::Val{accname}) where {accname}

Update the accumulator `accname` of `vi` by calling `func` on it and replacing it with the
return value.
"""
function map_accumulator!!(func::Function, vi::AbstractVarInfo, accname::Val)
    return setaccs!!(vi, map_accumulator(func, getaccs(vi), accname))
end

function map_accumulator!!(func::Function, vi::AbstractVarInfo, accname::Symbol)
    return error(
        """
        The method map_accumulator!!(func::Function, vi::AbstractVarInfo, accname::Symbol)
        does not exist. For type stability reasons use
        map_accumulator!!(func::Function, vi::AbstractVarInfo, ::Val{accname}) instead.
        """
    )
end

"""
    acclogprior!!(vi::AbstractVarInfo, logp)

Add `logp` to the value of the log of the prior probability in `vi`.

See also: [`accloglikelihood!!`](@ref), [`acclogp!!`](@ref), [`getlogprior`](@ref), [`setlogprior!!`](@ref).
"""
function acclogprior!!(vi::AbstractVarInfo, logp)
    return map_accumulator!!(acc -> acc + LogPrior(logp), vi, Val(:LogPrior))
end

"""
    accloglikelihood!!(vi::AbstractVarInfo, logp)

Add `logp` to the value of the log of the likelihood in `vi`.

See also: [`accloglikelihood!!`](@ref), [`acclogp!!`](@ref), [`getloglikelihood`](@ref), [`setloglikelihood!!`](@ref).
"""
function accloglikelihood!!(vi::AbstractVarInfo, logp)
    return map_accumulator!!(acc -> acc + LogLikelihood(logp), vi, Val(:LogLikelihood))
end

"""
    acclogp!!(vi::AbstractVarInfo, logp)

Add `logp` to the value of the log of the joint probability of the observed data and
parameters sampled in `vi`, mutating if it makes sense.
"""
function acclogp!!(vi::AbstractVarInfo, logp)
    Base.depwarn(
        "acclogp!! is deprecated, use acclogprior!! or accloglikelihood!! instead",
        :acclogp!!,
    )
    return accloglikelihood!!(vi, logp)
end

"""
    resetlogp!!(vi::AbstractVarInfo)

Reset the values of the log probabilities (prior and likelihood) in `vi`
"""
function resetlogp!!(vi::AbstractVarInfo)
    if hasacc(vi, Val(:LogPrior))
        vi = map_accumulator!!(zero, vi, Val(:LogPrior))
    end
    if hasacc(vi, Val(:LogLikelihood))
        vi = map_accumulator!!(zero, vi, Val(:LogLikelihood))
    end
    return vi
end

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

If `dist` is specified, the value(s) will be massaged into the representation expected by `dist`.
""" Base.getindex

"""
    getindex(vi::AbstractVarInfo, ::Colon)

Return the current value(s) of `vn` (`vns`) in `vi` in the support of its (their)
distribution(s) as a flattened `Vector`.

The default implementation is to call [`values_as`](@ref) with `Vector` as the type-argument.

See also: [`getindex(vi::AbstractVarInfo, vn::VarName, dist::Distribution)`](@ref)
"""
Base.getindex(vi::AbstractVarInfo, ::Colon) = values_as(vi, Vector)

"""
    getindex_internal(vi::AbstractVarInfo, vn::VarName)
    getindex_internal(vi::AbstractVarInfo, vns::Vector{<:VarName})
    getindex_internal(vi::AbstractVarInfo, ::Colon)

Return the internal value of the varname `vn`, varnames `vns`, or all varnames
in `vi` respectively. The internal value is the value of the variables that is
stored in the varinfo object; this may be the actual realisation of the random
variable (i.e. the value sampled from the distribution), or it may have been
transformed to Euclidean space, depending on whether the varinfo was linked.

See https://turinglang.org/docs/developers/transforms/dynamicppl/ for more
information on how transformed variables are stored in DynamicPPL.

See also: [`getindex(vi::AbstractVarInfo, vn::VarName, dist::Distribution)`](@ref)
"""
function getindex_internal end

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
OrderedDict{VarName{sym, typeof(identity)} where sym, Any} with 2 entries:
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

`VarInfo` with `NamedTuple` of `Metadata`:

```jldoctest
julia> # Just use an example model to construct the `VarInfo` because we're lazy.
       vi = DynamicPPL.typed_varinfo(DynamicPPL.TestUtils.demo_assume_dot_observe());

julia> vi[@varname(s)] = 1.0; vi[@varname(m)] = 2.0;

julia> # For the sake of brevity, let's just check the type.
       md = values_as(vi); md.s isa Union{DynamicPPL.Metadata, DynamicPPL.VarNamedVector}
true

julia> values_as(vi, NamedTuple)
(s = 1.0, m = 2.0)

julia> values_as(vi, OrderedDict)
OrderedDict{VarName{sym, typeof(identity)} where sym, Float64} with 2 entries:
  s => 1.0
  m => 2.0

julia> values_as(vi, Vector)
2-element Vector{Float64}:
 1.0
 2.0
```

`VarInfo` with `Metadata`:

```jldoctest
julia> # Just use an example model to construct the `VarInfo` because we're lazy.
       vi = DynamicPPL.untyped_varinfo(DynamicPPL.TestUtils.demo_assume_dot_observe());

julia> vi[@varname(s)] = 1.0; vi[@varname(m)] = 2.0;

julia> # For the sake of brevity, let's just check the type.
       values_as(vi) isa Union{DynamicPPL.Metadata, Vector}
true

julia> values_as(vi, NamedTuple)
(s = 1.0, m = 2.0)

julia> values_as(vi, OrderedDict)
OrderedDict{VarName{sym, typeof(identity)} where sym, Float64} with 2 entries:
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
    eltype(vi::AbstractVarInfo)

Return the `eltype` of the values returned by `vi[:]`.

!!! warning
    This should generally not be called explicitly, as it's only used in
    [`matchingvalue`](@ref) to determine the default type to use in place of
    type-parameters passed to the model.

    This method is considered legacy, and is likely to be deprecated in the future.
"""
function Base.eltype(vi::AbstractVarInfo)
    T = Base.promote_op(getindex, typeof(vi), Colon)
    if T === Union{}
        # In this case `getindex(vi, :)` errors
        # Let us throw a more descriptive error message
        # Ref https://github.com/TuringLang/Turing.jl/issues/2151
        return eltype(vi[:])
    end
    return eltype(T)
end

"""
    has_varnamedvector(varinfo::VarInfo)

Returns `true` if `varinfo` uses `VarNamedVector` as metadata.
"""
has_varnamedvector(vi::AbstractVarInfo) = false

# TODO: Should relax constraints on `vns` to be `AbstractVector{<:Any}` and just try to convert
# the `eltype` to `VarName`? This might be useful when someone does `[@varname(x[1]), @varname(m)]` which
# might result in a `Vector{Any}`.
"""
    subset(varinfo::AbstractVarInfo, vns::AbstractVector{<:VarName})

Subset a `varinfo` to only contain the variables `vns`.

The ordering of variables in the return value will be the same as in `varinfo`.

# Examples
```jldoctest varinfo-subset; setup = :(using Distributions, DynamicPPL)
julia> @model function demo()
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, sqrt(s))
           x = Vector{Float64}(undef, 2)
           x[1] ~ Normal(m, sqrt(s))
           x[2] ~ Normal(m, sqrt(s))
       end
demo (generic function with 2 methods)

julia> model = demo();

julia> varinfo = VarInfo(model);

julia> keys(varinfo)
4-element Vector{VarName}:
 s
 m
 x[1]
 x[2]

julia> for (i, vn) in enumerate(keys(varinfo))
           varinfo[vn] = i
       end

julia> varinfo[[@varname(s), @varname(m), @varname(x[1]), @varname(x[2])]]
4-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0

julia> # Extract one with only `m`.
       varinfo_subset1 = subset(varinfo, [@varname(m),]);


julia> keys(varinfo_subset1)
1-element Vector{VarName{:m, typeof(identity)}}:
 m

julia> varinfo_subset1[@varname(m)]
2.0

julia> # Extract one with both `s` and `x[2]`.
       varinfo_subset2 = subset(varinfo, [@varname(s), @varname(x[2])]);

julia> keys(varinfo_subset2)
2-element Vector{VarName}:
 s
 x[2]

julia> varinfo_subset2[[@varname(s), @varname(x[2])]]
2-element Vector{Float64}:
 1.0
 4.0
```

`subset` is particularly useful when combined with [`merge(varinfo::AbstractVarInfo)`](@ref)

```jldoctest varinfo-subset
julia> # Merge the two.
       varinfo_subset_merged = merge(varinfo_subset1, varinfo_subset2);

julia> keys(varinfo_subset_merged)
3-element Vector{VarName}:
 m
 s
 x[2]

julia> varinfo_subset_merged[[@varname(s), @varname(m), @varname(x[2])]]
3-element Vector{Float64}:
 1.0
 2.0
 4.0

julia> # Merge the two with the original.
       varinfo_merged = merge(varinfo, varinfo_subset_merged);

julia> keys(varinfo_merged)
4-element Vector{VarName}:
 s
 m
 x[1]
 x[2]

julia> varinfo_merged[[@varname(s), @varname(m), @varname(x[1]), @varname(x[2])]]
4-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
```

# Notes

## Type-stability

!!! warning
    This function is only type-stable when `vns` contains only varnames
    with the same symbol. For exmaple, `[@varname(m[1]), @varname(m[2])]` will
    be type-stable, but `[@varname(m[1]), @varname(x)]` will not be.
"""
function subset end

"""
    merge(varinfo, other_varinfos...)

Merge varinfos into one, giving precedence to the right-most varinfo when sensible.

This is particularly useful when combined with [`subset(varinfo, vns)`](@ref).

See docstring of [`subset(varinfo, vns)`](@ref) for examples.
"""
Base.merge(varinfo::AbstractVarInfo) = varinfo
# Define 3-argument version so 2-argument version will error if not implemented.
function Base.merge(
    varinfo1::AbstractVarInfo,
    varinfo2::AbstractVarInfo,
    varinfo3::AbstractVarInfo,
    varinfo_others::AbstractVarInfo...,
)
    return merge(Base.merge(varinfo1, varinfo2), varinfo3, varinfo_others...)
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

# For link!!, invlink!!, link, and invlink, we deliberately do not provide a fallback
# method for the case when no `vns` is provided, that would get all the keys from the
# `VarInfo`. Hence each subtype of `AbstractVarInfo` needs to implement separately the case
# where `vns` is provided and the one where it is not. This is because having separate
# implementations is typically much more performant, and because not all AbstractVarInfo
# types support partial linking.

"""
    link!!([t::AbstractTransformation, ]vi::AbstractVarInfo, model::Model)
    link!!([t::AbstractTransformation, ]vi::AbstractVarInfo, vns::NTuple{N,VarName}, model::Model)

Transform variables in `vi` to their linked space, mutating `vi` if possible.

Either transform all variables, or only ones specified in `vns`.

Use the  transformation `t`, or `default_transformation(model, vi)` if one is not provided.

See also: [`default_transformation`](@ref), [`invlink!!`](@ref).
"""
function link!!(vi::AbstractVarInfo, model::Model)
    return link!!(default_transformation(model, vi), vi, model)
end
function link!!(vi::AbstractVarInfo, vns::VarNameTuple, model::Model)
    return link!!(default_transformation(model, vi), vi, vns, model)
end

"""
    link([t::AbstractTransformation, ]vi::AbstractVarInfo, model::Model)
    link([t::AbstractTransformation, ]vi::AbstractVarInfo, vns::NTuple{N,VarName}, model::Model)

Transform variables in `vi` to their linked space without mutating `vi`.

Either transform all variables, or only ones specified in `vns`.

Use the  transformation `t`, or `default_transformation(model, vi)` if one is not provided.

See also: [`default_transformation`](@ref), [`invlink`](@ref).
"""
function link(vi::AbstractVarInfo, model::Model)
    return link(default_transformation(model, vi), vi, model)
end
function link(vi::AbstractVarInfo, vns::VarNameTuple, model::Model)
    return link(default_transformation(model, vi), vi, vns, model)
end

"""
    invlink!!([t::AbstractTransformation, ]vi::AbstractVarInfo, model::Model)
    invlink!!([t::AbstractTransformation, ]vi::AbstractVarInfo, vns::NTuple{N,VarName}, model::Model)

Transform variables in `vi` to their constrained space, mutating `vi` if possible.

Either transform all variables, or only ones specified in `vns`.

Use the (inverse of) transformation `t`, or `default_transformation(model, vi)` if one is
not provided.

See also: [`default_transformation`](@ref), [`link!!`](@ref).
"""
function invlink!!(vi::AbstractVarInfo, model::Model)
    return invlink!!(default_transformation(model, vi), vi, model)
end
function invlink!!(vi::AbstractVarInfo, vns::VarNameTuple, model::Model)
    return invlink!!(default_transformation(model, vi), vi, vns, model)
end

# Vector-based ones.
function link!!(
    t::StaticTransformation{<:Bijectors.Transform}, vi::AbstractVarInfo, ::Model
)
    b = inverse(t.bijector)
    x = vi[:]
    y, logjac = with_logabsdet_jacobian(b, x)

    lp_new = getlogprior(vi) - logjac
    vi_new = setlogprior!!(unflatten(vi, y), lp_new)
    return settrans!!(vi_new, t)
end

function invlink!!(
    t::StaticTransformation{<:Bijectors.Transform}, vi::AbstractVarInfo, ::Model
)
    b = t.bijector
    y = vi[:]
    x, logjac = with_logabsdet_jacobian(b, y)

    lp_new = getlogprior(vi) + logjac
    vi_new = setlogprior!!(unflatten(vi, x), lp_new)
    return settrans!!(vi_new, NoTransformation())
end

"""
    invlink([t::AbstractTransformation, ]vi::AbstractVarInfo, model::Model)
    invlink([t::AbstractTransformation, ]vi::AbstractVarInfo, vns::NTuple{N,VarName}, model::Model)

Transform variables in `vi` to their constrained space without mutating `vi`.

Either transform all variables, or only ones specified in `vns`.

Use the (inverse of) transformation `t`, or `default_transformation(model, vi)` if one is
not provided.

See also: [`default_transformation`](@ref), [`link`](@ref).
"""
function invlink(vi::AbstractVarInfo, model::Model)
    return invlink(default_transformation(model, vi), vi, model)
end
function invlink(vi::AbstractVarInfo, vns::VarNameTuple, model::Model)
    return invlink(default_transformation(model, vi), vi, vns, model)
end

"""
    maybe_invlink_before_eval!!([t::Transformation,] vi, model)

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
function maybe_invlink_before_eval!!(vi::AbstractVarInfo, model::Model)
    return maybe_invlink_before_eval!!(transformation(vi), vi, model)
end
function maybe_invlink_before_eval!!(::NoTransformation, vi::AbstractVarInfo, model::Model)
    return vi
end
function maybe_invlink_before_eval!!(
    ::DynamicTransformation, vi::AbstractVarInfo, model::Model
)
    # `DynamicTransformation` is meant to _not_ do the transformation statically, hence we
    # do nothing.
    return vi
end
function maybe_invlink_before_eval!!(
    t::StaticTransformation, vi::AbstractVarInfo, model::Model
)
    return invlink!!(t, vi, model)
end

# Utilities
"""
    unflatten(vi::AbstractVarInfo, x::AbstractVector)

Return a new instance of `vi` with the values of `x` assigned to the variables.
"""
function unflatten end

"""
    to_maybe_linked_internal(vi::AbstractVarInfo, vn::VarName, dist, val)

Return reconstructed `val`, possibly linked if `istrans(vi, vn)` is `true`.
"""
function to_maybe_linked_internal(vi::AbstractVarInfo, vn::VarName, dist, val)
    f = to_maybe_linked_internal_transform(vi, vn, dist)
    return f(val)
end

"""
    from_maybe_linked_internal(vi::AbstractVarInfo, vn::VarName, dist, val)

Return reconstructed `val`, possibly invlinked if `istrans(vi, vn)` is `true`.
"""
function from_maybe_linked_internal(vi::AbstractVarInfo, vn::VarName, dist, val)
    f = from_maybe_linked_internal_transform(vi, vn, dist)
    return f(val)
end

"""
    invlink_with_logpdf(varinfo::AbstractVarInfo, vn::VarName, dist[, x])

Invlink `x` and compute the logpdf under `dist` including correction from
the invlink-transformation.

If `x` is not provided, `getindex_internal(vi, vn)` will be used.

!!! warning
    The input value `x` should be according to the internal representation of
    `varinfo`, e.g. the value returned by `getindex_internal(vi, vn)`.
"""
function invlink_with_logpdf(vi::AbstractVarInfo, vn::VarName, dist)
    return invlink_with_logpdf(vi, vn, dist, getindex_internal(vi, vn))
end
function invlink_with_logpdf(vi::AbstractVarInfo, vn::VarName, dist, y)
    f = from_maybe_linked_internal_transform(vi, vn, dist)
    x, logjac = with_logabsdet_jacobian(f, y)
    return x, logpdf(dist, x) + logjac
end

# Legacy code that is currently overloaded for the sake of simplicity.
# TODO: Remove when possible.
increment_num_produce!!(::AbstractVarInfo) = nothing

"""
    from_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from the internal representation of `vn` with `dist`
in `varinfo` to a representation compatible with `dist`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function from_internal_transform end

"""
    from_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from the linked internal representation of `vn` with `dist`
in `varinfo` to a representation compatible with `dist`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function from_linked_internal_transform end

"""
    from_maybe_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from the possibly linked internal representation of `vn` with `dist`n
in `varinfo` to a representation compatible with `dist`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function from_maybe_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)
    return if istrans(varinfo, vn)
        from_linked_internal_transform(varinfo, vn, dist)
    else
        from_internal_transform(varinfo, vn, dist)
    end
end
function from_maybe_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName)
    return if istrans(varinfo, vn)
        from_linked_internal_transform(varinfo, vn)
    else
        from_internal_transform(varinfo, vn)
    end
end

"""
    to_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from a representation compatible with `dist` to the
internal representation of `vn` with `dist` in `varinfo`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function to_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)
    return inverse(from_internal_transform(varinfo, vn, dist))
end
function to_internal_transform(varinfo::AbstractVarInfo, vn::VarName)
    return inverse(from_internal_transform(varinfo, vn))
end

"""
    to_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from a representation compatible with `dist` to the
linked internal representation of `vn` with `dist` in `varinfo`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function to_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)
    return inverse(from_linked_internal_transform(varinfo, vn, dist))
end
function to_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName)
    return inverse(from_linked_internal_transform(varinfo, vn))
end

"""
    to_maybe_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from a representation compatible with `dist` to a
possibly linked internal representation of `vn` with `dist` in `varinfo`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function to_maybe_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)
    return inverse(from_maybe_linked_internal_transform(varinfo, vn, dist))
end
function to_maybe_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName)
    return inverse(from_maybe_linked_internal_transform(varinfo, vn))
end

"""
    internal_to_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)

Return a transformation that transforms from the internal representation of `vn` with `dist`
in `varinfo` to a _linked_ internal representation of `vn` with `dist` in `varinfo`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function internal_to_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)
    f_from_internal = from_internal_transform(varinfo, vn, dist)
    f_to_linked_internal = to_linked_internal_transform(varinfo, vn, dist)
    return f_to_linked_internal ∘ f_from_internal
end
function internal_to_linked_internal_transform(varinfo::AbstractVarInfo, vn::VarName)
    f_from_internal = from_internal_transform(varinfo, vn)
    f_to_linked_internal = to_linked_internal_transform(varinfo, vn)
    return f_to_linked_internal ∘ f_from_internal
end

"""
    linked_internal_to_internal_transform(varinfo::AbstractVarInfo, vn::VarName[, dist])

Return a transformation that transforms from a _linked_ internal representation of `vn` with `dist`
in `varinfo` to the internal representation of `vn` with `dist` in `varinfo`.

If `dist` is not present, then it is assumed that `varinfo` knows the correct output for `vn`.
"""
function linked_internal_to_internal_transform(varinfo::AbstractVarInfo, vn::VarName, dist)
    f_from_linked_internal = from_linked_internal_transform(varinfo, vn, dist)
    f_to_internal = to_internal_transform(varinfo, vn, dist)
    return f_to_internal ∘ f_from_linked_internal
end

function linked_internal_to_internal_transform(varinfo::AbstractVarInfo, vn::VarName)
    f_from_linked_internal = from_linked_internal_transform(varinfo, vn)
    f_to_internal = to_internal_transform(varinfo, vn)
    return f_to_internal ∘ f_from_linked_internal
end
