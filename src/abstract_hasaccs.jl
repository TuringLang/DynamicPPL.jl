#######################################
# Mandatory AbstractHasAccs interface #
#######################################

"""
    getaccs(aha::AbstractHasAccs)

Return the `AccumulatorTuple` of `aha`.

This should be implemented by each subtype of `AbstractHasAccs`.
"""
function getaccs end

"""
    setaccs!!(aha::AbstractHasAccs, accs::AccumulatorTuple)

Update the `AccumulatorTuple` of `aha` to `accs`, mutating if it makes sense.

This should be implemented by each subtype of `AbstractHasAccs`.
"""
function setaccs!! end

####################################################################
# Derived AbstractHasAccs interface                                #
# Everything below this line derives from the two functions above, #
# plus the AccumulatorTuple interface.                             #
####################################################################

"""
    setacc!!(aha::AbstractHasAccs, acc::AbstractAccumulator)

Add `acc` to the `AccumulatorTuple` of `aha`, mutating if it makes sense.

If an accumulator with the same [`accumulator_name`](@ref) already exists, it will be
replaced.
"""
function setacc!!(aha::AbstractHasAccs, acc::AbstractAccumulator)
    return setaccs!!(aha, setacc!!(getaccs(aha), acc))
end

"""
    deleteacc!!(aha::AccumulatorTUple, ::Val{accname})

Delete the accumulator with name `accname` from `aha`.
"""
function deleteacc!!(aha::AbstractHasAccs, accname::Val)
    return setaccs!!(aha, deleteacc!!(getaccs(aha), accname))
end

"""
    setaccs!!(aha::AbstractHasAccs, accs::NTuple{N,AbstractAccumulator}) where {N}

Update the `AccumulatorTuple` of `aha` to `accs`, mutating if it makes sense.
"""
function setaccs!!(aha::AbstractHasAccs, accs::NTuple{N,AbstractAccumulator}) where {N}
    return setaccs!!(aha, AccumulatorTuple(accs))
end

"""
    getacc(aha::AbstractHasAccs, ::Val{accname})

Return the `AbstractAccumulator` of `aha` with name `accname`.
"""
function getacc(aha::AbstractHasAccs, accname::Val)
    return getacc(getaccs(aha), accname)
end
function getacc(::AbstractHasAccs, ::Symbol)
    return error(
        """
        The method getacc(aha::AbstractHasAccs, accname::Symbol) does not exist. For type
        stability reasons use getacc(aha::AbstractHasAccs, Val(accname)) instead.
        """
    )
end

"""
    hasacc(aha::AbstractHasAccs, ::Val{accname}) where {accname}

Return a boolean indicating whether `aha` has an accumulator with name `accname`.
"""
hasacc(aha::AbstractHasAccs, accname::Val) = haskey(getaccs(aha), accname)
function hasacc(::AbstractHasAccs, ::Symbol)
    return error(
        """
        The method hasacc(aha::AbstractHasAccs, accname::Symbol) does not exist. For type
        stability reasons use hasacc(aha::AbstractHasAccs, Val(accname)) instead.
        """
    )
end

"""
    acckeys(aha::AbstractHasAccs)

Return the names of the accumulators in `aha`.
"""
acckeys(aha::AbstractHasAccs) = keys(getaccs(aha))

"""
    getlogjoint(aha::AbstractHasAccs)

Return the log-joint probability stored in `aha`.

See also: [`getlogprior`](@ref), [`getloglikelihood`](@ref).
"""
getlogjoint(aha::AbstractHasAccs) = getlogprior(aha) + getloglikelihood(aha)

"""
    getlogjoint_internal(aha::AbstractHasAccs)

Return the log-joint probability of the parameters in `aha`, including the log-Jacobian for
any parameters that were transformed.

In general, we have that:

```julia
getlogjoint_internal(aha) == getlogjoint(aha) - getlogjac(aha)
```
"""
getlogjoint_internal(aha::AbstractHasAccs) =
    getlogprior(aha) + getloglikelihood(aha) - getlogjac(aha)

"""
    getlogp(aha::AbstractHasAccs)

Return a NamedTuple of the log prior, log Jacobian, and log likelihood probabilities.

The keys are called `logprior`, `logjac`, and `loglikelihood`. If any of the necessary
accumulators are not present in `aha` an error will be thrown.
"""
function getlogp(aha::AbstractHasAccs)
    return (;
        logprior=getlogprior(aha),
        logjac=getlogjac(aha),
        loglikelihood=getloglikelihood(aha),
    )
end

"""
    getlogprior(aha::AbstractHasAccs)

Return the log-prior probability of the parameters in `aha`.

See also: [`getlogjoint`](@ref), [`getloglikelihood`](@ref), [`setlogprior!!`](@ref).
"""
getlogprior(aha::AbstractHasAccs) = getacc(aha, Val(:LogPrior)).logp

"""
    getlogprior_internal(aha::AbstractHasAccs)

Return the log-prior probability of the parameters, including the log-Jacobian for any
parameters that were transformed.

In general, we have that:

```julia
getlogprior_internal(aha) == getlogprior(aha) - getlogjac(aha)
```
"""
getlogprior_internal(aha::AbstractHasAccs) = getlogprior(aha) - getlogjac(aha)

"""
    getlogjac(aha::AbstractHasAccs)

Return the accumulated log-Jacobian term for any linked parameters in `aha`. The Jacobian
here is taken with respect to the forward transform (i.e. from the original space to the
transformed space).

See also: [`setlogjac!!`](@ref).
"""
getlogjac(aha::AbstractHasAccs) = getacc(aha, Val(:LogJacobian)).logjac

"""
    getloglikelihood(aha::AbstractHasAccs)

Return the log-likelihood of the observed data in `aha`.

See also: [`getlogjoint`](@ref), [`getlogprior`](@ref), [`setloglikelihood!!`](@ref).
"""
getloglikelihood(aha::AbstractHasAccs) = getacc(aha, Val(:LogLikelihood)).logp

"""
    setlogprior!!(aha::AbstractHasAccs, logp)

Set the log-prior probability of the parameters sampled in `aha` to `logp`.

See also: [`setloglikelihood!!`](@ref), [`setlogp!!`](@ref), [`getlogprior`](@ref).
"""
setlogprior!!(aha::AbstractHasAccs, logp) = setacc!!(aha, LogPriorAccumulator(logp))

"""
    setlogjac!!(aha::AbstractHasAccs, logjac)

Set the accumulated log-Jacobian term for any transformed parameters in `aha`. The Jacobian
here is taken with respect to the forward transform (i.e. from the original space to the
transformed space).

See also: [`getlogjac`](@ref), [`acclogjac!!`](@ref).
"""
setlogjac!!(aha::AbstractHasAccs, logjac) = setacc!!(aha, LogJacobianAccumulator(logjac))

"""
    setloglikelihood!!(aha::AbstractHasAccs, logp)

Set the log of the likelihood probability of the observed data sampled in `aha` to `logp`.

See also: [`setlogprior!!`](@ref), [`setlogp!!`](@ref), [`getloglikelihood`](@ref).
"""
setloglikelihood!!(aha::AbstractHasAccs, logp) =
    setacc!!(aha, LogLikelihoodAccumulator(logp))

"""
    setlogp!!(aha::AbstractHasAccs, logp::NamedTuple)

Set both the log prior and the log likelihood probabilities in `aha`.

`logp` should have fields `logprior` and `loglikelihood` and no other fields.

See also: [`setlogprior!!`](@ref), [`setloglikelihood!!`](@ref), [`getlogp`](@ref).
"""
function setlogp!!(aha::AbstractHasAccs, logp::NamedTuple{names}) where {names}
    if Set(names) != Set([:logprior, :logjac, :loglikelihood])
        error(
            "The second argument to `setlogp!!` must be a NamedTuple with the fields logprior, logjac, and loglikelihood.",
        )
    end
    aha = setlogprior!!(aha, logp.logprior)
    aha = setlogjac!!(aha, logp.logjac)
    aha = setloglikelihood!!(aha, logp.loglikelihood)
    return aha
end

function setlogp!!(::AbstractHasAccs, ::Number)
    return error(
        """
        `setlogp!!(aha::AbstractHasAccs, logp::Number)` is no longer supported. Use
        `setloglikelihood!!`, `setlogjac!!`, and/or `setlogprior!!` instead.
        """
    )
end

"""
    DynamicPPL.is_extracting_colon_eq_values(aha::AbstractHasAccs)

Return `true` if `aha` contains a `RawValueAccumulator` that is extracting values on the LHS
of `:=`. This function is used to determine whether `store_colon_eq!!` should be called when
encountering a `x := expr` statement in the model body.

Note that this function is not public.
"""
function is_extracting_colon_eq_values(aha::AbstractHasAccs)
    return hasacc(aha, Val(RAW_VALUE_ACCNAME)) &&
           getacc(aha, Val(RAW_VALUE_ACCNAME)).f.include_colon_eq
end

"""
    get_raw_values(aha::AbstractHasAccs)

Extract a `VarNamedTuple` of values from the `RawValueAccumulator` in `aha`. This is the
'raw' values as they are seen in the model, without any transformations applied to them.

If `aha` does not contain a `RawValueAccumulator`, this function will throw an error.
"""
get_raw_values(aha::AbstractHasAccs) = getacc(aha, Val(RAW_VALUE_ACCNAME)).values
