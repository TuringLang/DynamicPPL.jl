# [Existing accumulators in DynamicPPL](@id existing-accumulators)

DynamicPPL provides a number of built-in accumulators which can be used as-is; they cover many of the typical use-cases for accumulators.

## Log-probability accumulators

```@docs
LogPriorAccumulator
LogLikelihoodAccumulator
LogJacobianAccumulator
```

There are various convenience functions in DynamicPPL to combine the outputs from these accumulators:

```@docs; canonical=false
getlogjoint
getlogjoint_internal
getlogprior
getlogprior_internal
getloglikelihood
getlogjac
```

## Storing values

```@docs
RawValueAccumulator
VectorValueAccumulator
```

and their associated convenience functions:

```@docs
get_raw_values
get_vector_values
```

## Miscellaneous

```@docs
PriorDistributionAccumulator
```

(No convenience function for this one yet.)
