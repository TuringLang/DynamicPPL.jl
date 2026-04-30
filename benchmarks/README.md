To run the benchmarks locally, run this from the root directory of the repository:

```sh
julia --project=benchmarks benchmarks/benchmarks.jl
```

This prints absolute log-density times and gradient/log-density ratios for a
fixed set of model × AD-backend combinations. Run on each PR by the
`Benchmarking` CI workflow, which posts the resulting table as a comment.
There is no base-vs-head comparison: judge regressions by comparing against
the most recent main-branch run in the comment history.
