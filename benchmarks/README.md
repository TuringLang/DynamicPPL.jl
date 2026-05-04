Run from the repository root:

```sh
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks benchmarks/benchmarks.jl
```

The `Benchmarking` CI workflow runs this on each PR and posts the table as a
comment. There is no base-vs-head comparison: judge regressions by comparing
against the most recent main-branch run in the comment history.
