# Benchmarks

Run from the repository root:

```sh
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks benchmarks/benchmarks.jl
```

The `Benchmarking` CI workflow runs this on each PR and posts the table as a
comment.

## Interpreting results

Each row times one of DynamicPPL's reference models. `Dim` is the parameter
count. `Linked` is `true` when parameters have been mapped to unconstrained
space. `t(logdensity)` is the wall-clock time for one log-density evaluation.

The AD backend columns are performance ratios: each value is the gradient time
divided by `t(logdensity)`. For example, a value of `10` means computing the
gradient takes 10 times as long as evaluating the log-density. Lower is better.
`err` means the backend errored on that model.

If `t(logdensity)` is below about 100 ns, ratios are often dominated by timer
floor and fixed overhead. For those rows, raw `t(grad)` is more meaningful than
`t(grad)/t(logdensity)`. These microbenchmarks can also vary noticeably across
runs.

The CI comment shows the PR head table first and, when available, includes a
collapsed `main` table for comparison. Treat the numbers as approximate and use
the `main` table to spot likely regressions.
