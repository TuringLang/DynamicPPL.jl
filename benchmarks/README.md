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

Rows marked `*` have `t(logdensity)` below about 100 ns; their ratios can be
dominated by timer floor, fixed overhead, and run-to-run variation. For those
rows, raw `t(grad)` is more meaningful than `t(grad)/t(logdensity)`. These
microbenchmarks can also vary noticeably across runs and machines.

The CI comment shows the PR head table first and, when available, includes a
collapsed `main` table for comparison. Treat the numbers as approximate and use
the `main` table to spot likely regressions.

## PosteriorDB comparison

`posteriordb.jl` compares DynamicPPL with the matching Stan implementation for
all 147 PosteriorDB posteriors. The model catalog and Stan-to-DynamicPPL
coordinate mapping are vendored in this directory, so the benchmark does not
depend on a sibling checkout or on Turing.

The translations preserve each posterior but may use algebraic reductions or
sufficient statistics that differ from the Stan implementation. The Stan and
Turing columns therefore compare complete implementations, not identical
instruction sequences.

Run one model or the full catalog from the repository root:

```sh
julia --project=benchmarks benchmarks/posteriordb.jl \
    eight_schools-eight_schools_centered
julia --project=benchmarks benchmarks/posteriordb.jl
```

BridgeStan uses `BRIDGESTAN_PATH` when it is set and otherwise uses its default
installation. The full command measures ForwardDiff, Enzyme, and Mooncake and
regenerates `benchmarks/posteriordb.md`. Use `--logdensity-only` to omit
gradients, `--stan-only` or `--turing-only` to time only one implementation,
or `--mooncake-only` to omit ForwardDiff and Enzyme. `--turing-only` still uses
BridgeStan to select the matched parameter realization and map coordinates.
Runs that use local DynamicPPL or Mooncake checkouts require clean tracked files
because the report records their source revisions.
The `--logdensity-only` mode composes with either implementation-only mode. To
choose another Markdown path and also save the raw timings as tab-separated
data:

```sh
PDB_BENCH_MARKDOWN=benchmarks/posteriordb-results.md \
PDB_BENCH_OUTPUT=benchmarks/posteriordb-results.tsv \
julia --project=benchmarks benchmarks/posteriordb.jl
```

The full run can be split across independent processes with
`PDB_BENCH_SHARDS` and the one-based `PDB_BENCH_SHARD`. For example, set the
former to `16` and launch one process for each shard index from 1 through 16.
Each shard should set `PDB_BENCH_OUTPUT` to a distinct TSV path. Merge those
checkpoints and regenerate `posteriordb.md` with:

```sh
julia --project=benchmarks benchmarks/posteriordb.jl --merge shard-*.tsv
```

The merge rejects incompatible run metadata, duplicate or missing shards, and
incomplete model coverage.

Use `--verify` to compare relative log density and gradients with Stan at 30
reproducible random points in Stan's unconstrained parameter space. The density
comparison allows a parameter-independent normalization offset. Pass model
names for a focused check, or use the same shard variables for the full catalog:

```sh
julia --project=benchmarks benchmarks/posteriordb.jl --verify \
    eight_schools-eight_schools_centered
```

`PDB_TEST_DRAWS`, `PDB_TEST_SCALE`, and `PDB_TEST_SEED` override the verifier's
defaults of 30, 0.2, and 468.
