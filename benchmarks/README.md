To run the benchmarks, simply do:
```sh
julia --project -e 'using Weave; Weave.weave("benchmarks.jmd", doctype="github", args=Dict(:benchmarkbody => "benchmark_body.jmd"));'
```

Furthermore:
- If you want to save the output of `code_typed` for the evaluator of the different models, add a `:prefix => "myprefix"` to the `args`.
- If `:prefix_old` is specified in `args`, a `diff` of the `code_typed` loaded using `:prefix_old` and the output of `code_typed` for the current run will be included in the weaved document.
