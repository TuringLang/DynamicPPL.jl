To run the benchmarks, simply do:
```sh
julia --project -e 'using DynamicPPLBenchmarks; weave_benchmarks();'
```

```julia
help?> weave_benchmarks
search: weave_benchmarks

  weave_benchmarks(input="benchmarks.jmd"; kwargs...)

  Weave benchmarks present in benchmarks.jmd into a single file.

  Keyword arguments
  ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

    •  benchmarkbody: JMD-file to be rendered for each model.

    •  include_commit_id=false: specify whether to include commit-id in the default name.

    •  name: the name of directory in results/ to use as output directory.

    •  name_old=nothing: if specified, comparisons of current run vs. the run pinted to by name_old
       will be included in the generated document.

    •  include_typed_code=false: if true, output of code_typed for the evaluator of the model will be
       included in the weaved document.

    •  Rest of the passed kwargs will be passed on to Weave.weave.
```
