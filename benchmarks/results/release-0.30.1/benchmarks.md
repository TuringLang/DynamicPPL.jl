
## release-0.30.1 ##

### Setup

```julia
using BenchmarkTools, DynamicPPL, Distributions, Serialization
```


```julia
using DynamicPPLBenchmarks
using DynamicPPLBenchmarks: time_model_def, make_suite, typed_code, weave_child
```




### Environment


Computer Information:
```
Julia Version 1.10.5
Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 16 × 12th Gen Intel(R) Core(TM) i5-12500H
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, alderlake)
Threads: 1 default, 0 interactive, 1 GC (on 16 virtual cores)

```

Package Information:

```
Project DynamicPPLBenchmarks v0.1.0
Status `~/.julia/dev/DynamicPPL/benchmarks/Project.toml`
  [7a57a42e] AbstractPPL v0.9.0
  [6e4b80f9] BenchmarkTools v1.5.0
⌃ [b0b7db55] ComponentArrays v0.15.17
  [8294860b] DiffUtils v0.1.0
⌃ [31c24e10] Distributions v0.25.112
  [634d3b9d] DrWatson v2.18.0
⌃ [366bfd00] DynamicPPL v0.30.1
  [08abe8d2] PrettyTables v2.4.0
  [bd369af6] Tables v1.12.0
  [44d3d7a6] Weave v0.10.12
  [b77e0a4c] InteractiveUtils
  [76f85450] LibGit2
  [d6f4376e] Markdown
  [44cfe95a] Pkg v1.10.0
  [9a3f8284] Random
Info Packages marked with ⌃ have new versions available and may be upgradable.

```



### Models

#### `demo1`

```julia
@model function demo1(x)
    m ~ Normal()
    x ~ Normal(m, 1)

    return (m = m, x = x)
end

model_def = demo1;
data = (1.0,);
```



```julia
@time model_def(data...)();
```

```
0.917790 seconds (1.17 M allocations: 79.641 MiB, 99.97% compilation time
)
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000020 seconds (2 allocations: 32 bytes)
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  1.052 μs … 158.020 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     1.136 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   1.247 μs ±   2.599 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▁█▆▅▅▂▂▁                                                    
  ▁█████████▆▆▅▄▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  1.05 μs         Histogram: frequency by time        1.79 μs <

 Memory estimate: 1.47 KiB, allocs estimate: 30.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  209.000 ns …  29.072 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     267.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   283.939 ns ± 349.661 ns  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

   ▁█                                                            
  ▁██▆▄▄▄▄▄▃▂▃▅▃▃▂▃▃▃▃▃▄▃▃▆▇▅▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  209 ns           Histogram: frequency by time          508 ns <

 Memory estimate: 160 bytes, allocs estimate: 3.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  56.000 ns …  57.045 μs  ┊ GC (min … max): 0.00% … 0.00
%
 Time  (median):     74.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   79.286 ns ± 569.832 ns  ┊ GC (mean ± σ):  0.00% ± 0.00
%

                       ▃▇▅ ▅█▄ ▂                                
  ▂▃▃▄▁▄▄▄▁▄▅▅▁▅▆▇▁▇██▁███▁███▁██▆▁▃▃▂▁▂▂▂▁▂▂▂▁▂▂▂▁▂▂▂▁▂▂▂▁▂▂▂ ▃
  56 ns           Histogram: frequency by time          101 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  962.000 ns … 192.163 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):       1.201 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.334 μs ±   3.102 μs  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

            ▂█▁ ▂▁                                               
  ▃▃▃▃▃▃▃▃▄▄██████▆▆▅▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂ ▃
  962 ns           Histogram: frequency by time         2.05 μs <

 Memory estimate: 704 bytes, allocs estimate: 26.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  54.000 ns … 372.000 ns  ┊ GC (min … max): 0.00% … 0.00
%
 Time  (median):     59.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   59.742 ns ±   7.311 ns  ┊ GC (mean ± σ):  0.00% ± 0.00
%

           ▃  █  ▇  ▄                                           
  ▂▁▁▂▁▁▅▁▁█▁▁█▁▁█▁▁█▁▁▁▇▁▁▄▁▁▃▁▁▂▁▁▂▁▁▂▁▁▁▂▁▁▃▁▁▃▁▁▂▁▁▂▁▁▂▁▁▂ ▃
  54 ns           Histogram: frequency by time           73 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```




#### `demo2`

```julia
@model function demo2(y)
    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)

    # The number of observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end

    return (; p)
end

model_def = demo2;
data = (rand(0:1, 10),);
```



```julia
@time model_def(data...)();
```

```
0.603370 seconds (637.06 k allocations: 42.818 MiB, 99.96% compilation ti
me)
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000010 seconds (1 allocation: 16 bytes)
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  2.899 μs …  1.229 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.172 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.411 μs ± 30.753 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▄▇█▆▃▁                                                   
  ▂▄███████▆▅▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▂▂▂▁▂▂▂ ▃
  2.9 μs         Histogram: frequency by time        5.41 μs <

 Memory estimate: 3.47 KiB, allocs estimate: 67.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  308.000 ns … 355.299 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     334.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   385.875 ns ±   3.560 μs  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

   ▂▅▇██▇▅▄▄▄▂▁                                                 ▂
  ▅██████████████▇▇▇▆▇▇▇▇▇▇▆▆▅▆▅▅▆▆▆▇▆▄▅▄▄▅▅▃▄▃▄▅▁▃▅▅▄▄▃▅▁▄▄▄▄▅ █
  308 ns        Histogram: log(frequency) by time        609 ns <

 Memory estimate: 160 bytes, allocs estimate: 3.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  153.000 ns …  16.333 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     160.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   164.726 ns ± 220.698 ns  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

                 ▃   ▇  █   █   █   ▅                            
  ▂▁▁▂▁▁▁▃▁▁▁▆▁▁▁█▁▁▁█▁▁█▁▁▁█▁▁▁█▁▁▁█▁▁▁█▁▁▆▁▁▁▄▁▁▁▃▁▁▁▃▁▁▁▃▁▁▂ ▃
  153 ns           Histogram: frequency by time          169 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  2.468 μs … 124.036 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.649 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.901 μs ±   2.990 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▄██▃                                                        
  ▃█████▆▅▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▂▁▂▂ ▃
  2.47 μs         Histogram: frequency by time        5.34 μs <

 Memory estimate: 1.42 KiB, allocs estimate: 60.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  133.000 ns …   8.961 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     147.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   149.194 ns ± 116.622 ns  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

                           ▃  ▅ ▇ █  █ ▆ ▅  ▁                    
  ▂▁▂▁▂▁▁▂▁▃▁▃▁▁▄▁▅▁▆▁▁▇▁█▁█▁▁█▁█▁█▁▁█▁█▁█▁▁█▁▇▁▄▁▁▃▁▃▁▂▁▁▂▁▂▁▂ ▃
  133 ns           Histogram: frequency by time          159 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```




#### `demo3`

```julia
@model function demo3(x)
    D, N = size(x)

    # Draw the parameters for cluster 1.
    μ1 ~ Normal()

    # Draw the parameters for cluster 2.
    μ2 ~ Normal()

    μ = [μ1, μ2]

    # Comment out this line if you instead want to draw the weights.
    w = [0.5, 0.5]

    # Draw assignments for each datum and generate it from a multivariate normal.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ Categorical(w)
        x[:, i] ~ MvNormal([μ[k[i]], μ[k[i]]], 1.0)
    end

    return (; μ1, μ2, k)
end

model_def = demo3

# Construct 30 data points for each cluster.
N = 30

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]

# Construct the data points.
data = (mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.0), N), hcat, 1:2),);
```



```julia
@time model_def(data...)();
```

```
1.297892 seconds (1.48 M allocations: 100.038 MiB, 99.83% compilation tim
e)
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000010 seconds (1 allocation: 16 bytes)
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```
Error: ArgumentError: invalid index: 1.0 of type Float64
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  2.899 μs …  1.229 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.172 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.411 μs ± 30.753 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▄▇█▆▃▁                                                   
  ▂▄███████▆▅▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▂▂▂▁▂▂▂ ▃
  2.9 μs         Histogram: frequency by time        5.41 μs <

 Memory estimate: 3.47 KiB, allocs estimate: 67.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  308.000 ns … 355.299 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     334.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   385.875 ns ±   3.560 μs  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

   ▂▅▇██▇▅▄▄▄▂▁                                                 ▂
  ▅██████████████▇▇▇▆▇▇▇▇▇▇▆▆▅▆▅▅▆▆▆▇▆▄▅▄▄▅▅▃▄▃▄▅▁▃▅▅▄▄▃▅▁▄▄▄▄▅ █
  308 ns        Histogram: log(frequency) by time        609 ns <

 Memory estimate: 160 bytes, allocs estimate: 3.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  153.000 ns …  16.333 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     160.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   164.726 ns ± 220.698 ns  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

                 ▃   ▇  █   █   █   ▅                            
  ▂▁▁▂▁▁▁▃▁▁▁▆▁▁▁█▁▁▁█▁▁█▁▁▁█▁▁▁█▁▁▁█▁▁▁█▁▁▆▁▁▁▄▁▁▁▃▁▁▁▃▁▁▁▃▁▁▂ ▃
  153 ns           Histogram: frequency by time          169 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  2.468 μs … 124.036 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.649 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.901 μs ±   2.990 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▄██▃                                                        
  ▃█████▆▅▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▂▁▂▂ ▃
  2.47 μs         Histogram: frequency by time        5.34 μs <

 Memory estimate: 1.42 KiB, allocs estimate: 60.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  133.000 ns …   8.961 μs  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     147.000 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   149.194 ns ± 116.622 ns  ┊ GC (mean ± σ):  0.00% ± 0.0
0%

                           ▃  ▅ ▇ █  █ ▆ ▅  ▁                    
  ▂▁▂▁▂▁▁▂▁▃▁▃▁▁▄▁▅▁▆▁▁▇▁█▁█▁▁█▁█▁█▁▁█▁█▁█▁▁█▁▇▁▄▁▁▃▁▃▁▂▁▁▂▁▂▁▂ ▃
  133 ns           Histogram: frequency by time          159 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```




#### `demo4`: lots of variables

```julia
@model function demo4_1k(::Type{TV}=Vector{Float64}) where {TV}
    m ~ Normal()
    x = TV(undef, 1_000)
    for i in eachindex(x)
        x[i] ~ Normal(m, 1.0)
    end

    return (; m, x)
end

model_def = demo4_1k
data = ();
```



```julia
@time model_def(data...)();
```

```
0.650808 seconds (1.37 M allocations: 51.395 MiB, 52.80% compilation time
)
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000008 seconds
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  429.839 μs … 209.568 ms  ┊ GC (min … max):  0.00% … 99
.63%
 Time  (median):     494.701 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   613.034 μs ±   2.157 ms  ┊ GC (mean ± σ):  13.40% ± 10
.47%

  █▇▅▄▃▂▁                                                       ▂
  ████████▇▇▆▆▅▅▅▅▅▁▃▁▄▅▄▃▃▃▁▃▃▁▁▁▁▁▁▃▁▁▁▁▃▁▃▁▁▁▁▁▁▁▃▃▄▃▁▅▅▅▆▅▆ █
  430 μs        Histogram: log(frequency) by time       4.39 ms <

 Memory estimate: 689.03 KiB, allocs estimate: 16025.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   92.371 μs … 269.236 ms  ┊ GC (min … max):  0.00% … 99
.92%
 Time  (median):     127.332 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   167.268 μs ±   2.697 ms  ┊ GC (mean ± σ):  18.58% ±  2
.90%

   ▃█                                                            
  ▄██▆▆▃▂▂▂▂▂▇▄▂▂▂▂▂▃▄▄▂▂▂▃▄▃▄▅▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  92.4 μs          Histogram: frequency by time          284 μs <

 Memory estimate: 102.28 KiB, allocs estimate: 2005.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  4.406 μs … 167.457 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     7.073 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   7.227 μs ±   3.472 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

           ▂ ▂█▂                                               
  ▄▄▃▃▅▄▂▂██▆████▆▄▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  4.41 μs         Histogram: frequency by time        17.5 μs <

 Memory estimate: 7.94 KiB, allocs estimate: 1.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  173.077 μs … 252.426 ms  ┊ GC (min … max):  0.00% … 99
.81%
 Time  (median):     233.252 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   330.108 μs ±   2.589 ms  ┊ GC (mean ± σ):  22.35% ±  8
.10%

      ▂██▂                                                       
  ▄▄▃▅█████▅▄▄▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂ ▃
  173 μs           Histogram: frequency by time          723 μs <

 Memory estimate: 352.20 KiB, allocs estimate: 14019.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  236.053 μs … 86.194 ms  ┊ GC (min … max):  0.00% … 99.
50%
 Time  (median):     309.271 μs              ┊ GC (median):     0.00%
 Time  (mean ± σ):   418.100 μs ±  1.053 ms  ┊ GC (mean ± σ):  13.35% ±  7.
80%

  ▂▁██▇▅▄▄▃▃▂▂▃▂▂▂▃▃▂▂▁                                        ▂
  █████████████████████████▇▇▇▇▇▆▅▄▆▅▆▄▅▇▇▅▆▃▅▃▅▃▅▆▄▃▄▁▄▅▃▃▄▄▄ █
  236 μs        Histogram: log(frequency) by time      1.43 ms <

 Memory estimate: 375.45 KiB, allocs estimate: 15507.
```


```julia
@model function demo4_10k(::Type{TV}=Vector{Float64}) where {TV}
    m ~ Normal()
    x = TV(undef, 10_000)
    for i in eachindex(x)
        x[i] ~ Normal(m, 1.0)
    end

    return (; m, x)
end

model_def = demo4_10k
data = ();
```



```julia
@time model_def(data...)();
```

```
32.613410 seconds (100.35 M allocations: 2.631 GiB, 1.81% gc time, 0.42% c
ompilation time)
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000003 seconds
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 1741 samples with 1 evaluation.
 Range (min … max):  4.707 ms … 207.078 ms  ┊ GC (min … max):  0.00% … 95.5
9%
 Time  (median):     5.573 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   6.425 ms ±   5.103 ms  ┊ GC (mean ± σ):  10.57% ± 14.1
7%

     ▆█▃▁                                                      
  ▅▅▆█████▆▄▄▃▄▃▃▄▃▃▂▁▂▂▁▂▂▁▂▂▂▁▂▂▂▂▂▃▃▂▃▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  4.71 ms         Histogram: frequency by time          12 ms <

 Memory estimate: 6.72 MiB, allocs estimate: 160125.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 7634 samples with 1 evaluation.
 Range (min … max):  849.474 μs … 273.128 ms  ┊ GC (min … max):  0.00% … 99
.42%
 Time  (median):       1.073 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.290 ms ±   3.206 ms  ┊ GC (mean ± σ):  10.28% ± 11
.14%

    █                                                            
  ▃▇█▆▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▂▂▂▂▂▂▂▂▂▂▂ ▂
  849 μs           Histogram: frequency by time         6.01 ms <

 Memory estimate: 1016.11 KiB, allocs estimate: 20008.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  50.552 μs … 173.631 ms  ┊ GC (min … max):  0.00% … 99.
85%
 Time  (median):     54.333 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   87.992 μs ±   1.738 ms  ┊ GC (mean ± σ):  21.66% ±  3.
20%

  ██▅▃▂▂▂▁▂  ▂▅▅▄▃▂▂▁ ▁▁▃▃▃▂      ▁▁          ▁▁               ▂
  ██████████▇████████████████▇▇███████▇▇▇▇▇▇▆▇███▇▇▃▆▅▆▄▅▅▆▅▅▆ █
  50.6 μs       Histogram: log(frequency) by time       156 μs <

 Memory estimate: 78.17 KiB, allocs estimate: 2.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 3189 samples with 1 evaluation.
 Range (min … max):  2.228 ms … 207.975 ms  ┊ GC (min … max):  0.00% … 98.4
0%
 Time  (median):     2.474 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.094 ms ±   4.005 ms  ┊ GC (mean ± σ):  16.85% ± 18.3
2%

  ▅█▇▄▃▂▂▁ ▁▁                                                 ▁
  ████████████▆▆▅▅▄▁▅▁▅▅▅▁▃▃▃▁▁▁▃▁▁▁▁▃▁▁▁▁▃▆▁▆█▇▇▇▇███▇▇▆▆▆▆▆ █
  2.23 ms      Histogram: log(frequency) by time      9.54 ms <

 Memory estimate: 3.43 MiB, allocs estimate: 140069.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 2750 samples with 1 evaluation.
 Range (min … max):  2.661 ms … 196.827 ms  ┊ GC (min … max):  0.00% … 97.9
4%
 Time  (median):     2.921 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.588 ms ±   4.066 ms  ┊ GC (mean ± σ):  15.38% ± 17.9
5%

  ▅█▇▅▃▂▂▁ ▁  ▁                                      ▁        ▁
  ██████████▇███▇▅▃▄▄▅▃▃▁▅▁▁▃▅▁▁▁▁▁▁▃▁▁▁▁▁▁▁▇▆▆█▇█▇▇██▇█▆▇▆▆▅ █
  2.66 ms      Histogram: log(frequency) by time      9.82 ms <

 Memory estimate: 3.73 MiB, allocs estimate: 159519.
```


```julia
@model function demo4_100k(::Type{TV}=Vector{Float64}) where {TV}
    m ~ Normal()
    x = TV(undef, 100_000)
    for i in eachindex(x)
        x[i] ~ Normal(m, 1.0)
    end

    return (; m, x)
end

model_def = demo4_100k
data = ();
```



```julia
@time model_def(data...)();
```

```
Error: InterruptException:
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000002 seconds
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```
Error: InterruptException:
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 1741 samples with 1 evaluation.
 Range (min … max):  4.707 ms … 207.078 ms  ┊ GC (min … max):  0.00% … 95.5
9%
 Time  (median):     5.573 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   6.425 ms ±   5.103 ms  ┊ GC (mean ± σ):  10.57% ± 14.1
7%

     ▆█▃▁                                                      
  ▅▅▆█████▆▄▄▃▄▃▃▄▃▃▂▁▂▂▁▂▂▁▂▂▂▁▂▂▂▂▂▃▃▂▃▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  4.71 ms         Histogram: frequency by time          12 ms <

 Memory estimate: 6.72 MiB, allocs estimate: 160125.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 7634 samples with 1 evaluation.
 Range (min … max):  849.474 μs … 273.128 ms  ┊ GC (min … max):  0.00% … 99
.42%
 Time  (median):       1.073 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.290 ms ±   3.206 ms  ┊ GC (mean ± σ):  10.28% ± 11
.14%

    █                                                            
  ▃▇█▆▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▂▂▂▂▂▂▂▂▂▂▂ ▂
  849 μs           Histogram: frequency by time         6.01 ms <

 Memory estimate: 1016.11 KiB, allocs estimate: 20008.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  50.552 μs … 173.631 ms  ┊ GC (min … max):  0.00% … 99.
85%
 Time  (median):     54.333 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   87.992 μs ±   1.738 ms  ┊ GC (mean ± σ):  21.66% ±  3.
20%

  ██▅▃▂▂▂▁▂  ▂▅▅▄▃▂▂▁ ▁▁▃▃▃▂      ▁▁          ▁▁               ▂
  ██████████▇████████████████▇▇███████▇▇▇▇▇▇▆▇███▇▇▃▆▅▆▄▅▅▆▅▅▆ █
  50.6 μs       Histogram: log(frequency) by time       156 μs <

 Memory estimate: 78.17 KiB, allocs estimate: 2.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 3189 samples with 1 evaluation.
 Range (min … max):  2.228 ms … 207.975 ms  ┊ GC (min … max):  0.00% … 98.4
0%
 Time  (median):     2.474 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.094 ms ±   4.005 ms  ┊ GC (mean ± σ):  16.85% ± 18.3
2%

  ▅█▇▄▃▂▂▁ ▁▁                                                 ▁
  ████████████▆▆▅▅▄▁▅▁▅▅▅▁▃▃▃▁▁▁▃▁▁▁▁▃▁▁▁▁▃▆▁▆█▇▇▇▇███▇▇▆▆▆▆▆ █
  2.23 ms      Histogram: log(frequency) by time      9.54 ms <

 Memory estimate: 3.43 MiB, allocs estimate: 140069.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 2750 samples with 1 evaluation.
 Range (min … max):  2.661 ms … 196.827 ms  ┊ GC (min … max):  0.00% … 97.9
4%
 Time  (median):     2.921 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.588 ms ±   4.066 ms  ┊ GC (mean ± σ):  15.38% ± 17.9
5%

  ▅█▇▅▃▂▂▁ ▁  ▁                                      ▁        ▁
  ██████████▇███▇▅▃▄▄▅▃▃▁▅▁▁▃▅▁▁▁▁▁▁▃▁▁▁▁▁▁▁▇▆▆█▇█▇▇██▇█▆▇▆▆▅ █
  2.66 ms      Histogram: log(frequency) by time      9.82 ms <

 Memory estimate: 3.73 MiB, allocs estimate: 159519.
```




#### `demo4_dotted`: `.~` for large number of variables

```julia
@model function demo4_100k_dotted(::Type{TV}=Vector{Float64}) where {TV}
    m ~ Normal()
    x = TV(undef, 100_000)
    x .~ Normal(m, 1.0)

    return (; m, x)
end

model_def = demo4_100k_dotted
data = ();
```



```julia
@time model_def(data...)();
```

```
Error: InterruptException:
```

```julia
m = time_model_def(model_def, data...);
```

```
0.000003 seconds
```

```julia
suite = make_suite(m);
results = run(suite; seconds=WEAVE_ARGS[:seconds]);
```

```
Error: InterruptException:
```

```julia
results["evaluation_untyped"]
```

```
BenchmarkTools.Trial: 1741 samples with 1 evaluation.
 Range (min … max):  4.707 ms … 207.078 ms  ┊ GC (min … max):  0.00% … 95.5
9%
 Time  (median):     5.573 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   6.425 ms ±   5.103 ms  ┊ GC (mean ± σ):  10.57% ± 14.1
7%

     ▆█▃▁                                                      
  ▅▅▆█████▆▄▄▃▄▃▃▄▃▃▂▁▂▂▁▂▂▁▂▂▂▁▂▂▂▂▂▃▃▂▃▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  4.71 ms         Histogram: frequency by time          12 ms <

 Memory estimate: 6.72 MiB, allocs estimate: 160125.
```

```julia
results["evaluation_typed"]
```

```
BenchmarkTools.Trial: 7634 samples with 1 evaluation.
 Range (min … max):  849.474 μs … 273.128 ms  ┊ GC (min … max):  0.00% … 99
.42%
 Time  (median):       1.073 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):     1.290 ms ±   3.206 ms  ┊ GC (mean ± σ):  10.28% ± 11
.14%

    █                                                            
  ▃▇█▆▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▂▂▂▂▂▂▂▂▂▂▂ ▂
  849 μs           Histogram: frequency by time         6.01 ms <

 Memory estimate: 1016.11 KiB, allocs estimate: 20008.
```

```julia
let k = "evaluation_simple_varinfo_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  50.552 μs … 173.631 ms  ┊ GC (min … max):  0.00% … 99.
85%
 Time  (median):     54.333 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   87.992 μs ±   1.738 ms  ┊ GC (mean ± σ):  21.66% ±  3.
20%

  ██▅▃▂▂▂▁▂  ▂▅▅▄▃▂▂▁ ▁▁▃▃▃▂      ▁▁          ▁▁               ▂
  ██████████▇████████████████▇▇███████▇▇▇▇▇▇▆▇███▇▇▃▆▅▆▄▅▅▆▅▅▆ █
  50.6 μs       Histogram: log(frequency) by time       156 μs <

 Memory estimate: 78.17 KiB, allocs estimate: 2.
```

```julia
let k = "evaluation_simple_varinfo_componentarray"
    haskey(results, k) && results[k]
end
```

```
false
```

```julia
let k = "evaluation_simple_varinfo_dict"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 3189 samples with 1 evaluation.
 Range (min … max):  2.228 ms … 207.975 ms  ┊ GC (min … max):  0.00% … 98.4
0%
 Time  (median):     2.474 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.094 ms ±   4.005 ms  ┊ GC (mean ± σ):  16.85% ± 18.3
2%

  ▅█▇▄▃▂▂▁ ▁▁                                                 ▁
  ████████████▆▆▅▅▄▁▅▁▅▅▅▁▃▃▃▁▁▁▃▁▁▁▁▃▁▁▁▁▃▆▁▆█▇▇▇▇███▇▇▆▆▆▆▆ █
  2.23 ms      Histogram: log(frequency) by time      9.54 ms <

 Memory estimate: 3.43 MiB, allocs estimate: 140069.
```

```julia
let k = "evaluation_simple_varinfo_dict_from_nt"
    haskey(results, k) && results[k]
end
```

```
BenchmarkTools.Trial: 2750 samples with 1 evaluation.
 Range (min … max):  2.661 ms … 196.827 ms  ┊ GC (min … max):  0.00% … 97.9
4%
 Time  (median):     2.921 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   3.588 ms ±   4.066 ms  ┊ GC (mean ± σ):  15.38% ± 17.9
5%

  ▅█▇▅▃▂▂▁ ▁  ▁                                      ▁        ▁
  ██████████▇███▇▅▃▄▄▅▃▃▁▅▁▁▃▅▁▁▁▁▁▁▃▁▁▁▁▁▁▁▇▆▆█▇█▇▇██▇█▆▇▆▆▅ █
  2.66 ms      Histogram: log(frequency) by time      9.82 ms <

 Memory estimate: 3.73 MiB, allocs estimate: 159519.
```



