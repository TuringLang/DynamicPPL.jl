## DynamicPPL Benchmark Results (benchmarks_2025-02-02_04-21-46)

### Execution Environment
- Julia version: 1.10.5
- DynamicPPL version: 0.32.2
- Benchmark date: 2025-02-02T04:22:00.341

| Model | Evaluation Type                           |       Time |    Memory | Allocs | Samples |
|-------|-------------------------------------------|------------|-----------|--------|---------|
| demo1 | evaluation typed                          | 194.000 ns | 160 bytes |      3 |   10000 |
| demo1 | evaluation untyped                        |   1.027 μs |  1.52 KiB |     32 |   10000 |
| demo1 | evaluation simple varinfo dict            | 694.000 ns | 704 bytes |     26 |   10000 |
| demo1 | evaluation simple varinfo nt              |  43.000 ns |   0 bytes |      0 |   10000 |
| demo1 | evaluation simple varinfo dict from nt    |  48.000 ns |   0 bytes |      0 |   10000 |
| demo1 | evaluation simple varinfo componentarrays |  44.000 ns |   0 bytes |      0 |   10000 |
| demo2 | evaluation typed                          | 273.000 ns | 160 bytes |      3 |   10000 |
| demo2 | evaluation untyped                        |   2.528 μs |  3.47 KiB |     67 |   10000 |
| demo2 | evaluation simple varinfo dict            |   2.189 μs |  1.42 KiB |     60 |   10000 |
| demo2 | evaluation simple varinfo nt              | 136.000 ns |   0 bytes |      0 |   10000 |
| demo2 | evaluation simple varinfo dict from nt    | 119.000 ns |   0 bytes |      0 |   10000 |
| demo2 | evaluation simple varinfo componentarrays | 136.000 ns |   0 bytes |      0 |   10000 |

