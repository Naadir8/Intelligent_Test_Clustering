# Performance Profiling and Optimization

## Objective
This document presents the results of performance profiling and optimization for the Intelligent Test Case Clustering system, developed as part of the bachelor's thesis "Intelligent system for grouping and evaluating test scenarios in automated software testing".

## Methodology
Profiling was conducted using a standardized benchmark that includes three main stages:
1. Generation of synthetic test cases (`TestCaseLoader.generate_synthetic`)
2. Generation of semantic embeddings (`TestCaseEmbedder.embed_dataframe`)
3. Clustering of embeddings using KMeans (`TestCaseClusterer.fit_predict`)

All measurements were performed on a dataset of **3000 synthetic test cases** with **8 clusters**.  
The same hardware and environment were used for both baseline and optimized runs.  
Each stage was measured for wall-clock time, cumulative CPU time, and memory consumption.

## Tools Used
- **cProfile** — built-in deterministic CPU profiler (for function-level timing)
- **memory_profiler** — line-by-line memory usage analysis
- **time.perf_counter()** — high-resolution wall-clock time measurement
- **pstats** — for sorting and displaying cProfile results

## Test Scenarios
- **Dataset size**: 3000 test cases
- **Embedding model**: `all-MiniLM-L6-v2`
- **Number of clusters**: 8
- **Hardware**: 
  - CPU: AMD Ryzen 5 5600
  - RAM: 16GB

## Baseline Results (Before Optimization)

**Total pipeline execution time:** 14.20 seconds

**CPU Profiling Results (cProfile):**

| Rank | Function / Method                                      | Cumulative Time (s) | % of Total Time |
|------|--------------------------------------------------------|---------------------|-----------------|
| 1    | `run_benchmark`                                        | 14.20               | 100%            |
| 2    | `decorate_context` (torch)                             | 8.75                | ~61.6%          |
| 3    | `embed_dataframe`                                      | 6.61                | ~46.5%          |
| 4    | `SentenceTransformer.encode`                           | 6.61                | ~46.5%          |
| 5    | `_call_impl` (torch)                                   | 6.17                | ~43.5%          |
| 6    | `forward` (various transformer layers)                 | ~5.9 – 6.1          | ~41–43%         |

The dominant bottleneck is the **embedding generation stage**, which accounts for the majority of execution time due to multiple transformer forward passes.

**Memory Profiling Results:**

- Peak memory usage during embedding generation: **488.7 MiB**
- Significant memory increment occurs during model inference (`embed_dataframe` call)

## Identified Bottlenecks
1. **Embedding generation** (`SentenceTransformer.encode` and internal transformer layers) — the most time-consuming part (~46–61% of total time).
2. Heavy usage of PyTorch internals (`decorate_context`, `_call_impl`, `forward` methods).
3. Repeated model loading (no caching).
4. Default inference settings (small effective batch size, no optimizations).

## Optimizations Applied (Planned)
- Implementation of `batch_size` parameter in `embed_dataframe`
- Singleton pattern for `TestCaseEmbedder` (load model only once)
- Configurable device selection (CPU/GPU)
- Increased batch size for inference

## Results After Optimization (to be filled)
- Total execution time: __.__ seconds
- Performance improvement: __.__ %
- Peak memory usage: __.__ MiB
- New bottlenecks: ...

## Conclusions and Recommendations for Further Improvements
The baseline profiling clearly shows that embedding generation is the primary performance bottleneck in the system.

**Recommended further optimizations:**
- Add `batch_size` parameter (128–256) to `SentenceTransformer.encode()`
- Implement singleton pattern to avoid reloading the model
- Enable GPU acceleration (`device='cuda'`) when available
- Export the model to ONNX format for faster inference
- Apply model quantization (8-bit or 4-bit)
- Consider using lighter models (e.g. `paraphrase-MiniLM-L3-v2` or distilled variants)
- Switch to `MiniBatchKMeans` for very large datasets

These improvements are expected to reduce total execution time by 40–60% while maintaining clustering quality.

**Files created/modified:**
- `src/performance/benchmark.py`
- `src/embedding/embedder.py`
- `docs/performance.md`