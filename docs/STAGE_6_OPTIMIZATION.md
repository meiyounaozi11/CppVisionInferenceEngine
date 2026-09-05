# Stage 6 Optimization Results

Stage 5 identified serial file/decode as the end-to-end bottleneck. This stage adds `ImagePreparationPipeline`: a bounded source queue and joinable decode+preprocess workers feeding the existing bounded inference queue. Workers separate binary file read from `cv::imdecode`, use the existing preprocessor, and move tensors into `InferenceTask`. Decode failures are reported and never escape a worker thread.

The Stage 3 result queue and shutdown contract are unchanged. Preparation is stopped and joined before inference is stopped, while a consumer continuously drains results. Decode completion order is not guaranteed; task IDs correlate results.

Initial Debug smoke (20 tasks, MobileNetV2, capacity 2, four inference workers) observed 13.3 images/s with one preparation worker and 36.6 images/s with four. These are smoke observations, not release conclusions. In the Release matrix (20 measured tasks, two warm-ups, three repetitions), the primary disk workload had median throughput 24.74/s with one preparation worker, 43.83/s with two, and 66.00/s with four (four inference workers, ORT 1/1, capacity 2). The Stage 5 serial baseline was 25.69/s, so the four-worker preparation path is about 2.57x that baseline. Median E2E p95 for that configuration was 33.59 ms; ORT inference mean was 27.42 ms under the combined CPU load. Memory-backed runs were similar (25.14/s, 45.06/s, 67.15/s for one/two/four preparation workers), showing JPEG decode dominates this input more than file read.

The second image was checked with a Release smoke run using four preparation and four inference workers (capacity 2, ORT 1/1): 92.96/s for 20 tasks, p95 32.28 ms, decode mean 33.38 ms. This single run is cross-workload evidence, not a full matrix.

Reproducible Release runs are produced by `scripts/run_stage6_benchmarks.ps1` into ignored `benchmark_results/stage6_optimization.csv`.
