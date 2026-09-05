# Stage 6 Optimization Results

Stage 5 identified serial file/decode as the end-to-end bottleneck. This stage adds `ImagePreparationPipeline`: a bounded source queue and joinable decode+preprocess workers feeding the existing bounded inference queue. Workers separate binary file read from `cv::imdecode`, use the existing preprocessor, and move tensors into `InferenceTask`. Decode failures are reported and never escape a worker thread.

The Stage 3 result queue and shutdown contract are unchanged. Preparation is stopped and joined before inference is stopped, while a consumer continuously drains results. Decode completion order is not guaranteed; task IDs correlate results.

Initial Debug smoke (20 tasks, MobileNetV2, capacity 2, four inference workers) observed 13.3 images/s with one preparation worker and 36.6 images/s with four. These are smoke observations, not release conclusions. Reproducible Release runs are produced by `scripts/run_stage6_benchmarks.ps1` into ignored `benchmark_results/stage6_optimization.csv`.
