---
name: use-vast-for-long-inference
description: "Run benchmarks expected to exceed five minutes on Vast instead of locally"
condition: "\"i\"\\s*:\\s*\"(?:Running canonical fixed baseline|Validating final benchmark harness|Rerunning stable immutable baseline)\""
scope: "tool"
---

Do not launch this multi-hour benchmark locally. Use the repository’s canonical Vast workflow for inference or benchmarking expected to exceed roughly five minutes. Reserve local execution for bounded sub-five-minute smoke or profiling checks, while preserving the fixed workload and measurement contract.