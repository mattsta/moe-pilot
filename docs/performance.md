# Local engine performance

Measured on 2026-10-19: Apple M5 Max, 18 logical CPU cores, 128 GiB unified
memory, macOS 26.5.1, Python 3.12.13, PyTorch 2.14.1. CPU measurements use eight
host threads; MPS measurements use the same host thread setting and float32
precision. MPS per-operation CPU fallback was unset.

MPS improves throughput for the larger tested batches. It is slower at the
smaller sizes. Automatic backend selection checks capability, not which backend
will be fastest for a particular workload.

These are **synthetic engine timings**, not full OLMoE research-pipeline results.
The deterministic model has two layers, eight experts, top-two routing, hidden
size 512, intermediate size 768, vocabulary 1,024, and approximately 88 MB of
safetensors weights. Documents contain 64 random token IDs each.

## Measurements

Times below are medians of seven synchronized runs after two warmups per case
and backend. CPU/MPS order alternates between repetitions. Speedup is CPU time
divided by MPS time; values below 1 mean MPS was slower.

| Input tokens | Workload | CPU (ms) | MPS (ms) | MPS speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | Baseline | 7.90 | 21.49 | 0.37× |
| 64 | Routing + geometry | 261.25 | 389.90 | 0.67× |
| 64 | Ablations | 18.42 | 64.03 | 0.29× |
| 1,024 | Baseline | 25.83 | 37.31 | 0.69× |
| 1,024 | Routing + geometry | 283.01 | 398.83 | 0.71× |
| 1,024 | Ablations | 59.65 | 66.19 | 0.90× |
| 8,192 | Baseline | 114.26 | 53.43 | 2.14× |
| 8,192 | Routing + geometry | 367.74 | 370.70 | 0.99× |
| 8,192 | Ablations | 281.17 | 117.85 | 2.39× |
| 16,384 | Baseline | 209.21 | 74.29 | 2.82× |
| 16,384 | Routing + geometry | 468.60 | 384.17 | 1.22× |
| 16,384 | Ablations | 526.51 | 168.11 | 3.13× |

At 16,384 input tokens, baseline throughput was approximately 78,314 input
tokens/second on CPU and 220,536 on MPS. This is evaluation throughput, not
autoregressive generation throughput. Ablations include baseline plus an early
and a late single-expert intervention. Routing + geometry includes routing
arrays, all-layer expert-weight geometry, and activation geometry on up to 100
sampled tokens, with private CPU-generated random controls transferred to MPS.

All timed losses were checked against the same CPU model and inputs. The
largest absolute loss difference across measured cases was `2.38e-6`, within
the `2e-4` absolute/relative acceptance tolerance.

The [raw JSON record](benchmarks/macos-mps-medium-2026-10-19.json) preserves all
168 elapsed samples, per-case min/max, model configuration, software/device
identity, source hashes, and parity errors. No statistical confidence interval
or universal performance guarantee is inferred from these seven repetitions.

## Reproduce and interpret

```sh
uv run --locked moe-pilot benchmark --devices cpu mps --documents 1 16 128 256 --warmup 2 --repeats 7 --threads 8 --output runs/benchmarks/macos-mps-medium.json
```

The engine still streams layers and transfers weights on each invocation.
Timings include these reads/transfers, expert dispatch, numerical work, geometry
where requested, and returned host arrays. They exclude imports, checkpoint
generation, tokenization, device probes, and writing the measurement JSON.
Process/filesystem caches are warm; persisted inference results are never
reused. The geometry workload has substantial fixed work, which helps explain
why increasing input tokens changes its timing less than baseline inference.
No profiler attribution of individual bottlenecks has been performed.

The original full OLMoE weights are absent. These measurements do not establish
speedup for that model, corpus preparation, report generation, cold starts,
checkpoint persistence, or an entire experiment. CUDA and ROCm hardware were
unavailable and have no measured performance results here. Use the same
[benchmark API and CLI](runbook.md#measure-performance) with a local checkpoint
and the corresponding backend to measure those environments.
