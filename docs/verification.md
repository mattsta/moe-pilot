# Verification scope

The checked-in suite verifies configuration parsing, path independence,
numerical inference, experiment provenance, conditions, sampling, statistics,
and report production. Run `uv run --locked pytest -W error` after changes.

The end-to-end test invokes the actual configured CLI from an unrelated
directory while forbidding `os.chdir`, under both forced CPU and automatic
device selection. It runs a tiny two-layer/four-expert
local OLMoE, produces routing and geometry, generates both control families,
evaluates held-out losses, and writes domain, ablation, and subtopic tables and
figures. It then verifies that valid completed stages are reused. The tiny
dimensions exercise the absence of hard-coded pilot expert/layer counts.

Other tests compare baseline and masked losses with Hugging Face, check router
normalization/QKV clipping, cover safetensors layouts and document boundaries,
validate checkpoint resume identities, and reject invalid configuration.
Experiment lifecycle tests check tampered outputs, changed inputs/settings,
failed steps, input mutation during execution, and single-writer exclusion.
Dataset tests use stubbed pinned sources; no source datasets are downloaded.

Backend-order tests exercise CUDA and ROCm selection, unsupported-kernel
rejection, all GPU indices before MPS, MPS before CPU, and strict explicit
device overrides. On every advertised real accelerator, integration tests check
baseline/masked parity, geometry, device-resident weights, host artifact arrays,
checkpoint round trips, and rejection of cross-backend checkpoint reuse.

The offline smoke command independently exercises tiny-model baseline parity,
an intervention, and checkpoint resume. It can produce a JSON software-check
record using `smoke --output PATH`. It is separate from scientific experiments.
Use `devices` to inspect automatic selection or `smoke --device mps|cuda|rocm|cpu`
to require and verify a specific backend.

Local verification uses macOS arm64, Python 3.12.13, uv 0.11.19, PyTorch 2.14.1,
and Transformers 5.20.1. CI is configured for Linux Python 3.11/3.12 and macOS
Python 3.12; hosted CI has not been executed locally.

Actual local accelerator verification used Apple MPS, including the automatic
end-to-end experiment, geometry, both control families, ablation checkpoints,
and reports. The MPS CLI smoke's baseline loss differed from the CPU/Hugging
Face reference by approximately `2.38e-7`, below the `2e-4` accelerator tolerance.
CUDA and ROCm hardware were absent: their runtime selection paths are tested
with mocks and real-device integration tests are provided, but actual CUDA/ROCm
kernel execution is not claimed by this local record.

An optional manually dispatched accelerator workflow targets self-hosted
runners labeled `gpu` and Linux or macOS. It requires the chosen backend before
running the suite, so an absent expected GPU cannot be hidden by skipped tests.

Local checks passed on 2026-10-19:

| Check                                         | Result                                               |
| --------------------------------------------- | ---------------------------------------------------- |
| Locked uv synchronization                     | Passed                                               |
| Ruff lint and formatting                      | Passed                                               |
| `pytest -q -W error`                          | 97 passed; 2 absent CUDA/ROCm hardware tests skipped; no warnings |
| Offline CLI smoke                             | CPU and real MPS passed against CPU/Hugging Face |
| Configured read-only inventory                | Passed without creating input/output directories     |
| Wheel and source distribution                 | Built; new modules and sample configuration verified |
| Built-wheel installation outside the checkout | Installed separately; offline CLI smoke passed       |

Historical corpora, 37 CSV tables, and plots remain unchanged in `research/pilot`.
Original model weights, routing/geometry arrays, and per-token ablation losses
were absent. The full downloaded model and source datasets have not been run;
the historical pilot's scientific claims are not independently revalidated.
