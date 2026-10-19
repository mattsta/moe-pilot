# Verification scope

The checked-in suite verifies configuration parsing, path independence,
numerical inference, experiment provenance, conditions, sampling, statistics,
and report production. Run `uv run --locked pytest -W error` after changes.

The end-to-end test invokes the actual configured CLI from an unrelated
directory while forbidding `os.chdir`. It runs a tiny two-layer/four-expert
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

The offline smoke command independently exercises tiny-model baseline parity,
an intervention, and checkpoint resume. It can produce a JSON software-check
record using `smoke --output PATH`. It is separate from scientific experiments.

Local verification uses macOS arm64, Python 3.12.13, uv 0.11.19, PyTorch 2.14.1,
and Transformers 5.20.1. CI is configured for Linux Python 3.11/3.12 and macOS
Python 3.12; hosted CI has not been executed locally.

Local checks passed on 2026-10-19:

| Check | Result |
| --- | --- |
| Locked uv synchronization | Passed |
| Ruff lint and formatting | Passed |
| `pytest -q -W error` | 78 passed; no warnings |
| Offline CLI smoke | Passed; baseline loss error 0 on this machine |
| Configured read-only inventory | Passed without creating input/output directories |
| Wheel and source distribution | Built; new modules and sample configuration verified |
| Built-wheel installation outside the checkout | Installed separately; offline CLI smoke passed |

Historical corpora, 37 CSV tables, and plots remain unchanged in `research/pilot`.
Original model weights, routing/geometry arrays, and per-token ablation losses
were absent. The full downloaded model and source datasets have not been run;
the historical pilot's scientific claims are not independently revalidated.
