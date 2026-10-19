# Migration verification

Verified locally on 2026-10-19 on macOS arm64 with Python 3.12.13,
uv 0.11.19, PyTorch 2.14.1, and Transformers 5.20.1.

| Check | Result |
| --- | --- |
| `uv sync --locked --all-extras` | Passed; lockfile and environment agree |
| `uv run ruff check .` | Passed |
| `uv run ruff format --check .` | Passed |
| `uv run pytest -q -W error::RuntimeWarning` | 46 tests passed; no runtime warnings |
| Offline tiny-model smoke command | Passed; default baseline loss matched Hugging Face exactly on this machine |
| `uv build` | Wheel and source distribution built successfully |
| `research/pilot/` inventory | 500 analysis documents, 200 held-out documents, 1,140 subtopic documents; 37 historical CSVs |

Tests verify baseline and masked next-token losses against Hugging Face,
variable-length document boundaries, top-k normalization and QKV clipping,
original sharded and packed safetensors layouts, independent engine state,
checkpoint restart, model/corpus/mask identity rejection, and completed-work
skipping. Control matching is compared with the original sampling procedure.

The offline integration test produces routing and orthogonality arrays, domain
CSVs/plots, activation probes, condition definitions, held-out ablation losses,
both control families, summaries, and subtopic CSVs/plots. It uses a synthetic
16-layer, 64-expert model with a tiny hidden dimension and a reduced control
search. Corpus builders are exercised with stubbed dataset sources. Tests
explicitly disable network access through Hugging Face's offline settings.

The full downloaded OLMoE checkpoint and source datasets were not executed.
Model weights, original routing/orthogonality arrays, and original per-token
ablation losses were absent. Historical result tables and plots were preserved;
the pilot's scientific claims were not independently revalidated. CI is
configured for Linux and macOS but has not been run by this local verification.

To repeat verification after changes, follow [the runbook](runbook.md). Use
checked-in tests and the reusable smoke command; no inline Python is needed.
