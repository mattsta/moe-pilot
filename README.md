# MoE Pilot

A Python library and executable research pipeline for studying expert routing,
specialization, orthogonality, and causal ablations in
`allenai/OLMoE-1B-7B-0924`. CPU inference streams one decoder layer at a time
from local safetensors rather than loading the full model into RAM.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --all-extras
uv run --locked moe-pilot --help
uv run --locked moe-pilot --workspace research/pilot inventory
uv run --locked moe-pilot --workspace /tmp/moe-pilot-smoke smoke
uv run --locked pytest
```

The smoke check runs offline with a tiny random model and compares losses with
Hugging Face. It needs no model download or GPU. Development uses Python 3.12;
uv manages `.venv` and exact dependencies in `uv.lock`. Runtime installation
without development tools or dataset downloads is `uv sync --locked --no-dev`.

## Documentation

- [Operator runbook](docs/runbook.md): full bring-up, research pipeline,
  prerequisites, outputs, checkpoint recovery, and release records.
- [Architecture and API](docs/architecture.md): module responsibilities,
  reusable engine, supported configurations, and compatibility changes.
- [Contributing](CONTRIBUTING.md): development checks and change guidelines.
- [Verification record](docs/verification.md): tested behavior and remaining limits.
- [Historical pilot notes](docs/pilot-notes.md): original experiment claims and
  hardware context, retained separately from software verification.

## Repository contents and verification limits

`src/moe_pilot/` contains all application code; `tests/` contains offline tests.
`research/pilot/` preserves the original sampled corpora, source metadata, CSV
tables, and plots. Use the installed `moe-pilot` command to run workflows.

The supplied checkout has no model weights, `routing.npz`, `ortho_raw.npz`, or
per-token ablation losses. Existing tables are historical evidence. Software
checks exercise tiny synthetic inputs and do not independently revalidate the
published pilot numbers. See the runbook to regenerate the full experiment in
a fresh workspace. The experiment-specific reports target the original
16-layer, 64-expert model; the reusable engine has broader configuration support.

Before public release, choose a code license and review redistribution terms
for bundled source texts and model artifacts. No license has been inferred or
assigned on the owner's behalf.
