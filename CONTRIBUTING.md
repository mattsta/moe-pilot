# Contributing

See [the operator runbook](docs/runbook.md) for complete bring-up and
[the architecture](docs/architecture.md) for module boundaries and library use.

```sh
uv sync --locked --extra data
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -W error
uv build
```

Use `uv add` for direct runtime requirements, `uv add --optional data` for
dataset-only requirements, and `uv add --dev` for development tools. Commit
`pyproject.toml` and the generated `uv.lock` together. Do not edit the lockfile
manually or regenerate the historical environment dump.

Put reusable behavior in `src/moe_pilot/` and verification in checked-in tests.
Do not use inline Python or throwaway command-line validation programs. Prefer
focused modules to additional service layers or dependencies. Keep imports
free of downloads, model loading, experiment execution, and output writes.

Research execution is local, using inspectable model weights and direct access
to routing, activations, expert outputs, and intervention points. Managed model
APIs do not satisfy that execution contract. Network use is limited to explicit
input/model acquisition and literature access. Follow the
[research program](docs/research-program.md) when extending scientific scope;
new model families require validated local architecture support.

Tests are offline and use tiny local random models. For numerical changes,
compare to Hugging Face and test padding, document boundaries, router masking,
checkpoint restart, and both safetensors layouts. Preserve existing research
artifact schemas or document migrations. Changes to sampling/statistical
methods need an explanation of how reported results may change.

Do not commit downloaded models, virtual environments, caches, logs, or large
raw arrays. Archived pilot CSVs/plots/corpora live in `research/pilot/`; new
experiments use explicit configuration with separate input/output directories.
Generated experiment outputs belong outside source control.
Code licensing remains an owner decision; dataset/model terms remain
separate from any future code license.

Keep application code under `src/moe_pilot/` and tests under `tests/`; do not
add root-level Python scripts or compatibility wrappers. Use the documented CLI
and reusable package APIs for execution.

There is one new-system API: validated `Experiment` settings and
`ExperimentRunner`. Do not add working-directory changes, legacy command aliases,
filename-based scientific metadata, or alternate inference implementations.
Use the engine's observer API for new probes. Add provenance/dependency checks
and return explicit output paths for new stages. Configuration changes must
have one source of truth shared by CLI and library callers.

Preserve automatic CUDA/ROCm → MPS → CPU selection and capability validation.
Keep temporary tensors on the chosen compute device and artifacts on the host.
Add accelerator tests for new operations; tests skip only hardware absent from
the host and fail when an advertised backend cannot execute required kernels.
Use `--extra rocm` consistently for AMD installs, or `--extra cpu` for optional
CPU-only Linux installs. Do not use `--all-extras`: those wheel profiles conflict.
