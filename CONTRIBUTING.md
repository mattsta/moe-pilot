# Contributing

See [the operator runbook](docs/runbook.md) for complete bring-up and
[the architecture](docs/architecture.md) for module boundaries and library use.

```sh
uv sync --locked --all-extras
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
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

Tests are offline and use tiny local random models. For numerical changes,
compare to Hugging Face and test padding, document boundaries, router masking,
checkpoint restart, and both safetensors layouts. Preserve existing research
artifact schemas or document migrations. Changes to sampling/statistical
methods need an explanation of how reported results may change.

Do not commit downloaded models, virtual environments, caches, logs, or large
raw arrays. Archived pilot CSVs/plots/corpora live in `research/pilot/`; run
experiments in separate workspaces. Runtime workspaces belong outside source
control. Code licensing remains an owner decision; dataset/model terms remain
separate from any future code license.

Keep application code under `src/moe_pilot/` and tests under `tests/`; do not
add root-level Python scripts or compatibility wrappers. Use the documented CLI
and reusable package APIs for execution.
