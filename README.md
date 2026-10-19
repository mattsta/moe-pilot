# MoE Pilot

A Python library and executable research pipeline for studying expert routing,
specialization, geometry, and causal ablations in OLMoE. Inference streams
one decoder layer at a time onto CUDA, ROCm, MPS, or CPU from local safetensors.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --extra data
uv run --locked moe-pilot --help
uv run --locked moe-pilot devices
uv run --locked moe-pilot smoke
uv run --locked pytest -W error
uv run --locked moe-pilot --config configurations/pilot.toml inventory
```

The default `execution.device = "auto"` tries available CUDA/ROCm GPUs, then
Apple MPS, and uses CPU only after accelerator availability/capability checks
fail. Explicit `cuda[:index]`, `rocm[:index]`, `mps`, and `cpu` choices are also
supported. `devices` reports the selected backend and rejected candidates.

For synchronized CPU/accelerator timings, use the reusable `benchmark` command;
see the [benchmark procedure](docs/runbook.md#measure-performance) and
[recorded local results](docs/performance.md). Backend availability alone does
not establish a speedup.

The smoke check is offline and uses a tiny random model on the selected device.
The experiment pipeline
uses one validated [TOML configuration](configurations/pilot.toml) for model,
data, output paths, sampling, inference, controls, and reporting. Paths resolve
relative to that file. The CLI and library share the same experiment runner;
execution does not depend on the working directory.

## Run an experiment

Prepare the local model and input corpora using the [operator runbook](docs/runbook.md),
then execute:

```sh
uv run --locked moe-pilot --config configurations/pilot.toml run
```

The runner performs routing/geometry inference, domain reporting, condition
generation, checkpointed ablations, ablation reporting, and subtopic reporting.
It records configuration, source identities, dependency versions, input/output
hashes, and stage status. Completed stages are reused only while their inputs,
outputs, and dependencies remain valid. Different settings require a new output
directory. Inference and probes use the same engine.

## Library and repository

```python
from moe_pilot.config import load_experiment
from moe_pilot.experiment import ExperimentRunner

experiment = load_experiment("configurations/pilot.toml")
ExperimentRunner(experiment).run_pipeline()
```

- `src/moe_pilot/`: reusable inference, experiment infrastructure, and workflows.
- `configurations/`: checked-in experiment definitions.
- `tests/`: offline numerical and end-to-end regression checks.
- `research/pilot/`: unchanged historical corpora, attribution, tables, and plots.
- [Architecture](docs/architecture.md), [runbook](docs/runbook.md),
  [contributing](CONTRIBUTING.md), and [verification](docs/verification.md).

Development uses Python 3.12, `pyproject.toml`, and `uv.lock`. Runtime-only
installation is `uv sync --locked --no-dev`. Downloading source datasets also
requires the `data` extra. Default wheels support CUDA on Linux and native MPS
on macOS. Linux AMD users select `--extra rocm`; CPU-only Linux wheels are an
opt-in `--extra cpu` profile. Use the same profile on `uv sync` and `uv run`;
see [backend installation and validation](docs/runbook.md#devices-and-pytorch-installation).

The historical pilot's model weights, raw routing/geometry arrays, and per-token
ablation losses were not supplied. Software checks use synthetic local models
and do not independently validate the historical scientific claims. The archived
outputs are evidence; they are not active experiment inputs or cache formats.
Code licensing remains an owner decision; dataset and model terms remain separate.
