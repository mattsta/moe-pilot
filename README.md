# MoE Pilot

A local Python research library and executable reference project for studying
expert routing, specialization, geometry, and causal contributions in open
mixture-of-experts models. The current engine supports OLMoE and streams one
decoder layer at a time onto CUDA, ROCm, MPS, or CPU from local safetensors.
Experiments inspect and intervene on model internals during local execution.

The [research program](docs/research-program.md) defines the project purpose,
audits current capabilities, and prioritizes additional models and studies.
The [causal specialization protocol](docs/studies/causal-specialization.md)
specifies the next proposed experiment and its analysis requirements.

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

## Measured research results

The [full OLMoE 0924 reconstruction](research/reconstruction-0924/README.md)
completed locally on MPS with a pinned 6.9B-parameter checkpoint: 500 analysis
documents, 29,522 valid next-token targets, mean loss **2.130555 nats/token**,
420 held-out intervention conditions, and subtopic analysis.

The displayed layer-7 single-expert comparisons have approximately **22–93%**
relative routing-mass gaps between specialists and their mean controls.
Those gaps limit causal interpretation despite visible routing preferences
and loss differences. The new study adds independent selection/calibration
partitions, explicit intervention modes, and measured matching criteria.
See the [detailed reconstruction analysis](research/reconstruction-0924/analysis.md)
and [completion checklist](docs/research-progress.md) for evidence and status.

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

The historical pilot's original weights, raw routing/geometry arrays, and
per-token losses were not supplied. The new reconstruction uses newly pinned
weights and records its own raw evidence; the archive lacks the original model
commit needed to establish exact checkpoint replication. Archived outputs
remain separate from active experiment inputs and caches.
Code licensing remains an owner decision; dataset and model terms remain separate.
