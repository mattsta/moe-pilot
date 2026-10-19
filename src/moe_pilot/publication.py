"""Validate completed experiment evidence and publish compact local research reports."""

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from moe_pilot.artifacts import read_json, write_json

RECONSTRUCTION_STEPS = (
    "routing",
    "domain-report",
    "build-conditions",
    "ablate",
    "ablation-report",
    "subtopic-report",
)


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_evidence(run_dir, required_steps=RECONSTRUCTION_STEPS):
    """Check recorded source/input/output identities without adopting current code.

    Frozen runtimes retain their own source files and dependency records. A
    report can inspect their completed artifacts without rerunning their model
    or pretending that a different installed version produced the results.
    """
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "experiment.json")
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported experiment evidence schema")
    records = manifest.get("steps", {})
    for name in required_steps:
        record = records.get(name, {})
        if record.get("status") != "complete" or not record.get("outputs"):
            raise ValueError(f"Incomplete evidence for {name}")
        for filename, expected in record.get("inputs", {}).items():
            path = Path(filename)
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"Missing recorded input: {filename}")
            if re.fullmatch(r"\d+:\d+", expected):
                stat = path.stat()
                actual = f"{stat.st_size}:{stat.st_mtime_ns}"
            else:
                actual = _hash(path)
            if actual != expected:
                raise ValueError(f"Changed recorded input: {filename}")
        for filename, expected in record["outputs"].items():
            path = Path(filename)
            if not path.is_relative_to(run_dir) or not path.is_file() or _hash(path) != expected:
                raise ValueError(f"Changed or invalid output: {filename}")
        for parent, outputs in record.get("dependencies", {}).items():
            if (
                records.get(parent, {}).get("status") != "complete"
                or records[parent]["outputs"] != outputs
            ):
                raise ValueError(f"Incoherent dependency for {name}: {parent}")
    return manifest


def _table(frame, columns):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "undefined" if not np.isfinite(value) else f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend(
        "| " + " | ".join(cell(row[column]) for column in columns) + " |"
        for row in frame.to_dict("records")
    )
    return "\n".join(lines)


def publish_reconstruction(run_dir, destination):
    """Export validated small artifacts and generated reader/detailed analyses.

    Large raw arrays remain in the run; their hashes are retained in the evidence
    inventory. Existing unrelated destination files are preserved.
    """
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if destination == run_dir or destination.is_relative_to(run_dir):
        raise ValueError("Publication destination must be separate from the source run")
    with FileLock(run_dir / ".experiment.lock", timeout=0):
        manifest = validate_evidence(run_dir)
        manifest_hash = _hash(run_dir / "experiment.json")
        config = manifest["configuration"]
        with np.load(run_dir / "artifacts/routing.npz") as data:
            valid, loss, doc, labels = data["valid"], data["loss"], data["doc"], data["dom"]
            domains = [str(name) for name in data["doms"]]
            baseline = float(loss[valid].mean(dtype=np.float64))
            dimensions = data["topi"].shape
            domain_rows = []
            for index, domain in enumerate(domains):
                selected = valid & (labels == index)
                domain_rows.append(
                    {
                        "domain": domain,
                        "documents": len(np.unique(doc[labels == index])),
                        "valid_targets": int(selected.sum()),
                        "mean_loss_nats_per_token": float(loss[selected].mean(dtype=np.float64)),
                    }
                )
        domain_frame = pd.DataFrame(domain_rows)
        routing = pd.read_csv(run_dir / "tables/layer_domain_metrics.csv")
        ablations = pd.read_csv(run_dir / "tables/ablations/ablation_summary.csv")
        geometry = pd.read_csv(run_dir / "tables/orthogonality_by_layer.csv")
        centered = pd.read_csv(run_dir / "tables/activation_orthogonality.csv")
        subtopics = pd.read_csv(run_dir / "tables/subtopics/layer_within_vs_between_jsd.csv")
        selected_layers = sorted(routing.layer.unique())
        display_layers = sorted(config["analysis"]["layers"] or selected_layers)
        middle = display_layers[len(display_layers) // 2]
        contrasts = ablations[
            (ablations.control_family == "on-domain-control")
            & (ablations.expert_count == 1)
            & (ablations.scope == f"layer-{middle}")
        ].copy()
        contrasts["relative_mass_gap"] = np.abs(
            contrasts.specialist_routing_mass - contrasts.control_routing_mass_mean
        ) / contrasts.specialist_routing_mass.replace(0, np.nan)
        final_layer = max(selected_layers)
        final_routing = routing[routing.layer == final_layer]
        centered_mean = float(centered.mean_abs_cos_centered.mean())
        inventory = []
        destination.mkdir(parents=True, exist_ok=True)
        seen = set()
        for step in RECONSTRUCTION_STEPS:
            for filename, digest in manifest["steps"][step]["outputs"].items():
                path = Path(filename)
                if path in seen:
                    continue
                seen.add(path)
                relative = path.relative_to(run_dir)
                exported = path.suffix in (".csv", ".png", ".json")
                if exported:
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    if _hash(target) != digest:
                        raise ValueError("Published artifact checksum mismatch")
                inventory.append(
                    {
                        "source": str(path),
                        "relative_path": str(relative),
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                        "exported": exported,
                    }
                )
        domain_frame.to_csv(destination / "tables/baseline_by_domain.csv", index=False)
        condition_count = int(
            pd.read_csv(run_dir / "tables/ablations/delta_loss.csv").condition.nunique()
        )
        summary = f"""# OLMoE archived input reconstruction

The pinned local model `{config["model"]["repository"]}` at revision
`{config["model"]["revision"]}` has completed routing, geometry, held-out
ablations, and subtopic analysis on the archived input population.

Baseline evaluation covers {int(domain_frame.documents.sum()):,} documents and
{int(domain_frame.valid_targets.sum()):,} valid targets across {len(domains)} domains.
Mean loss is **{baseline:.6f} nats/token**. The held-out ablation run evaluated
{condition_count} declared conditions in addition to baseline.

{_table(domain_frame, ["domain", "documents", "valid_targets", "mean_loss_nats_per_token"])}

The single-expert contrasts below compare specialist removal with the mean
on-domain-routing-mass control at selected layer {middle}. Positive differences
mean greater specialist loss impact on its nominated domain. Values are
nats/token; `relative_mass_gap` is a fraction. Intervals are
unadjusted paired document bootstraps; these are exploratory comparisons.

{_table(contrasts, ["domain", "paired_difference", "ci95_low", "ci95_high", "relative_mass_gap"])}

Mass gaps quantify imperfect matching. Routing preference, low parameter
cosine, and an unadjusted loss contrast do not establish a semantic specialist.
The original corpus also associates domains with different dataset sources.
The [detailed analysis](analysis.md) separates these observations and limits.

This is a reconstruction with newly pinned weights. The archived notes do not
identify the exact original model commit, and the new run does not recover that
missing identity. [Evidence inventory](evidence.json) records the exact inputs,
software, device, and generated artifacts.
"""
        detailed = f"""# Reconstruction methods and analysis

## Model and execution

Repository: `{config["model"]["repository"]}`. Revision:
`{config["model"]["revision"]}`. All {dimensions[0]} decoder layers are evaluated;
top-{dimensions[2]} experts are selected per token. The recorded device is
`{manifest["steps"]["routing"]["device_selection"]["backend"]}` and inference is
float32. Configuration and version/source identities are in [evidence.json](evidence.json).

The discovery corpus is used for routing distributions and specialist selection.
Held-out documents are evaluated for ablation deltas. The original inputs and
fixed sample settings are retained; this reconstruction is distinct from the
new source-balanced confirmatory study. Input files, archived source records,
and all generated output hashes are validated before publication.

## Baseline and domain routing

{_table(domain_frame, ["domain", "documents", "valid_targets", "mean_loss_nats_per_token"])}

Loss averages weight valid next-token targets. Each document's final token is
excluded. Domain-routing divergence at the final layer is measured in bits:

{_table(final_routing, ["domain", "jsd_vs_overall", "jsd_null_mean", "jsd_null_max", "effective_experts"])}

The configured {config["analysis"]["permutations"]} document-label permutations
provide descriptive null summaries. The largest sampled null is not a
multiple-comparison-adjusted significance threshold. Token identity/category
diagnostics are retained in the full tables and figures.

![Routing divergence by layer](figures/jsd_by_layer.png)

## Expert geometry

The layer-average centered expert-output absolute cosine is {centered_mean:.6g}.
Full weight and output comparisons with Gaussian/norm-matched controls are in
[orthogonality_by_layer.csv](tables/orthogonality_by_layer.csv) and
[activation_orthogonality.csv](tables/activation_orthogonality.csv).

{_table(geometry.groupby(["space", "kind"], as_index=False)[["mean_abs_cos", "mean_signed_cos"]].mean(), ["space", "kind", "mean_abs_cos", "mean_signed_cos"])}

These are descriptive similarities on sampled inputs. Flattened weight cosine
is sensitive to internal neuron permutations and high-dimensional geometry;
it does not measure causal independence or distinct knowledge.

![Expert output cosine by layer](figures/activation_cos_by_layer.png)

## Held out interventions and matching

The reconstruction excludes target experts before softmax/top-k. It allows
replacement experts and changes gate weights. Its loss deltas therefore measure
exclusion with rerouting. New fixed-routing modes are tested separately.
Specialists and both control families are represented in
[the complete ablation summary](tables/ablations/ablation_summary.csv).

{_table(contrasts, ["domain", "specialist_on_domain_delta", "control_on_domain_delta", "paired_difference", "ci95_low", "ci95_high", "relative_mass_gap"])}

Displayed contrasts use selected layer {middle} and one expert, without
selecting rows for their observed effect. Relative mass gap is the absolute
specialist-minus-mean-control mass divided by specialist mass. Controls can
have substantial residual imbalance. Confidence intervals use
{config["analysis"]["bootstrap_samples"]} paired document bootstrap samples
and are unadjusted. The configured {config["ablations"]["control_seeds"]} control
draws are not independent trained models.

![Held out specialist loss changes across domains](figures/ablations/delta_loss.png)

## Subtopic observations

The separate subtopic corpus is evaluated across its declared groups.
Final-layer within-domain and between-domain routing divergences are:

{_table(subtopics[subtopics.layer == final_layer], ["domain", "within_domain_pair_jsd", "within_null_mean", "between_domain_subtopic_pair_jsd"])}

Subject, source, and language can contribute to these differences. The
subtopic analysis does not by itself isolate semantic domain expertise.

## Reproduction and interpretation

The run's input/source records and output checksums are in the evidence
inventory. Small CSV, JSON, and PNG artifacts are exported alongside this
analysis; large raw arrays and per-token checkpoints remain in the local run
with their hashes retained. The frozen runtime's source files are validated
without substituting the currently installed implementation.

The appropriate findings are measured routing/geometry patterns and conditional
intervention effects in this sample. The source-balanced causal study addresses
remaining source, lexical, intervention, and matching questions. Neither
reconstruction nor synthetic performance establishes general reasoning quality.
"""
        (destination / "README.md").write_text(summary, encoding="utf-8")
        (destination / "analysis.md").write_text(detailed, encoding="utf-8")
        if _hash(run_dir / "experiment.json") != manifest_hash:
            raise ValueError("Experiment changed during publication")
        write_json(
            destination / "evidence.json",
            {
                "schema": 1,
                "kind": "validated-reconstruction-publication",
                "source_manifest_sha256": manifest_hash,
                "publisher_source_sha256": _hash(Path(__file__)),
                "experiment": manifest,
                "artifacts": inventory,
                "derived": {
                    "baseline_mean_loss": baseline,
                    "documents": int(domain_frame.documents.sum()),
                    "valid_targets": int(domain_frame.valid_targets.sum()),
                    "conditions": condition_count,
                    "displayed_layer": int(middle),
                },
                "publication_files": {
                    str(path.relative_to(destination)): _hash(path)
                    for path in (
                        destination / "README.md",
                        destination / "analysis.md",
                        destination / "tables/baseline_by_domain.csv",
                    )
                },
            },
        )
    return destination / "README.md"
