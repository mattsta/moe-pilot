"""Resumable source-balanced local research, with explicit frozen analysis choices."""

import gc
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.catalog import load_catalog
from moe_pilot.engine import Engine
from moe_pilot.observations import encode_rows, observe_documents
from moe_pilot.perturbations import Intervention
from moe_pilot.resources import bounded_inference, peak_process_bytes
from moe_pilot.study_data import load_study_split

PLAN = {
    "schema": 1,
    "layer": 7,
    "primary_modes": ["remove", "reroute"],
    "secondary_modes": ["rescale", "substitute"],
    "primary_metric": "document-mean next-token loss, nats/token",
    "primary_contrast": "specialist-minus-control effect on-domain minus equally weighted off-domains",
    "primary_family": "three domains by two primary modes; Holm adjustment",
    "bootstrap": "paired documents stratified by source and domain; 10000 replicates",
    "permutation": "domain labels permuted within source; 9999 replicates; plus-one two-sided p",
    "selection": "rank by discovery on/off mass ratio; first of top eight admitting a calibration-matched control; otherwise top expert with failed balance explicitly retained",
    "control_pool": "discovery specificity at or below its median, excluding discovery top eight",
    "matching": "calibration on/off mass, on/off usage, and raw output norm; relative gaps <=10%, norm <=20%",
    "weighting": "equal documents; frozen balanced source/domain cells",
    "format": "identical plain question/choices/answer-label text across stages; no model-specific chat template",
    "interpretation": "post-training comparisons are observational; failed matching blocks causal-specialization claims",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def implementation():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "study.py",
            "engine.py",
            "model.py",
            "devices.py",
            "resources.py",
            "observations.py",
            "perturbations.py",
            "study_data.py",
        )
    }


def save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_observations(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete observations: {directory}")
    parts = []
    for item in manifest["chunks"]:
        path = directory / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Observation checkpoint bytes changed")
        with np.load(path, allow_pickle=False) as data:
            parts.append({key: data[key] for key in data.files})
    arrays = {}
    for key in parts[0]:
        if key in ("layers", "feature_names", "condition_names"):
            if any(not np.array_equal(part[key], parts[0][key]) for part in parts):
                raise ValueError("Observation checkpoint dimensions changed")
            arrays[key] = parts[0][key]
        elif key == "unigram":
            arrays[key] = sum(part[key] for part in parts)
        else:
            axis = (
                1
                if key in ("mass", "usage", "norm", "condition_loss", "condition_answer_loss")
                else 0
            )
            arrays[key] = np.concatenate([part[key] for part in parts], axis=axis)
    return arrays, manifest


def conditions_from_selection(selection):
    result = {}
    for item in selection["domains"]:
        domain, target, control = item["domain"], item["expert"], item["control"]
        for mode in (*PLAN["primary_modes"], *PLAN["secondary_modes"]):
            for role, expert, donor in (
                ("specialist", target, control),
                ("control", control, target),
            ):
                result[f"{domain}_{mode}_{role}"] = Intervention(
                    {PLAN["layer"]: [expert]},
                    mode,
                    donors={PLAN["layer"]: {expert: donor}} if mode == "substitute" else {},
                )
        result[f"{domain}_half"] = Intervention({PLAN["layer"]: [target]}, "remove", 0.5)
        result[f"{domain}_restored"] = Intervention({PLAN["layer"]: [target]}, "remove", 0)
        result[f"{domain}_joint"] = Intervention({PLAN["layer"]: [target, control]}, "remove")
    return result


@bounded_inference
def collect(config, catalog, names, partition, *, max_batches=None, interventions=False):
    """Collect only real study observations; every completed microbatch is reusable."""
    if partition not in ("discovery", "calibration", "evaluation", "format_pairs"):
        raise ValueError("Unknown study partition")
    targets = {target.name: target for target in load_catalog(catalog)}
    if not names or set(names) - set(targets):
        raise ValueError("Unknown model targets")
    if config.execution.device == "cpu":
        raise ValueError(
            "Research collection requires an accelerator; CPU model comparison is disabled"
        )
    rows = load_study_split(config, partition)
    root = config.paths.output
    root.mkdir(parents=True, exist_ok=True)
    plan = root / "plan.json"
    if plan.exists() and read_json(plan) != PLAN:
        raise ValueError("Research plan changed; use a new run directory")
    write_json(plan, PLAN)
    selection = read_json(root / "selection.json") if interventions else None
    if interventions and partition != "evaluation":
        raise ValueError("Frozen interventions are evaluated only on the evaluation split")
    conditions = conditions_from_selection(selection) if selection else {}
    completed = 0
    for name in names:
        target = targets[name]
        marker = read_json(target.directory / "model_revision.json")
        if (marker["repository"], marker["revision"]) != (target.repository, target.revision):
            raise ValueError("Local model does not match its immutable target")
        engine = Engine(
            target.directory,
            device=config.execution.device,
            weight_cache="stream",
            resources=config.execution.resource_policy(),
        )
        if engine.selection.backend == "cpu":
            raise ValueError("No compliant accelerator; research stopped without CPU inference")
        ids, _ = encode_rows(engine, rows, config.execution.max_tokens)
        discovery_ids = (
            ids
            if partition == "discovery"
            else encode_rows(
                engine, load_study_split(config, "discovery"), config.execution.max_tokens
            )[0]
        )
        frequency = np.bincount(np.concatenate(discovery_ids), minlength=engine.cfg.vocab_size)
        label = partition + ("-interventions" if interventions else "")
        directory = root / name / label
        directory.mkdir(parents=True, exist_ok=True)
        width = config.execution.microbatch_documents or 4
        identity = {
            "model": engine.store.fingerprint(),
            "target": {
                "name": name,
                "repository": target.repository,
                "revision": target.revision,
                "stage": target.stage,
                "lineage": target.lineage,
            },
            "rows": digest(rows),
            "frequency": hashlib.sha256(frequency.tobytes()).hexdigest(),
            "implementation": implementation(),
            "device": engine.selection.identity(),
            "max_tokens": config.execution.max_tokens,
            "batch_documents": width,
            "condition_group": config.execution.condition_group,
            "conditions": {key: value.record() for key, value in conditions.items()},
            "selection": digest(selection),
            "plan": digest(PLAN),
        }
        manifest_path = directory / "manifest.json"
        manifest = (
            read_json(manifest_path)
            if manifest_path.exists()
            else {
                "schema": 1,
                "identity": identity,
                "status": "running",
                "chunks": [],
                "documents": len(rows),
                "records": [
                    {key: row[key] for key in ("id", "domain", "source", "group_id")}
                    for row in rows
                ],
            }
        )
        if manifest["identity"] != identity:
            raise ValueError(
                "Study observation identity changed; retain old run and use a new output"
            )
        done = {chunk["start"]: chunk for chunk in manifest["chunks"]}
        for start in range(0, len(rows), width):
            filename = f"chunk-{start:05d}.npz"
            path = directory / filename
            if start in done:
                if (
                    not path.exists()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != done[start]["sha256"]
                ):
                    raise ValueError("Previously completed observation changed")
                continue
            before = perf_counter()
            arrays, _ = observe_documents(
                engine,
                rows[start : start + width],
                max_tokens=config.execution.max_tokens,
                layers=target.layers,
                seed=config.analysis.seed,
                microbatch_documents=width,
                discovery_counts=frequency,
                conditions=conditions,
                condition_group=config.execution.condition_group,
            )
            save_arrays(path, arrays)
            manifest["chunks"].append(
                {
                    "file": filename,
                    "start": start,
                    "documents": len(arrays["loss"]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "seconds": perf_counter() - before,
                    "peak_process_bytes": peak_process_bytes(),
                }
            )
            manifest["chunks"].sort(key=lambda item: item["start"])
            write_json(manifest_path, manifest)
            completed += 1
            print(
                f"{name}/{label}: {min(start + width, len(rows))}/{len(rows)} documents; "
                f"{manifest['chunks'][-1]['seconds']:.1f}s; peak RSS {peak_process_bytes() / 1024**3:.2f} GiB",
                flush=True,
            )
            if max_batches is not None and completed >= max_batches:
                return
        if engine.store.fingerprint() != identity["model"]:
            raise ValueError("Model changed during observation")
        load_study_split(config, partition)
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
        del engine
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None


def freeze_selection(config, name="olmoe-0924"):
    root = config.paths.output
    discovery, dm = load_observations(root / name / "discovery")
    calibration, cm = load_observations(root / name / "calibration")
    domains = sorted({row["domain"] for row in dm["records"]})
    labels = np.array([row["domain"] for row in dm["records"]])
    clabels = np.array([row["domain"] for row in cm["records"]])
    results = []
    for domain in domains:
        on = labels == domain
        score = discovery["mass"][0, on].mean(0) / np.maximum(
            discovery["mass"][0, ~on].mean(0), 1e-12
        )
        eligible = discovery["usage"][0, on].mean(0) >= 0.05
        ranking = sorted(
            range(len(score)), key=lambda expert: (not eligible[expert], -score[expert], expert)
        )
        pool = [
            expert
            for expert in range(len(score))
            if score[expert] <= np.median(score) and expert not in ranking[:8]
        ]
        if not pool:
            raise ValueError("No predeclared nonspecialist control candidates")
        on = clabels == domain
        features = np.stack(
            [
                calibration["mass"][0, on].mean(0),
                calibration["mass"][0, ~on].mean(0),
                calibration["usage"][0, on].mean(0),
                calibration["usage"][0, ~on].mean(0),
                calibration["norm"][0].mean(0),
            ],
            axis=1,
        )
        tolerances = np.array([0.1, 0.1, 0.1, 0.1, 0.2])
        candidates = []
        for expert in ranking[:8]:
            gaps = np.abs(features[pool] - features[expert]) / np.maximum(features[expert], 1e-12)
            index = min(
                range(len(pool)),
                key=lambda index: (np.max(gaps[index] / tolerances), int(pool[index])),
            )
            candidates.append(
                {
                    "expert": expert,
                    "control": pool[index],
                    "gaps": gaps[index].tolist(),
                    "balance_passed": bool((gaps[index] <= tolerances).all()),
                    "discovery_specificity": float(score[expert]),
                }
            )
        selected = next((item for item in candidates if item["balance_passed"]), candidates[0])
        results.append({"domain": domain, **selected, "candidate_audit": candidates})
    selection = {
        "schema": 1,
        "model": name,
        "plan": PLAN,
        "domains": results,
        "discovery_identity": digest(dm["identity"]),
        "calibration_identity": digest(cm["identity"]),
        "matching_features": ["on_mass", "off_mass", "on_usage", "off_usage", "raw_output_norm"],
    }
    path = root / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("Frozen selection differs; use a new run")
    write_json(path, selection)
    return selection
