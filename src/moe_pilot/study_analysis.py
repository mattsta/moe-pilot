"""Paired, source-aware research inference from completed observations only."""

import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.metrics import jensen_shannon
from moe_pilot.statistics import document_means


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", newline="", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Holm correction requires finite probabilities")
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return result


def contrast_weights(labels, sources, target):
    labels, sources = np.asarray(labels), np.asarray(sources)
    domains, datasets = np.unique(labels), np.unique(sources)
    if target not in domains or len(domains) < 2:
        raise ValueError("Interaction requires an on-domain and off-domain population")
    weights = np.zeros(len(labels))
    strata = []
    for source in datasets:
        for domain in domains:
            indices = np.flatnonzero((sources == source) & (labels == domain))
            if not len(indices):
                raise ValueError("Source/domain interaction requires crossed nonempty cells")
            weights[indices] = (1 if domain == target else -1 / (len(domains) - 1)) / (
                len(datasets) * len(indices)
            )
            strata.append(indices)
    return weights, strata


def interaction_inference(
    effect, labels, sources, target, *, seed, samples=10000, permutations=9999
):
    effect, labels, sources = np.asarray(effect), np.asarray(labels), np.asarray(sources)
    if (
        effect.ndim != 1
        or not np.isfinite(effect).all()
        or len(effect) != len(labels)
        or len(effect) != len(sources)
    ):
        raise ValueError("Inference needs aligned finite paired document effects")
    weights, strata = contrast_weights(labels, sources, target)
    if any(len(indices) < 2 for indices in strata):
        raise ValueError(
            "Studentized inference requires at least two observations per source/domain cell"
        )
    random = np.random.default_rng(seed)
    bootstrap = np.zeros(samples)
    for indices in strata:
        for start in range(0, samples, 256):
            count = min(256, samples - start)
            draws = random.choice(indices, (count, len(indices)), replace=True)
            bootstrap[start : start + count] += effect[draws].mean(1) * weights[indices].sum()
    observed = float(weights @ effect)

    def statistic(values):
        variance = sum(
            weights[indices].sum() ** 2 * values[indices].var(ddof=1) / len(indices)
            for indices in strata
        )
        estimate = float(weights @ values)
        if variance <= 1e-30:
            return 0.0 if abs(estimate) <= 1e-14 else float(np.copysign(np.inf, estimate))
        return estimate / np.sqrt(variance)

    observed_statistic = statistic(effect)
    extreme = 0
    source_groups = [np.flatnonzero(sources == source) for source in np.unique(sources)]
    for _ in range(permutations):
        permuted = effect.copy()
        for indices in source_groups:
            permuted[indices] = random.permutation(permuted[indices])
        extreme += abs(statistic(permuted)) >= abs(observed_statistic) - 1e-14
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "interaction": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p": (extreme + 1) / (permutations + 1),
    }


def reconstruction_interactions(reconstruction, output):
    """Extend the old on-domain result to explicit on/off contrasts, without inference."""
    reconstruction, output = Path(reconstruction), Path(output)
    evidence = read_json(reconstruction / "evidence.json")
    source = Path(evidence["experiment"]["configuration"]["paths"]["output"])
    declared = {item["source"]: item["sha256"] for item in evidence["artifacts"]}

    def checked(relative):
        path = source / relative
        if (
            str(path) not in declared
            or hashlib.sha256(path.read_bytes()).hexdigest() != declared[str(path)]
        ):
            raise ValueError(f"Reconstruction input not validated: {relative}")
        return path

    valid = np.load(checked("ablations/checkpoint/valid.npy"), allow_pickle=False)
    docs = np.load(checked("ablations/token_doc.npy"), allow_pickle=False)
    domains = np.load(checked("ablations/token_domain.npy"), allow_pickle=False)
    baseline = np.load(checked("ablations/checkpoint/losses/base.npy"), allow_pickle=False)
    if not (baseline.shape == valid.shape == docs.shape == domains.shape):
        raise ValueError("Reconstruction token metadata disagree")
    conditions = read_json(checked("ablations/conditions.json"))
    labels = np.array(
        [
            conditions["domains"][int(domains[np.flatnonzero(docs == index)[0]])]
            for index in range(int(docs.max()) + 1)
        ]
    )
    # Source and domain are confounded here. A single pooled source permits
    # descriptive label permutations but is explicitly not source adjustment.
    sources = np.repeat("archived-confounded-population", len(labels))
    selected = [
        item
        for item in conditions["items"]
        if item["family"] == "specialist" and item["scope"] == "layer-7" and item["count"] == 1
    ]
    rows, document_rows = [], []
    for index, item in enumerate(selected):
        controls = [
            control
            for control in conditions["items"]
            if control["parent"] == item["name"] and control["family"] == "on-domain-control"
        ]
        specialist = np.load(
            checked(f"ablations/checkpoint/losses/{item['name']}.npy"), allow_pickle=False
        )
        control = np.mean(
            [
                np.load(
                    checked(f"ablations/checkpoint/losses/{control['name']}.npy"),
                    allow_pickle=False,
                )
                for control in controls
            ],
            axis=0,
        )
        effect = document_means(specialist - control, docs, valid)
        inference = interaction_inference(
            effect, labels, sources, item["domain"], seed=20261019 + index
        )
        on = labels == item["domain"]
        rows.append(
            {
                "domain": item["domain"],
                "mode": "reroute",
                "layer": 7,
                "expert_count": 1,
                "on_excess": float(effect[on].mean()),
                "off_excess": float(effect[~on].mean()),
                **inference,
                "mass_gap": abs(
                    item["on_domain_mass"] - np.mean([c["on_domain_mass"] for c in controls])
                )
                / item["on_domain_mass"],
                "interpretation": "post-hoc exploratory; source/domain confounded; imperfect mass matching",
            }
        )
        for document, value in enumerate(effect):
            document_rows.append(
                {
                    "specialist_domain": item["domain"],
                    "document": document,
                    "evaluation_domain": labels[document],
                    "paired_excess": float(value),
                }
            )
    for row, p in zip(rows, holm([row["p"] for row in rows]), strict=True):
        row["holm_p"] = float(p)
    write_table(output / "reconstruction_interactions.csv", rows)
    write_table(output / "reconstruction_paired_documents.csv", document_rows)
    write_json(
        output / "reconstruction_analysis.json",
        {
            "kind": "post-hoc-on-off-interaction",
            "input_evidence_sha256": hashlib.sha256(
                (reconstruction / "evidence.json").read_bytes()
            ).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bootstrap_samples": 10000,
            "permutations": 9999,
            "seed": 20261019,
            "family": "all five domains, layer 7, one expert, on-domain mass controls",
            "multiplicity": "Holm across five interactions",
            "permutation_statistic": "studentized interaction with cellwise variance; paired effects permuted within source",
            "status": "complete",
            "limits": "Exploratory reanalysis; no new model execution; domain/source confounded; matching imperfect",
        },
    )
    return rows


def analyse_study(config):
    from moe_pilot.study import PLAN, digest, load_observations, plan_for

    root = config.paths.output
    active_plan = plan_for(config)
    if read_json(root / "plan.json") != active_plan:
        raise ValueError("Configured analysis differs from the frozen research plan")
    tables = root / "tables"
    baselines, observed = [], {}
    for path in sorted(root.glob("*/evaluation*/manifest.json")):
        if path.parent.name not in ("evaluation", "evaluation-interventions"):
            continue
        if read_json(path).get("status") != "complete":
            continue
        arrays, record = load_observations(path.parent)
        name = path.parent.parent.name
        observed[name] = (arrays, record)
    summaries, documents, sources_table = [], [], []
    for name, (arrays, record) in observed.items():
        labels = np.array([row["domain"] for row in record["records"]])
        sources = np.array([row["source"] for row in record["records"]])
        ids = [row["group_id"] for row in record["records"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated content groups require cluster-aware inference")
        for domain in sorted(set(labels)):
            selected = labels == domain
            baselines.append(
                {
                    "model": name,
                    "stage": record["identity"]["target"]["stage"],
                    "domain": domain,
                    "documents": int(selected.sum()),
                    "loss": float(arrays["loss"][selected].mean()),
                    "answer_loss": float(arrays["answer_loss"][selected].mean()),
                    "accuracy": float(arrays["choice_accuracy"][selected].mean()),
                }
            )
            for source in sorted(set(sources)):
                selected = (labels == domain) & (sources == source)
                sources_table.append(
                    {
                        "model": name,
                        "source": source,
                        "domain": domain,
                        "documents": int(selected.sum()),
                        "loss": float(arrays["loss"][selected].mean()),
                        "accuracy": float(arrays["choice_accuracy"][selected].mean()),
                    }
                )
        for metric, values, unit in (
            ("accuracy", arrays["choice_accuracy"], "fraction correct"),
            ("loss", arrays["loss"], "nats/token"),
            ("choice_nll", -arrays["choice_log_probability"], "nats/question"),
        ):
            estimate, low, high = stratified_mean_interval(
                values,
                labels,
                sources,
                seed=config.analysis.seed,
                samples=config.analysis.bootstrap_samples,
            )
            summaries.append(
                {
                    "model": name,
                    "stage": record["identity"]["target"]["stage"],
                    "lineage": record["identity"]["target"]["lineage"],
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "unit": unit,
                    "documents": len(values),
                }
            )
        for index, row in enumerate(record["records"]):
            documents.append(
                {
                    "model": name,
                    **row,
                    "loss": float(arrays["loss"][index]),
                    "answer_loss": float(arrays["answer_loss"][index]),
                    "choice_nll": float(-arrays["choice_log_probability"][index]),
                    "correct": int(arrays["choice_accuracy"][index]),
                }
            )
    write_table(tables / "baseline_by_model_domain.csv", baselines)
    write_table(tables / "baseline_by_model_source_domain.csv", sources_table)
    write_table(tables / "model_summary.csv", summaries)
    write_table(tables / "model_paired_documents.csv", documents)
    comparisons, routing_rows, norm_rows = [], [], []
    if "olmoe-0125-base" in observed:
        base, bm = observed["olmoe-0125-base"]
        labels = np.array([row["domain"] for row in bm["records"]])
        sources = np.array([row["source"] for row in bm["records"]])
        for name, (arrays, record) in observed.items():
            if name == "olmoe-0125-base" or record["identity"]["target"]["lineage"] != "olmoe-0125":
                continue
            if record["records"] != bm["records"]:
                raise ValueError("Post-training comparisons need identical paired documents")
            same_tokens = (
                record["identity"]["token_ids_sha256"] == bm["identity"]["token_ids_sha256"]
            )
            routing_shift, norm_shift = stage_internal_comparison(
                base["mass"][0], arrays["mass"][0], base["norm"][0], arrays["norm"][0]
            )
            for domain in ("all", *sorted(set(labels))):
                selected = np.ones(len(labels), dtype=bool) if domain == "all" else labels == domain
                estimate, low, high = stratified_mean_interval(
                    routing_shift[selected],
                    labels[selected],
                    sources[selected],
                    seed=config.analysis.seed,
                    samples=config.analysis.bootstrap_samples,
                )
                routing_rows.append(
                    {
                        "model": name,
                        "reference": "olmoe-0125-base",
                        "domain": domain,
                        "layer": 7,
                        "mean_jsd_bits": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "documents": int(selected.sum()),
                    }
                )
            if same_tokens:
                for expert, value in enumerate(norm_shift.mean(0)):
                    norm_rows.append(
                        {
                            "model": name,
                            "reference": "olmoe-0125-base",
                            "expert": expert,
                            "mean_log_relative_norm_change": float(value),
                            "relative_norm_fold_change": float(np.exp(value)),
                            "interpretation": "matched token positions, model-specific incoming hidden states; relative output magnitude only",
                        }
                    )
            for metric, values in (
                ("accuracy", arrays["choice_accuracy"] - base["choice_accuracy"]),
                ("choice_nll", base["choice_log_probability"] - arrays["choice_log_probability"]),
                ("loss", arrays["loss"] - base["loss"]),
            ):
                if metric == "loss" and not same_tokens:
                    continue
                estimate, low, high = stratified_mean_interval(
                    values,
                    labels,
                    sources,
                    seed=config.analysis.seed,
                    samples=config.analysis.bootstrap_samples,
                )
                comparisons.append(
                    {
                        "model": name,
                        "reference": "olmoe-0125-base",
                        "metric": metric,
                        "difference": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "same_token_ids": same_tokens,
                        "interpretation": "observational paired stage contrast",
                    }
                )
    write_table(tables / "posttraining_contrasts.csv", comparisons)
    write_table(tables / "routing_shift.csv", routing_rows)
    write_table(tables / "expert_norm_changes.csv", norm_rows)
    effect_rows = []
    selection_file = root / "selection.json"
    if selection_file.exists():
        selection = read_json(selection_file)
        name = selection["model"]
        if name in observed and observed[name][0]["condition_names"].size:
            arrays, record = observed[name]
            if record["identity"]["selection"] != digest(selection):
                raise ValueError("Intervention outcomes do not match the frozen selection")
            labels = np.array([row["domain"] for row in record["records"]])
            sources = np.array([row["source"] for row in record["records"]])
            conditions = {
                str(name): arrays["condition_loss"][index]
                for index, name in enumerate(arrays["condition_names"])
            }
            for index, item in enumerate(selection["domains"]):
                domain = item["domain"]
                np.testing.assert_array_equal(conditions[f"{domain}_restored"], arrays["loss"])
                for mode in (*PLAN["primary_modes"], *PLAN["secondary_modes"]):
                    effect = (
                        conditions[f"{domain}_{mode}_specialist"]
                        - conditions[f"{domain}_{mode}_control"]
                    )
                    stats = interaction_inference(
                        effect,
                        labels,
                        sources,
                        domain,
                        seed=config.analysis.seed + index,
                        samples=config.analysis.bootstrap_samples,
                        permutations=config.analysis.permutations,
                    )
                    on = labels == domain
                    features = arrays["features"]
                    columns = (features - features.mean(0)) / np.maximum(features.std(0), 1e-12)
                    source_columns = (
                        np.column_stack([sources == source for source in sorted(set(sources))[1:]])
                        if len(set(sources)) > 1
                        else np.empty((len(labels), 0))
                    )
                    controls = np.column_stack((np.ones(len(labels)), source_columns, columns))
                    residual_on = on - controls @ np.linalg.lstsq(controls, on, rcond=None)[0]
                    denominator = float(residual_on @ on)
                    effect_rows.append(
                        {
                            "domain": domain,
                            "mode": mode,
                            "primary": mode in PLAN["primary_modes"],
                            **stats,
                            "balance_passed": item["balance_passed"],
                            "holm_p": None,
                            "on_excess": float(effect[on].mean()),
                            "off_excess": float(effect[~on].mean()),
                            "source_lexical_adjusted_coefficient": float(residual_on @ effect)
                            / denominator
                            if denominator > 1e-12
                            else None,
                            "interpretation": "exploratory; control balance failed"
                            if not item["balance_passed"]
                            else "conditional on this checkpoint and cohort",
                        }
                    )
            primary = [row for row in effect_rows if row["primary"]]
            for row, p in zip(primary, holm([row["p"] for row in primary]), strict=True):
                row["holm_p"] = float(p)
    write_table(tables / "causal_effects.csv", effect_rows)
    balance = []
    if selection_file.exists():
        for item in selection["domains"]:
            balance.append(
                {
                    "domain": item["domain"],
                    "expert": item["expert"],
                    "control": item["control"],
                    "balance_passed": item["balance_passed"],
                    **dict(zip(selection["matching_features"], item["gaps"], strict=True)),
                }
            )
    write_table(tables / "control_balance.csv", balance)
    write_json(
        tables / "analysis_record.json",
        {
            "status": "complete for listed outputs",
            "plan": active_plan,
            "completed_models": list(observed),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "permutation_statistic": "studentized source-stratified interaction",
            "limits": "Intervals condition on one checkpoint. Paired stage contrasts are observational. Failed matching prevents causal-specialization claims.",
            "producers": {name: record["identity"] for name, (_, record) in observed.items()},
            "table_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(tables.glob("*.csv"))
            },
        },
    )
    return {"completed_models": len(observed), "causal_endpoints": len(effect_rows)}


def stratified_mean_interval(values, labels, sources, *, seed, samples=10000):
    values, labels, sources = np.asarray(values), np.asarray(labels), np.asarray(sources)
    cells = [
        np.flatnonzero((labels == domain) & (sources == source))
        for source in np.unique(sources)
        for domain in np.unique(labels)
    ]
    if any(not len(indices) for indices in cells) or not np.isfinite(values).all():
        raise ValueError("Stratified means require finite observations and crossed cells")
    random = np.random.default_rng(seed)
    bootstrap = np.zeros(samples)
    for indices in cells:
        for start in range(0, samples, 256):
            draws = random.choice(indices, (min(256, samples - start), len(indices)), replace=True)
            bootstrap[start : start + len(draws)] += values[draws].mean(1) / len(cells)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(np.mean([values[indices].mean() for indices in cells])), float(low), float(high)


def stage_internal_comparison(base_mass, stage_mass, base_norm, stage_norm):
    matrices = [
        np.asarray(value, dtype=float) for value in (base_mass, stage_mass, base_norm, stage_norm)
    ]
    if any(
        value.ndim != 2
        or value.shape != matrices[0].shape
        or not np.isfinite(value).all()
        or (value < 0).any()
        for value in matrices
    ):
        raise ValueError(
            "Internal comparisons require aligned nonnegative document/expert matrices"
        )
    if any((value.sum(1) <= 0).any() for value in matrices):
        raise ValueError("Cannot normalize empty expert profiles")
    base_mass, stage_mass, base_norm, stage_norm = matrices
    divergence = jensen_shannon(
        base_mass / base_mass.sum(1, keepdims=True), stage_mass / stage_mass.sum(1, keepdims=True)
    )
    norm_change = np.log(
        np.maximum(stage_norm / stage_norm.mean(1, keepdims=True), 1e-12)
    ) - np.log(np.maximum(base_norm / base_norm.mean(1, keepdims=True), 1e-12))
    return divergence, norm_change
