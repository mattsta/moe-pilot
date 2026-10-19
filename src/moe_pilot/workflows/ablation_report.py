"""Document-level ablation contrasts using explicit parent/family metadata."""

import hashlib

import numpy as np
import pandas as pd

from moe_pilot.interventions import Conditions
from moe_pilot.plotting import subplots
from moe_pilot.statistics import bootstrap_interval, document_means


def run(config, *, device=None):
    paths = config.paths
    definition = Conditions.load(paths.ablations / "conditions.json")
    checkpoint = paths.ablations / "checkpoint"
    valid = np.load(checkpoint / "valid.npy")
    domain = np.load(paths.ablations / "token_domain.npy")
    document = np.load(paths.ablations / "token_doc.npy")
    baseline = np.load(checkpoint / "losses/base.npy")
    if not (valid.shape == domain.shape == document.shape == baseline.shape):
        raise ValueError("Ablation token metadata and loss shapes disagree")
    document_domain = np.array(
        [domain[np.flatnonzero(document == index)[0]] for index in range(int(document.max()) + 1)]
    )
    differences = {}
    long_rows = []
    for item in definition.items:
        loss = np.load(checkpoint / f"losses/{item.name}.npy")
        if loss.shape != baseline.shape or not np.isfinite(loss).all():
            raise ValueError(f"Invalid token losses for {item.name}")
        delta = loss - baseline
        differences[item.name] = document_means(delta, document, valid)
        for index, name in enumerate(definition.domains):
            selected = valid & (domain == index)
            docs = document_domain == index
            long_rows.append(
                {
                    "condition": item.name,
                    "family": item.family,
                    "specialist_domain": item.domain,
                    "evaluation_domain": name,
                    "baseline_loss_token_mean": baseline[selected].mean(),
                    "condition_loss_token_mean": loss[selected].mean(),
                    "delta_loss_token_mean": delta[selected].mean(),
                    "delta_loss_document_mean": differences[item.name][docs].mean(),
                    "n_tokens": int(selected.sum()),
                    "n_documents": int(docs.sum()),
                }
            )
    directory = paths.tables / "ablations"
    directory.mkdir(parents=True, exist_ok=True)
    figures = paths.figures / "ablations"
    figures.mkdir(parents=True, exist_ok=True)
    long_path = directory / "delta_loss.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    rows = []
    specialists = [item for item in definition.items if item.family == "specialist"]
    for item in specialists:
        on = document_domain == definition.domains.index(item.domain)
        off = ~on
        specialist = differences[item.name]
        for family in ("global-control", "on-domain-control"):
            controls = [
                candidate
                for candidate in definition.items
                if candidate.parent == item.name and candidate.family == family
            ]
            if not controls:
                raise ValueError(f"Missing {family} family for {item.name}")
            control = np.mean([differences[c.name] for c in controls], axis=0)
            contrast = specialist[on] - control[on]
            seed = config.analysis.seed + int.from_bytes(
                hashlib.sha256(f"{item.name}:{family}".encode()).digest()[:4], "big"
            )
            low, high = bootstrap_interval(
                contrast, samples=config.analysis.bootstrap_samples, seed=seed
            )
            rows.append(
                {
                    "condition": item.name,
                    "domain": item.domain,
                    "scope": item.scope,
                    "expert_count": item.count,
                    "control_family": family,
                    "n_controls": len(controls),
                    "specialist_routing_mass": item.global_mass
                    if family == "global-control"
                    else item.on_domain_mass,
                    "control_routing_mass_mean": np.mean(
                        [
                            c.global_mass if family == "global-control" else c.on_domain_mass
                            for c in controls
                        ]
                    ),
                    "specialist_on_domain_delta": specialist[on].mean(),
                    "control_on_domain_delta": control[on].mean(),
                    "specialist_off_domain_delta": specialist[off].mean(),
                    "control_off_domain_delta": control[off].mean(),
                    "paired_difference": contrast.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "interval_method": "paired document bootstrap; unadjusted 95%",
                }
            )
    summary = pd.DataFrame(rows)
    summary_path = directory / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    matrix = np.array(
        [
            [
                differences[item.name][document_domain == index].mean()
                for index in range(len(definition.domains))
            ]
            for item in specialists
        ]
    )
    matrix_path = directory / "delta_loss_matrix.csv"
    pd.DataFrame(
        matrix, index=[item.name for item in specialists], columns=definition.domains
    ).to_csv(matrix_path)
    figure, axes = subplots(
        figsize=(max(6, len(definition.domains) * 1.4), max(3, len(specialists) * 0.25))
    )
    magnitude = max(float(np.abs(matrix).max()), 1e-12)
    image = axes.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-magnitude, vmax=magnitude)
    axes.set_xticks(range(len(definition.domains)), definition.domains)
    axes.set_yticks(range(len(specialists)), [item.name for item in specialists], fontsize=6)
    axes.set_title("Specialist ablations: document-mean Δ loss [nats/token]")
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure_path = figures / "delta_loss.png"
    figure.savefig(figure_path, dpi=130)
    return [long_path, summary_path, matrix_path, figure_path]
