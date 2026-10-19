"""Export scientific figures from analysed tables; never load model weights."""

import hashlib

import numpy as np

from moe_pilot.artifacts import write_json
from moe_pilot.plotting import subplots
from moe_pilot.reports import read_table

NAMES = {
    "olmoe-0924": "0924 base",
    "olmoe-0125-base": "0125 base",
    "olmoe-0125-sft": "0125 SFT",
    "olmoe-0125-dpo": "0125 DPO",
    "olmoe-0125-instruct": "0125 final",
}


def build_study_figures(root):
    tables, output = root / "tables", root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    inventory = {}

    def load(name):
        path = tables / f"{name}.csv"
        return read_table(path) if path.exists() else []

    def save(figure, name, sources, caption):
        figure.tight_layout(pad=1.6)
        files = {}
        for extension in ("png", "svg", "pdf"):
            path = output / f"{name}.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory[name] = {
            "caption": caption,
            "sources": {
                source: hashlib.sha256((tables / source).read_bytes()).hexdigest()
                for source in sources
            },
            "files": files,
        }
        figure.clear()

    summary = load("model_summary")
    if summary:
        figure, axes = subplots(1, 2, figsize=(12, 4.8))
        for ax, metric, scale, label in (
            (axes[0], "accuracy", 100, "Forced-choice accuracy (%)"),
            (axes[1], "choice_nll", 1, "Normalized choice loss (nats/question)"),
        ):
            rows = sorted(
                [row for row in summary if row["metric"] == metric],
                key=lambda row: list(NAMES).index(row["model"]),
            )
            estimates = np.array([row["estimate"] for row in rows]) * scale
            bounds = np.array([[row["ci_low"], row["ci_high"]] for row in rows]) * scale
            ax.errorbar(
                estimates,
                np.arange(len(rows)),
                xerr=np.stack((estimates - bounds[:, 0], bounds[:, 1] - estimates)),
                fmt="o",
                color="#006d77",
                capsize=4,
            )
            ax.set_yticks(np.arange(len(rows)), [NAMES[row["model"]] for row in rows])
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.grid(axis="x", alpha=0.2)
        figure.suptitle(
            "Same 180 held-out questions · plain prompts · 95% stratified bootstrap intervals",
            fontsize=12,
        )
        save(
            figure,
            "checkpoint_comparison",
            ["model_summary.csv"],
            "Selected MMLU/MMLU-Pro questions, zero-shot forced-letter scoring. Unadjusted source/domain-stratified intervals; not official benchmark scores.",
        )

    for name, title in (
        ("reconstruction_interactions", "Archived-input reconstruction · post-hoc interactions"),
        ("causal_effects", "New source-balanced study · exploratory interactions"),
    ):
        rows = load(name)
        if not rows:
            continue
        figure, ax = subplots(figsize=(9, max(4, 0.45 * len(rows) + 1.6)))
        values = np.array([row["interaction"] for row in rows])
        bounds = np.array([[row["ci_low"], row["ci_high"]] for row in rows])
        ax.errorbar(
            values,
            np.arange(len(rows)),
            xerr=np.stack((values - bounds[:, 0], bounds[:, 1] - values)),
            fmt="o",
            color="#006d77",
            capsize=4,
        )
        ax.set_yticks(np.arange(len(rows)), [f"{row['domain']} / {row['mode']}" for row in rows])
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_xlabel("On-domain minus off-domain specialist−control loss (nats/token)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        save(
            figure,
            name,
            [f"{name}.csv"],
            "Paired 95% bootstrap intervals are unadjusted. Read the table for primary-family Holm p values. Imperfect controls prevent semantic-specialization claims.",
        )

    norms = load("expert_norm_changes")
    if norms:
        models = list(dict.fromkeys(row["model"] for row in norms))
        matrix = np.array(
            [
                [row["mean_log_relative_norm_change"] for row in norms if row["model"] == model]
                for model in models
            ]
        )
        figure, ax = subplots(figsize=(12, 3.8))
        limit = max(float(np.abs(matrix).max()), 0.001)
        image = ax.imshow(
            matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
        )
        ax.set_yticks(range(len(models)), [NAMES[model] for model in models])
        ax.set_xlabel("Layer-7 expert index within the 0125 lineage")
        ax.set_title("Change in relative expert-output magnitude from 0125 base")
        figure.colorbar(image, ax=ax, label="Mean log relative norm ratio")
        save(
            figure,
            "expert_norm_changes",
            ["expert_norm_changes.csv"],
            "Same token positions, model-specific incoming hidden states. Each document's expert norms are divided by their mean before the paired log ratio. This measures magnitude, not function direction or causal necessity.",
        )

    formats = load("format_effects")
    if formats:
        figure, axes = subplots(1, 2, figsize=(11, 4))
        for ax, metric, scale, label in (
            (axes[0], "accuracy", 100, "Accuracy change (percentage points)"),
            (axes[1], "aligned_loss", 1, "Aligned question loss change (nats/token)"),
        ):
            rows = [row for row in formats if row["metric"] == metric]
            values = np.array([row["estimate"] for row in rows]) * scale
            bounds = np.array([[row["ci_low"], row["ci_high"]] for row in rows]) * scale
            ax.errorbar(
                values,
                np.arange(len(rows)),
                xerr=np.stack((values - bounds[:, 0], bounds[:, 1] - values)),
                fmt="o",
                color="#345ca0",
                capsize=4,
            )
            ax.set_yticks(range(len(rows)), [NAMES[row["model"]] for row in rows])
            ax.set_xlabel(label)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
            ax.grid(axis="x", alpha=0.2)
        figure.suptitle(
            "Native-chat minus plain · identical questions and aligned token spans", fontsize=12
        )
        save(
            figure,
            "format_effects",
            ["format_effects.csv"],
            "Exploratory protocol follow-up selected after the plain-prompt comparison. Paired unadjusted 95% intervals. The checkpoint and response-label strings are held fixed; routing and span loss use identical contiguous question tokens.",
        )
    write_json(output / "manifest.json", inventory)
    return inventory
