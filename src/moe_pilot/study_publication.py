"""Reader-facing study documents generated from the same tables as the HTML UI."""

from moe_pilot.artifacts import read_json
from moe_pilot.reports import read_table


def markdown_table(rows, columns):
    def value(item):
        if item is None:
            return "—"
        if isinstance(item, float):
            return f"{item:.6f}"
        return str(item).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(label for _, label in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(value(row[key]) for key, _ in columns) + " |" for row in rows),
        ]
    )


def write_reader_analysis(root):
    def load(name):
        path = root / "tables" / f"{name}.csv"
        return read_table(path) if path.exists() else []

    summary = load("model_summary")
    stages = load("posttraining_contrasts")
    effects = load("causal_effects")
    formats = load("format_effects")
    balance = load("control_balance")
    reconstruction = load("reconstruction_interactions")
    models = sorted({row["model"] for row in summary})
    document = [
        "# Source-balanced local OLMoE study\n",
        "Open [the interactive results explorer](index.html) for charts, filters, numeric tables, CSV downloads, and printable results. Figure exports include PNG, SVG, and PDF. This document is generated from the same validated tables.\n",
        "## Findings and scope\n",
        f"Completed plain-prompt evaluations: **{len(models)} checkpoints × 180 held-out questions**. Each checkpoint is pinned to an immutable revision. The six equally weighted source/domain cells contain MMLU and origin-filtered MMLU-Pro questions in computing, mathematics, and science.\n",
        markdown_table(
            [row for row in summary if row["metric"] == "accuracy"],
            [
                ("model", "Checkpoint"),
                ("estimate", "Forced-choice accuracy (fraction)"),
                ("ci_low", "95% low"),
                ("ci_high", "95% high"),
            ],
        )
        + "\n",
        "Accuracy means selecting the highest-scoring next-token answer letter among the supplied choices under a plain prompt. These are zero-shot subset measurements, **not official MMLU/MMLU-Pro scores or a general reasoning evaluation**. Loss per token covers the supplied question, options, and answer; normalized choice loss is a separate outcome in nats/question.\n",
    ]
    accuracy = [row for row in stages if row["metric"] == "accuracy"]
    if accuracy:
        supported = [row["model"] for row in accuracy if row["ci_low"] > 0]
        document.append(
            (
                "Positive paired accuracy intervals: " + ", ".join(supported)
                if supported
                else "No post-training checkpoint has a paired 95% accuracy interval wholly above zero relative to 0125 base in this plain-prompt cohort."
            )
            + " The intervals are unadjusted; failure to detect improvement does not establish equivalence.\n"
        )
        document.append(
            markdown_table(
                stages,
                [
                    ("model", "Checkpoint vs 0125 base"),
                    ("metric", "Metric"),
                    ("difference", "Difference"),
                    ("ci_low", "95% low"),
                    ("ci_high", "95% high"),
                ],
            )
            + "\n"
        )
    if balance:
        passed = sum(row["balance_passed"] in (True, "True") for row in balance)
        document.extend(
            [
                f"**{passed}/{len(balance)} domains passed every frozen control-balance threshold.** A failed match cannot support a semantic-specialization claim. The comparison remains useful as an explicitly exploratory intervention contrast. Selection used only discovery routing and calibration mass/usage/norm; final losses were not selection inputs.\n",
                markdown_table(
                    balance,
                    [
                        ("domain", "Domain"),
                        ("expert", "Expert"),
                        ("control", "Control"),
                        ("on_mass", "Relative on-mass gap"),
                        ("off_mass", "Relative off-mass gap"),
                        ("on_usage", "On-usage gap"),
                        ("off_usage", "Off-usage gap"),
                        ("raw_output_norm", "Norm gap"),
                    ],
                )
                + "\n",
            ]
        )
    if effects:
        primary = [row for row in effects if row["primary"] in (True, "True")]
        surviving = [row for row in primary if row["holm_p"] < 0.05]
        document.append(
            "**Primary interaction results:** "
            + (
                "; ".join(
                    f"{row['domain']} / {row['mode']} ({row['interaction']:+.6f} nats/token, Holm p={row['holm_p']:.4f})"
                    for row in surviving
                )
                + " clear the declared Holm threshold."
                if surviving
                else "None of the six primary interactions clears the declared Holm threshold."
            )
            + " This threshold result does not repair failed control matching or establish equivalence for undetected effects.\n"
        )
        document.extend(
            [
                "## Held-out intervention findings\n",
                "The 0924 model completed all 33 frozen intervention conditions on 180 held-out questions, plus baseline. Primary endpoints are three domains × removal/rerouting, with Holm correction across six tests. Rescaling and donor substitution are secondary. Restoration must reproduce baseline losses exactly before any contrast is analysed.\n",
                markdown_table(
                    effects,
                    [
                        ("domain", "Domain"),
                        ("mode", "Mode"),
                        ("interaction", "On−off specialist−control (nats/token)"),
                        ("ci_low", "95% low"),
                        ("ci_high", "95% high"),
                        ("holm_p", "Primary Holm p"),
                    ],
                )
                + "\n",
                "The source/domain-stratified paired bootstrap uses 10,000 replicates. The two-sided studentized permutation test permutes paired effects within source (9,999 permutations; plus-one tail estimate). Its interpretation requires the label-exchangeability/null assumptions; domain labels were not randomized. Confidence intervals are unadjusted. Failed control overlap remains a causal limitation even if a statistical contrast is positive.\n",
                "[Per-document contrasts](data/study/causal_paired_documents.csv), [source/domain effects](data/study/causal_by_source_domain.csv), and [half-strength and joint-removal diagnostics](data/study/intervention_diagnostics.csv) are available in full. Source/lexical sensitivity residualizes domain membership on source indicators and standardized recorded token-composition features; this is an exploratory conditional association and does not repair missing control overlap.\n",
                "[Gate and output diagnostics](data/study/intervention_gate_diagnostics.csv) retain targeted-token counts, selected mass before/after treatment, expert-set changes, and perturbation norms. They are input-token means per batch, including document-final positions; they are not document-weighted loss outcomes.\n",
                "The [correct-answer-label sensitivity](data/study/answer_label_effects.csv) uses already collected full-vocabulary correct-label losses. It is exploratory and separate from the frozen primary text-loss family, with its own Holm correction across twelve domain × mode tests. A correct-label loss is not task accuracy and does not score generated answers.\n",
                markdown_table(
                    load("answer_label_effects"),
                    [
                        ("domain", "Domain"),
                        ("mode", "Mode"),
                        ("interaction", "Answer-label on−off excess (nats)"),
                        ("ci_low", "95% low"),
                        ("ci_high", "95% high"),
                        ("exploratory_holm_p", "Exploratory Holm p"),
                    ],
                )
                + "\n",
            ]
        )
    else:
        document.append(
            "## Held-out intervention status\n\nThe new 33-condition intervention evaluation is not complete. No new causal endpoint is reported from partial observations.\n"
        )
    document.extend(
        [
            "## Routing and expert-output changes\n",
            markdown_table(
                [row for row in load("routing_shift") if row["domain"] == "all"],
                [
                    ("model", "Checkpoint vs 0125 base"),
                    ("mean_jsd_bits", "Mean per-question routing JSD (bits)"),
                    ("ci_low", "95% low"),
                    ("ci_high", "95% high"),
                ],
            )
            + "\n",
            "Routing distributions normalize selected gate mass within each question. The comparison uses identical token IDs across the 0125 stages. A routing shift alone does not demonstrate better reasoning, semantic partitioning, or causal necessity.\n",
            "[Raw selected gate mass](data/study/selected_gate_mass.csv) reports the per-question sum of selected layer-7 probabilities before that normalization, with source/domain-stratified uncertainty. It keeps gate scale visible alongside distributional changes.\n",
            "Expert-output magnitude uses a deterministic matched token position per question. Raw output norms are divided by the document's mean across experts before computing paired log ratios. Incoming hidden states differ across checkpoints. Expert indices are compared only within the 0125 lineage. This measures relative magnitude, not output direction, functional equivalence, or knowledge.\n",
            "## Native-chat protocol follow-up\n",
        ]
    )
    if formats:
        document.append(
            "Native-chat accuracy differences: "
            + "; ".join(
                f"{row['model']}: {100 * row['estimate']:+.2f} percentage points (95% interval {100 * row['ci_low']:+.2f} to {100 * row['ci_high']:+.2f})"
                for row in formats
                if row["metric"] == "accuracy"
            )
            + ". These are protocol effects on forced-letter scoring, not generated-answer quality.\n"
        )
        document.append(
            "Aligned question-span loss changes (chat minus plain): "
            + "; ".join(
                f"{row['model']}: {row['estimate']:+.6f} nats/token [{row['ci_low']:.6f}, {row['ci_high']:.6f}]"
                for row in formats
                if row["metric"] == "aligned_loss"
            )
            + ". A wrapper can change routing and question likelihood even when the measured question tokens are exactly identical.\n"
        )
        document.extend(
            [
                markdown_table(
                    formats,
                    [
                        ("model", "Checkpoint"),
                        ("metric", "Chat−plain metric (routing JSD unsigned)"),
                        ("estimate", "Estimate"),
                        ("ci_low", "95% low"),
                        ("ci_high", "95% high"),
                        ("pairs", "Pairs"),
                    ],
                )
                + "\n",
                "The wrapper is the checkpoint's recorded native user/assistant template. Question/choice strings and response-label strings stay fixed. The longest exactly identical contiguous question-token span covers at least 90% of the plain question prefix; loss scores only transitions inside that span, and routing samples only its tokens. Whole-prompt token losses are not contrasted because wrapper lengths differ. Deterministic probes use the same aligned position within each pair.\n",
                "This follow-up was chosen **after inspecting the plain-prompt stage results**. It is exploratory and within-checkpoint; it is not a causal training-algorithm comparison or a test of free-form generated answer quality. All intervals are paired across content groups and stratified by source and domain.\n",
            ]
        )
    else:
        document.append(
            "No native-chat checkpoint comparison has completed yet; partial runs are visible in the HTML execution table.\n"
        )
    if reconstruction:
        document.extend(
            [
                "## Archived-input reconstruction: corrected post-hoc interaction\n",
                markdown_table(
                    reconstruction,
                    [
                        ("domain", "Domain"),
                        ("interaction", "On−off excess (nats/token)"),
                        ("ci_low", "95% low"),
                        ("ci_high", "95% high"),
                        ("holm_p", "Holm p"),
                    ],
                )
                + "\n",
                "Only multilingual clears Holm's 0.05 threshold with the studentized test in this saved reconstruction. Science has Holm p=0.0508. The source/domain confounding and 22–93% control-mass gaps remain. This reanalysis uses saved arrays and no new model inference; it is post hoc, not a confirmatory semantic-specialization result.\n",
            ]
        )
    document.extend(
        [
            "## Sampling, reproducibility, and limits\n",
            "Discovery has 180 questions, calibration 120, evaluation 180; cells contribute 30/20/30 each. Questions are grouped across sources by normalized identity or word-five-gram Jaccard ≥0.8, with transitive groups kept within a single split. MMLU-derived origins are excluded from MMLU-Pro. The rendered plain input is at most 900 UTF-8 bytes. These checks limit textual overlap; they do not establish semantic independence or absence from pretraining. Source-specific difficulty and option counts still differ.\n",
            "Selection ranks discovery on/off routing mass ratios at layer 7. It takes the first of the top eight candidates admitting a calibration match; otherwise it retains the top expert and marks failure. Control candidates have discovery specificity at or below its median and exclude the top eight. Calipers are ≤10% for on/off mass and usage and ≤20% for output norm. No held-out outcome was used to relax these rules.\n",
            "Interventions are local to layer 7. Removal keeps the local routing choices and zeros the targeted contribution; rerouting excludes its router logit and permits replacements. Rescaling and donor substitution have separate declared semantics. Downstream states and routes can change. Half-strength curvature and joint-removal nonadditivity are descriptive diagnostics, not proof of redundant knowledge.\n",
            "Runs use local float32 streaming execution on MPS, two host threads, a 5 GiB process budget, and current 8 GiB available-memory reserve (earlier phases used 16 GiB). OS pressure stops inference. Admission can halve a batch before inference; it never retries a failed model run on CPU. Peak process RSS includes neither every system allocation nor a guarantee about other applications. Execution timings are operational records, not controlled speedup measurements.\n",
            "All intervals condition on one checkpoint per stage, this selected cohort, and the recorded protocol. The training-stage sequence is observational; algorithms and training data change together. No independent-family, training-trajectory, multilingual replication, generated-answer, or precision study is claimed by these outputs.\n",
            "Reproduce with the [operator runbook](../../docs/runbook.md). The [analysis record](data/study/analysis_record.json), [figure inventory](figures/manifest.json), [frozen plan](data/plan.json), and copied per-phase manifests record hashes, revisions, numerical dependencies, and producer identities. Retain the local raw arrays, source parquet files, frozen runtimes, model snapshots, configuration, and uv.lock to rerun inference. The HTML and figure exports require no model loading.\n",
        ]
    )
    (root / "analysis.md").write_text("\n".join(document), encoding="utf-8")
    (root / "README.md").write_text(
        "# Local OLMoE results\n\nOpen [the interactive report](index.html) for charts, filters, comparisons, and downloads. Read [the detailed analysis](analysis.md) for methods, uncertainty, and limits.\n\n"
        + f"Completed: {len(models)} plain-prompt checkpoints, {len(effects)} intervention contrasts, {len(formats)} native-chat protocol outcomes. Only completed, hash-validated observations enter these summaries.\n",
        encoding="utf-8",
    )
    return read_json(root / "tables" / "analysis_record.json")["completed_models"]
