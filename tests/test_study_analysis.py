import numpy as np
import pytest

from moe_pilot.study_analysis import (
    contrast_weights,
    format_pair_indices,
    holm,
    interaction_inference,
    stage_internal_comparison,
)


def test_holm_is_monotone_in_sorted_probabilities():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_source_balance_and_constant_effect_have_zero_interaction():
    labels = np.tile(["a", "a", "b", "b", "c", "c"], 2)
    sources = np.repeat(["one", "two"], 6)
    weights, _ = contrast_weights(labels, sources, "a")
    assert np.isclose(weights.sum(), 0)
    assert np.isclose(weights[labels == "a"].sum(), 1)
    assert np.isclose(weights[labels != "a"].sum(), -1)
    stats = interaction_inference(
        np.ones(12), labels, sources, "a", seed=1, samples=100, permutations=99
    )
    assert abs(stats["interaction"]) < 1e-12
    assert abs(stats["ci_low"]) < 1e-12 and abs(stats["ci_high"]) < 1e-12
    assert stats["p"] == 1


def test_known_domain_effect_is_preserved_by_stratified_bootstrap():
    labels = np.tile(["a", "a", "b", "b"], 2)
    sources = np.repeat(["one", "two"], 4)
    stats = interaction_inference(
        (labels == "a").astype(float), labels, sources, "a", seed=1, samples=100, permutations=99
    )
    assert stats["interaction"] == 1
    assert stats["ci_low"] == stats["ci_high"] == 1


def test_confounding_is_rejected_for_source_adjusted_design():
    with pytest.raises(ValueError, match="crossed nonempty"):
        contrast_weights(np.array(["a", "b"]), np.array(["one", "two"]), "a")


def test_internal_profiles_remove_global_scale_without_inventing_a_shift():
    mass = np.array([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1]])
    norms = np.array([[2.0, 3.0, 4.0], [4.0, 3.0, 2.0]])
    divergence, change = stage_internal_comparison(mass, mass * 3, norms, norms * 2)
    np.testing.assert_allclose(divergence, 0, atol=1e-12)
    np.testing.assert_allclose(change, 0, atol=1e-12)


def test_format_pairing_rejects_content_drift_and_missing_variants():
    common = {
        "source_id": "q",
        "group_id": "g",
        "source": "s",
        "domain": "d",
        "aligned_tokens_sha256": "abc",
        "alignment_coverage": 0.99,
    }
    plain, chat = {**common, "format_variant": "plain"}, {**common, "format_variant": "chat"}
    np.testing.assert_array_equal(format_pair_indices([chat, plain]), [[1, 0]])
    with pytest.raises(ValueError, match="incomplete"):
        format_pair_indices([plain])
    with pytest.raises(ValueError, match="content and token alignment"):
        format_pair_indices([plain, {**chat, "aligned_tokens_sha256": "changed"}])


def test_analysis_preserves_all_intervention_modes_and_separates_diagnostics(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from moe_pilot.artifacts import write_json
    from moe_pilot.reports import read_table
    from moe_pilot.study import conditions_from_selection, digest, plan_for, save_arrays
    from moe_pilot.study_analysis import analyse_study

    config = SimpleNamespace(
        paths=SimpleNamespace(output=tmp_path),
        analysis=SimpleNamespace(seed=1, bootstrap_samples=50, permutations=19),
    )
    plan = plan_for(config)
    write_json(tmp_path / "plan.json", plan)
    domains = ("computing", "mathematics", "science")
    selection = {
        "model": "olmoe-0924",
        "plan": plan,
        "matching_features": ["on_mass", "off_mass", "on_usage", "off_usage", "raw_output_norm"],
        "domains": [
            {
                "domain": domain,
                "expert": 0,
                "control": 1,
                "balance_passed": False,
                "gaps": [0.2] * 5,
            }
            for domain in domains
        ],
    }
    write_json(tmp_path / "selection.json", selection)
    records = [
        {"id": str(index), "group_id": str(index), "source": source, "domain": domain}
        for index, (source, domain) in enumerate(
            (source, domain) for source in ("one", "two") for domain in domains for _ in range(2)
        )
    ]
    names = list(conditions_from_selection(selection))
    losses = np.ones((len(names), len(records)))
    for index, name in enumerate(names):
        for domain in domains:
            if not name.startswith(domain + "_"):
                continue
            on = np.array([row["domain"] == domain for row in records])
            for multiplier, mode in enumerate(("remove", "reroute", "rescale", "substitute"), 1):
                if name == f"{domain}_{mode}_specialist":
                    losses[index] += on * multiplier * 0.1
    directory = tmp_path / "olmoe-0924" / "evaluation-interventions"
    path = directory / "chunk-00000.npz"
    save_arrays(
        path,
        {
            "loss": np.ones(12),
            "mass": np.full((1, 12, 4), 0.25),
            "answer_loss": np.ones(12),
            "choice_accuracy": np.zeros(12),
            "choice_log_probability": -np.ones(12),
            "condition_names": np.array(names),
            "condition_loss": losses,
            "condition_answer_loss": losses,
            "features": np.zeros((12, 2)),
        },
    )
    write_json(
        directory / "manifest.json",
        {
            "status": "complete",
            "documents": 12,
            "records": records,
            "identity": {
                "target": {"stage": "base", "lineage": "0924"},
                "selection": digest(selection),
            },
            "chunks": [
                {
                    "file": path.name,
                    "start": 0,
                    "documents": 12,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    monkeypatch.setattr("moe_pilot.study_figures.build_study_figures", lambda root: None)
    result = analyse_study(config)
    assert result["causal_endpoints"] == 12
    effects = read_table(tmp_path / "tables/causal_effects.csv")
    assert len({(row["domain"], row["mode"]) for row in effects}) == 12
    assert sum(row["holm_p"] is not None for row in effects) == 6
    for row in effects:
        multiplier = ("remove", "reroute", "rescale", "substitute").index(row["mode"]) + 1
        assert row["interaction"] == pytest.approx(multiplier * 0.1)
    assert len(read_table(tmp_path / "tables/intervention_diagnostics.csv")) == 27
