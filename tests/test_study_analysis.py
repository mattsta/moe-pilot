import numpy as np
import pytest

from moe_pilot.study_analysis import contrast_weights, holm, interaction_inference


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
