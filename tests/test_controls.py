import numpy as np
import pytest

from moe_pilot.controls import matched_experts


def test_control_matching_preserves_original_sampling_procedure():
    mass = np.linspace(0.1, 0.9, 10)
    selected = [2, 3]
    rng = np.random.default_rng(91)
    pool = [e for e in range(len(mass)) if e not in selected]
    candidates = [rng.choice(pool, 2, replace=False) for _ in range(100)]
    expected = min(candidates, key=lambda c: abs(mass[c].sum() - mass[selected].sum()))
    actual = matched_experts(mass, selected, 0, random_seed=91, candidates=100)
    assert actual == sorted(expected)


def test_control_matching_excludes_specialists():
    mass = np.linspace(0.1, 0.9, 10)
    controls = matched_experts(mass, [2, 3], 0, random_seed=1, exclude=[4, 5], candidates=100)
    assert len(set(controls)) == 2
    assert not set(controls) & {2, 3, 4, 5}
    assert matched_experts(mass, [2], 0, random_seed=1) == [1]


def test_control_matching_fails_when_pool_is_too_small():
    with pytest.raises(ValueError, match="Insufficient"):
        matched_experts(np.ones(4), [0, 1], 0, random_seed=1, exclude=[2])
