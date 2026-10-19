import random

import numpy as np
import pytest

from moe_pilot.corpora import CorpusBuilder
from moe_pilot.metrics import entropy_bits, jensen_shannon, ratio


def test_corpus_sampling_is_deterministic_disjoint_and_deduplicated():
    texts = [f"text {index}" for index in range(20)]
    state = random.getstate()
    first, second = CorpusBuilder(5, 4), CorpusBuilder(5, 4)
    for builder in (first, second):
        builder.add("code", texts + texts, "source", min_length=1)
    assert first.records == second.records
    heldout = CorpusBuilder(5, 6, exclude=(r["text"] for r in first.records))
    heldout.add("code", texts, "source", min_length=1)
    assert not {r["text"] for r in first.records} & {r["text"] for r in heldout.records}
    assert random.getstate() == state


def test_short_corpus_source_fails_explicitly():
    with pytest.raises(ValueError, match="need 5 texts"):
        CorpusBuilder(5, 1).add("code", ["a"], "source", min_length=1)


def test_probability_metric_units_and_symmetry():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert entropy_bits(np.array([0.5, 0.5])) == pytest.approx(1)
    assert jensen_shannon(p, p) == pytest.approx(0)
    assert jensen_shannon(p, q) == pytest.approx(1)
    assert jensen_shannon(q, p) == pytest.approx(jensen_shannon(p, q))


def test_unused_experts_remain_undefined_and_plots_use_neutral_lift():
    counts = np.array([[0.0, 1.0], [0.0, 3.0]])
    normalized = ratio(counts, counts.sum(0))
    assert np.isnan(normalized[:, 0]).all()
    np.testing.assert_allclose(normalized[:, 1], [0.25, 0.75])
    np.testing.assert_allclose(ratio(counts, counts.mean(0), empty=1)[:, 0], 1)
