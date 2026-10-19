import numpy as np
import pytest

from moe_pilot.statistics import bootstrap_interval, document_means


def test_document_means_exclude_last_tokens_and_keep_documents_equally_weighted():
    values = np.array([1.0, 3.0, 999.0, 9.0, 999.0])
    documents = np.array([0, 0, 0, 1, 1])
    valid = np.array([True, True, False, True, False])
    np.testing.assert_allclose(document_means(values, documents, valid), [2, 9])


def test_bootstrap_is_deterministic_and_checks_inputs():
    assert bootstrap_interval([1, 2, 3], samples=100, seed=1) == bootstrap_interval(
        [1, 2, 3], samples=100, seed=1
    )
    assert bootstrap_interval([4, 4], samples=100, seed=1) == (4, 4)
    with pytest.raises(ValueError):
        bootstrap_interval([], samples=10, seed=1)
