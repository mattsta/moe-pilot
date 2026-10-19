"""Document-level statistics for token loss comparisons."""

import numpy as np


def document_means(values, documents, valid):
    count = np.bincount(documents[valid], minlength=int(documents.max()) + 1)
    if (count == 0).any():
        raise ValueError("Every document needs a valid next-token target")
    return np.bincount(documents[valid], weights=values[valid], minlength=len(count)) / count


def bootstrap_interval(values, *, samples, seed):
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    values = np.asarray(values)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite document observations")
    random = np.random.default_rng(seed)
    # Bound allocation even for large corpora and bootstrap counts.
    means = np.concatenate(
        [
            values[random.integers(0, len(values), (min(256, samples - start), len(values)))].mean(
                1
            )
            for start in range(0, samples, 256)
        ]
    )
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))
