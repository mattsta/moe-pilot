"""Probability metrics shared by domain and subtopic reports (units: bits)."""

import numpy as np


def entropy_bits(probabilities, ax=-1):
    probabilities = np.clip(probabilities, 1e-12, 1)
    return -(probabilities * np.log2(probabilities)).sum(ax)


def jensen_shannon(p, q):
    return entropy_bits((p + q) / 2) - (entropy_bits(p) + entropy_bits(q)) / 2


def ratio(numerator, denominator, *, empty=np.nan):
    """Divide with an explicit value for undefined zero-denominator entries."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    output = np.full(numerator.shape, empty, dtype=float)
    return np.divide(numerator, denominator, out=output, where=denominator != 0)
