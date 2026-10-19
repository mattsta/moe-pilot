"""Headless figures without pyplot state or changes to the caller's backend."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def subplots(rows=1, columns=1, *, figsize, **options):
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns, **options)
