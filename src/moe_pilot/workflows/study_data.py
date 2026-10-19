"""Prepare frozen discovery, calibration, evaluation, and diagnostic inputs."""

from moe_pilot.study_data import prepare_study


def run(config, *, device=None):
    return prepare_study(config)
