from dataclasses import replace

import pytest

from moe_pilot.interventions import Condition, Conditions


def conditions():
    specialist = Condition(
        "opaque_identifier", "arbitrary-domain", "layer-0", 1, "specialist", {0: [0]}, 0.2, 0.4
    )
    control = Condition(
        "unrelated_identifier",
        "arbitrary-domain",
        "layer-0",
        1,
        "global-control",
        {0: [1]},
        0.19,
        0.3,
        parent=specialist.name,
        seed=0,
    )
    return Conditions(("arbitrary-domain", "other"), 2, 4, 2, (specialist, control))


def test_explicit_condition_metadata_roundtrips_without_parsing_names(tmp_path):
    definition = conditions()
    path = tmp_path / "conditions.json"
    definition.save(path)
    assert Conditions.load(path) == definition


@pytest.mark.parametrize(
    "change",
    [
        {"parent": "missing"},
        {"mask": {0: [0]}},
        {"mask": {1: [1]}},
        {"count": 0},
        {"name": "../unsafe"},
        {"global_mass": float("nan")},
    ],
)
def test_invalid_control_definitions_fail_before_inference(change):
    definition = conditions()
    invalid = replace(definition.items[1], **change)
    with pytest.raises(ValueError):
        replace(definition, items=(definition.items[0], invalid)).validate()
