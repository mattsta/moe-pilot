import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from moe_pilot.artifacts import write_json
from moe_pilot.resources import ResourceAdmissionError
from moe_pilot.study import (
    PLAN,
    admit_batch,
    conditions_from_selection,
    load_observations,
    save_arrays,
)


def test_chunk_aggregation_validates_bytes_and_preserves_document_axis(tmp_path):
    chunks = []
    for start, count in ((0, 2), (2, 1)):
        path = tmp_path / f"chunk-{start}.npz"
        save_arrays(
            path,
            {
                "layers": np.array([7]),
                "loss": np.ones(count),
                "mass": np.ones((1, count, 4)),
                "unigram": np.ones(10),
                "condition_names": np.array(["remove"]),
                "condition_loss": np.ones((1, count)),
            },
        )
        chunks.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "start": start,
                "documents": count,
            }
        )
    write_json(
        tmp_path / "manifest.json",
        {
            "status": "complete",
            "chunks": chunks,
            "documents": 3,
            "records": [{"id": str(index)} for index in range(3)],
        },
    )
    arrays, _ = load_observations(tmp_path)
    assert arrays["mass"].shape == (1, 3, 4)
    assert arrays["condition_loss"].shape == (1, 3)
    np.testing.assert_array_equal(arrays["unigram"], np.full(10, 2))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        load_observations(tmp_path)


def test_frozen_interventions_include_distinct_semantics_and_exact_restoration():
    conditions = conditions_from_selection(
        {"domains": [{"domain": "science", "expert": 1, "control": 2}]}
    )
    assert len(conditions) == 11
    assert not conditions["science_restored"]
    assert conditions["science_remove_specialist"].mask[PLAN["layer"]] == (1,)
    assert conditions["science_reroute_control"].mode == "reroute"
    assert conditions["science_substitute_specialist"].donors[PLAN["layer"]][1] == 2
    assert conditions["science_half"].strength == 0.5
    assert conditions["science_joint"].mask[PLAN["layer"]] == (1, 2)


def test_admission_shrinks_before_inference_but_does_not_retry_pressure():
    attempted = []

    def check(size):
        attempted.append(size)
        if size > 2:
            raise ResourceAdmissionError("smaller batch required")

    engine = SimpleNamespace(
        batch=lambda rows, limit: rows,
        estimate_memory=lambda rows, conditions, group: len(rows),
        resources=SimpleNamespace(check=check),
    )
    assert admit_batch(engine, list(range(10)), 0, 8, 64, {}, 1) == 2
    assert attempted == [8, 4, 2]
    attempted.clear()

    def pressure(size):
        attempted.append(size)
        raise ValueError("OS pressure")

    engine.resources.check = pressure
    with pytest.raises(ValueError, match="OS pressure"):
        admit_batch(engine, list(range(10)), 0, 8, 64, {}, 1)
    assert attempted == [8]
