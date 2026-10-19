import hashlib

import numpy as np
import pytest

from moe_pilot.artifacts import write_json
from moe_pilot.study import PLAN, conditions_from_selection, load_observations, save_arrays


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
        chunks.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(tmp_path / "manifest.json", {"status": "complete", "chunks": chunks})
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
