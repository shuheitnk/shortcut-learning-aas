"""Reading back where a feature shard found non-finite values."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.find_nonfinite_features import excluded_by, nonfinite_observations


def _manifest(instances: tuple[int, ...], feature: str = "ic.eps_ratio") -> dict:
    observations = [
        {
            "function_id": 4,
            "instance_id": instance,
            "transform": "post",
            "feature_variant": "value",
            "rep": 1,
            "value": "-inf",
        }
        for instance in instances
    ]
    return {
        "diagnostics": {
            "nonfinite_observations": {
                feature: {
                    "n_observations": len(observations),
                    "observations": observations,
                }
            }
        }
    }


def test_observations_outside_the_block_s_instances_are_dropped():
    table = nonfinite_observations(_manifest((3, 40)), "training", range(11, 111))
    assert table["instance_id"].tolist() == [40]


def test_the_rule_drops_what_the_blocks_read_hold():
    training = nonfinite_observations(_manifest((40,), "a.cond"), "training")
    power = nonfinite_observations(_manifest((4,)), "power/warpheld")
    both = pd.concat([training, power], ignore_index=True)
    assert excluded_by(both, ["training"]) == ["a.cond"]
    assert excluded_by(both, ["training", "power/warpheld"]) == [
        "a.cond",
        "ic.eps_ratio",
    ]


def test_a_manifest_that_lists_fewer_observations_than_it_counts_is_refused():
    manifest = _manifest((40,))
    manifest["diagnostics"]["nonfinite_observations"]["ic.eps_ratio"][
        "n_observations"
    ] = 2
    with pytest.raises(ValueError, match="counts 2"):
        nonfinite_observations(manifest, "training")
