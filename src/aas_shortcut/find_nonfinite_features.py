"""Appendix: which features a block's feature shard found non-finite, and where.

The feature stage drops a column that is NaN or inf anywhere in its shard and
records every non-finite observation in the shard's manifest; the columns the
experiments read were fixed from all of the paper's blocks
(`define_feature_columns.EXCLUDED_COLUMNS`). This reads the manifests back, so the
rule can be applied to a subset of the blocks or instances.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def nonfinite_observations(
    manifest: Mapping[str, Any], block: str, instances: Iterable[int] | None = None
) -> pd.DataFrame:
    """One row per non-finite observation a shard's manifest records.

    ``instances`` keeps only those instances, for a shard that holds more than the
    block it is read as.
    """
    observed = manifest["diagnostics"]["nonfinite_observations"]
    rows = [
        {"block": block, "feature": feature, **observation}
        for feature, record in observed.items()
        for observation in record["observations"]
    ]
    for feature, record in observed.items():
        held = sum(1 for row in rows if row["feature"] == feature)
        if held != record["n_observations"]:
            raise ValueError(
                f"{block}: the manifest counts {record['n_observations']} non-finite "
                f"{feature} observation(s) but lists {held}"
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "block",
            "feature",
            "function_id",
            "instance_id",
            "transform",
            "feature_variant",
            "rep",
            "value",
        ],
    )
    if instances is not None:
        table = table[table["instance_id"].isin(list(instances))]
    return table.reset_index(drop=True)


def excluded_by(observations: pd.DataFrame, blocks: Iterable[str]) -> list[str]:
    """The features the non-finite rule drops when it reads only ``blocks``."""
    wanted = sorted(set(blocks))
    return sorted(
        set(observations.loc[observations["block"].isin(wanted), "feature"].astype(str))
    )
