"""Benchmark units: the rule that builds them, and what they keep and partition."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from aas_shortcut.benchmark_focus_pairs import MEMBERS
from aas_shortcut.benchmark_objective_views import ViewBenchmarkConfig
from aas_shortcut.bind_portfolio_parameters import (
    CONDITION_SBS,
)
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS, FOCUS_PAIRS
from aas_shortcut.record_manifests import stage_fingerprint
from aas_shortcut.resolve_benchmark_units import (
    POST,
    PRE,
    assemble,
    benchmark_table,
    benchmark_unit,
    superiority_settings,
    superiority_table,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    build_context,
    compose_config,
    resolve_run_paths,
)
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP
from aas_shortcut.split_bbob_instances import (
    PROBLEM_CLASS,
    TEST_INSTANCES,
    TRAIN_INSTANCES,
)
from tests.conftest import published_binding
from tests.test_run_experiment_spec import SCRIPTS

BINDING = published_binding()
HELD_OUT = replace(DEFAULT_WARP, family="power", domain="warpheld")


@pytest.fixture(scope="module")
def context():
    """Experiment 4's context at the scale the paper reports."""
    spec = ExperimentSpec.load(SCRIPTS / "experiment4" / "config.yaml")
    cfg = compose_config(spec.hydra_overrides(spec.preset))
    return build_context(cfg, resolve_run_paths(cfg))


def _common(context) -> dict:
    return {
        "problem_class": PROBLEM_CLASS,
        "dim": context.dim,
        "max_functions": context.max_functions,
        "num_trials": context.num_trials,
        "base_seed": context.seed,
        "scale": context.scale,
        "tuning": BINDING.as_dict(),
        "n_jobs": context.n_jobs,
        "save_every": context.save_every,
    }


class TestUnits:
    def test_the_untransformed_view_is_built_under_the_default_warp(self, context):
        """`pre` runs the raw problem: a family's units share it with spline's."""
        held_out = benchmark_unit(
            context,
            algorithms=MEMBERS,
            binding=BINDING,
            view=PRE,
            block=TEST_INSTANCES,
            budget=context.search_budget,
            warp=HELD_OUT,
        )
        default = benchmark_unit(
            context,
            algorithms=MEMBERS,
            binding=BINDING,
            view=PRE,
            block=TEST_INSTANCES,
            budget=context.search_budget,
        )
        assert held_out.as_dict() == default.as_dict()

    def test_three_view_units_hold_exactly_one_shards_runs(self, context):
        old = ViewBenchmarkConfig(
            **_common(context),
            max_instances=max(TRAIN_INSTANCES),
            budget=context.search_budget,
            algorithms=MEMBERS,
            instances_by_transform={POST: TEST_INSTANCES},
        )
        units = [
            benchmark_unit(
                context,
                algorithms=MEMBERS,
                binding=BINDING,
                view=view,
                block=block,
                budget=context.search_budget,
            )
            for view, block in (
                (PRE, TRAIN_INSTANCES),
                (PRE, TEST_INSTANCES),
                (POST, TEST_INSTANCES),
            )
        ]
        jobs = [job for unit in units for job in unit.jobs()]
        assert len(jobs) == len(set(jobs)) == old.expected_rows()
        assert set(jobs) == set(old.jobs())

    def test_each_view_and_block_is_its_own_unit(self, context):
        fingerprints = {
            stage_fingerprint(
                benchmark_unit(
                    context,
                    algorithms=MEMBERS,
                    binding=BINDING,
                    view=view,
                    block=block,
                    budget=context.search_budget,
                ).as_dict()
            )
            for view in (PRE, POST)
            for block in (TRAIN_INSTANCES, TEST_INSTANCES)
        }
        assert len(fingerprints) == 4

    def test_the_tables_are_where_the_stages_write_them(self, context):
        unit = benchmark_unit(
            context,
            algorithms=ALGORITHMS,
            binding=BINDING,
            view=POST,
            block=TEST_INSTANCES,
            budget=context.total_budget,
            condition=CONDITION_SBS,
        )
        assert benchmark_table(unit) == (
            "performance/performance_views_sbs_bbob_10d.parquet"
        )
        assert superiority_table(10) == "superiority/view_superiority_bbob_10d.parquet"

    def test_the_superiorities_record_their_budget(self, context):
        settings = superiority_settings(context, FOCUS_PAIRS, context.search_budget)
        assert settings["budget"] == context.search_budget
        assert settings["pairs"] == [pair.slug for pair in FOCUS_PAIRS]


class TestAssemble:
    def test_units_are_joined_in_the_order_a_consolidated_shard_has(self):
        rows = pd.DataFrame(
            {
                "problem_class": "bbob",
                "function_id": [2, 1, 1],
                "instance_id": [1, 2, 1],
                "dimension": 10,
                "transform": ["pre", "pre", "post"],
                "pair": "a__b",
            }
        )
        joined = assemble([rows.iloc[:1], rows.iloc[1:]], "view_superiority")
        assert list(zip(joined["function_id"], joined["instance_id"], strict=True)) == [
            (1, 1),
            (1, 2),
            (2, 1),
        ]
