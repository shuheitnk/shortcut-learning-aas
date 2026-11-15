"""Experiment 1's two cells: the blocks they are fitted on, and where they are read."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import aas_shortcut.resolve_predictor_cells as cells_module
from aas_shortcut.compare_predictor_sets import ALL_PREDICTORS, INVARIANT_PREDICTORS
from aas_shortcut.measure_augmented_regression import DEPLOYMENT_SHARD
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.resolve_predictor_cells import PredictorCells, comparison_sources
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageUnit


def artifact(root: Path, name: str) -> StageArtifact:
    """Make a cell's unit under ``root`` with an absolute path, as the store does."""
    unit = StageUnit("augmented_regression", {"cell": name})
    path = root / "_stages" / unit.stage / unit.fingerprint
    path.mkdir(parents=True)
    return StageArtifact(unit, path.resolve(), {"files": {}})


class TestComparisonSources:
    def test_each_half_is_read_from_its_own_unit(self, tmp_path):
        every, invariant = artifact(tmp_path, "all"), artifact(tmp_path, "invariant")
        source = comparison_sources(PredictorCells(invariant, every, ("a",)), tmp_path)
        assert source.all_cell.predictors == ALL_PREDICTORS
        assert source.invariant_cell.predictors == INVARIANT_PREDICTORS
        assert source.all_cell.run == str(every.path.relative_to(tmp_path))
        assert source.invariant_cell.run == str(invariant.path.relative_to(tmp_path))

    def test_each_half_reads_the_stage_s_own_table(self, tmp_path):
        cells = PredictorCells(
            artifact(tmp_path, "invariant"), artifact(tmp_path, "all"), ("a",)
        )
        for cell in comparison_sources(cells, tmp_path).cells:
            assert cell.deployments == DEPLOYMENT_SHARD

    def test_a_relative_output_root_reads_the_same_units(self, tmp_path, monkeypatch):
        """The store hands out absolute paths; `paths.output_root` is relative."""
        monkeypatch.chdir(tmp_path)
        every = artifact(tmp_path / "outputs", "all")
        invariant = artifact(tmp_path / "outputs", "invariant")
        source = comparison_sources(
            PredictorCells(invariant, every, ("a",)), Path("outputs")
        )
        assert source.all_cell.run == str(every.path.relative_to(tmp_path / "outputs"))
        root = source.invariant_cell.paths(Path("outputs")).root
        assert Path(root).resolve() == invariant.path


class TestTheBlocks:
    def test_the_training_block_is_resolved_for_plain_cells_alone(self, monkeypatch):
        """Both cells are plain, so the transformed training benchmark is not run."""
        asked: dict[str, object] = {}

        def training(store, context, binding, tuning, step, **kwargs):
            asked.update(kwargs)
            return "training"

        monkeypatch.setattr(
            cells_module,
            "resolve_pair_binding",
            lambda store, context, step: ("binding", "tuning"),
        )
        monkeypatch.setattr(cells_module, "resolve_training_block", training)
        monkeypatch.setattr(
            cells_module,
            "resolve_held_out_blocks",
            lambda store, context, binding, tuning, step: {"spline/warpheld": "block"},
        )
        blocks = cells_module.resolve_predictor_blocks(None, SimpleNamespace(), "E2-1")  # type: ignore[arg-type]
        assert asked == {"trainings": (TRAINING_PLAIN,)}
        assert blocks.training == "training"
