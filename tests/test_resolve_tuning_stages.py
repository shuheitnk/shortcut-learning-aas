"""The tunings' units keep the fingerprints their runs are recorded under."""

from __future__ import annotations

import pytest

from aas_shortcut.measure_superiority_calibration import CalibrationConfig
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageUnit
from aas_shortcut.resolve_tuning_stages import (
    binding_provenance,
    calibration_unit,
    pair_tuning_unit,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    build_context,
    compose_config,
    resolve_run_paths,
)
from tests.test_run_experiment_spec import SCRIPTS


def _context(number: int):
    spec = ExperimentSpec.load(SCRIPTS / f"experiment{number}" / "config.yaml")
    cfg = compose_config(spec.hydra_overrides(spec.preset))
    return build_context(cfg, resolve_run_paths(cfg))


@pytest.fixture(scope="module")
def experiment2():
    return _context(2)


@pytest.fixture(scope="module")
def experiment3():
    return _context(3)


class TestCalibration:
    def test_it_is_chained_to_the_superiorities_it_reads(self, experiment2, tmp_path):
        superiority = StageArtifact(
            StageUnit("view_superiority", {"stage": "view_superiority"}), tmp_path, {}
        )
        unit, config = calibration_unit(experiment2, superiority)
        assert unit.inputs == {"superiority": superiority.fingerprint}
        assert unit.settings == CalibrationConfig(dim=experiment2.dim).as_dict()
        assert config.dim == experiment2.dim


class TestProvenance:
    def test_a_binding_records_the_tuning_it_came_from(self, experiment2, tmp_path):
        tuning = StageArtifact(pair_tuning_unit(experiment2), tmp_path, {})
        assert binding_provenance(tuning, "E2-1") == {
            "binding": {
                "stage": "tune_pair",
                "fingerprint": "ccdef07d6e569b5b",
                "step": "E2-1",
            }
        }
