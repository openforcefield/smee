import pytest

pytest.importorskip("openmm")

import numpy  # noqa: E402
import openmm.unit  # noqa: E402

from smee.mm._reporters import TensorReporter, tensor_reporter, unpack_frames  # noqa: E402


class TestTensorReporter:
    def test_describe_next(self, mocker):
        simulation = mocker.MagicMock()
        simulation.currentStep = 5

        reporter = TensorReporter(
            mocker.MagicMock(), 2, 1.0 / openmm.unit.kilocalories_per_mole, None
        )
        assert reporter.describeNextReport(simulation) == (1, True, False, False, True)

    def test_report(self, tmp_path, mocker):
        expected_potential = 1.0 * openmm.unit.kilocalories_per_mole
        expected_kinetic = 2.0 * openmm.unit.kilojoules_per_mole

        box_length = 3.0
        expected_volume = box_length**3 * openmm.unit.angstrom**3

        expected_box_vectors = numpy.eye(3) * box_length
        expected_coords = numpy.ones((1, 3))

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = expected_potential
        mock_state.getKineticEnergy.return_value = expected_kinetic
        mock_state.getPeriodicBoxVectors.return_value = (
            expected_box_vectors * openmm.unit.angstrom
        )
        mock_state.getPeriodicBoxVolume.return_value = expected_volume
        mock_state.getPositions.return_value = expected_coords * openmm.unit.angstrom

        expected_output_path = tmp_path / "output.msgpack"

        beta = 1.0 / (298.15 * openmm.unit.kelvin * openmm.unit.MOLAR_GAS_CONSTANT_R)
        pressure = 1.0 * openmm.unit.atmospheres

        with expected_output_path.open("wb") as file:
            reporter = TensorReporter(file, 1, beta, pressure)
            reporter.report(None, mock_state)

        with expected_output_path.open("rb") as file:
            frames = [*unpack_frames(file)]

        assert len(frames) == 1
        coords, box_vectors, reduced_potential, kinetic = frames[0]

        expected_reduced_potential = beta * (
            expected_potential
            + pressure * expected_volume * openmm.unit.AVOGADRO_CONSTANT_NA
        )

        assert coords == pytest.approx(expected_coords)
        assert box_vectors == pytest.approx(expected_box_vectors)

        assert isinstance(reduced_potential, float)
        assert reduced_potential == pytest.approx(expected_reduced_potential)

        assert isinstance(kinetic, float)
        assert kinetic == pytest.approx(
            expected_kinetic.value_in_unit(openmm.unit.kilocalories_per_mole)
        )

    @pytest.mark.parametrize(
        "potential, contains", [(numpy.nan, "nan"), (numpy.inf, "inf")]
    )
    def test_report_energy_check(self, potential, contains, mocker):
        potential = potential * openmm.unit.kilocalories_per_mole
        kinetic = 2.0 * openmm.unit.kilocalories_per_mole

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = potential
        mock_state.getKineticEnergy.return_value = kinetic

        beta = 1.0 / openmm.unit.kilocalories_per_mole

        with pytest.raises(ValueError, match=f"total energy is {contains}"):
            reporter = TensorReporter(mocker.MagicMock(), 1, beta, None)
            reporter.report(None, mock_state)


def test_tensor_reporter(tmp_path):
    output = tmp_path / "frames.msgpack"

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * 298.15 * openmm.unit.kelvin)

    pressure = 1.0 * openmm.unit.atmospheres

    with tensor_reporter(output, 2, beta, pressure) as reporter:
        assert isinstance(reporter, TensorReporter)

    assert output.exists() and output.is_file()
