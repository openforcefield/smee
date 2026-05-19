import pytest

pytest.importorskip("openmm")

import openff.interchange  # noqa: E402
import openff.toolkit  # noqa: E402
import openff.units  # noqa: E402

import smee  # noqa: E402
import smee.converters  # noqa: E402


@pytest.fixture()
def mock_argon_params() -> tuple[openff.units.Quantity, openff.units.Quantity]:
    return (
        0.1 * openff.units.unit.kilojoules / openff.units.unit.mole,
        3.0 * openff.units.unit.angstrom,
    )


@pytest.fixture()
def mock_argon_ff(mock_argon_params) -> openff.toolkit.ForceField:
    epsilon, sigma = mock_argon_params

    ff = openff.toolkit.ForceField()
    ff.get_parameter_handler("Electrostatics")
    ff.get_parameter_handler("LibraryCharges").add_parameter(
        {
            "smirks": "[Ar:1]",
            "charge1": 0.0 * openff.units.unit.elementary_charge,
        }
    )
    ff.get_parameter_handler("vdW").add_parameter(
        {"smirks": "[Ar:1]", "epsilon": epsilon, "sigma": sigma}
    )
    return ff


@pytest.fixture()
def mock_argon_tensors(
    mock_argon_ff,
) -> tuple[smee.TensorForceField, smee.TensorTopology]:
    interchange = openff.interchange.Interchange.from_smirnoff(
        mock_argon_ff, openff.toolkit.Molecule.from_smiles("[Ar]").to_topology()
    )
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
    tensor_ff.potentials = [p for p in tensor_ff.potentials if p.type == "vdW"]

    return tensor_ff, tensor_top
