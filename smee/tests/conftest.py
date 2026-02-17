import pathlib

import openff.interchange
import openff.toolkit
import openff.units
import pytest
import torch

_ANGSTROM = openff.units.unit.angstrom
_NM = openff.units.unit.nanometer

_DEGREES = openff.units.unit.degree

_KJ_PER_MOLE = openff.units.unit.kilojoules / openff.units.unit.mole
_KCAL_PER_MOLE = openff.units.unit.kilocalories / openff.units.unit.mole

_E = openff.units.unit.elementary_charge


@pytest.fixture
def toolkit_registry_rdkit_first():
    """Returns a toolkit registry with RDKit as the first toolkit."""
    return openff.toolkit.utils.ToolkitRegistry(
        toolkit_precedence=[
            openff.toolkit.utils.NAGLToolkitWrapper,
            openff.toolkit.utils.RDKitToolkitWrapper,
            openff.toolkit.utils.OpenEyeToolkitWrapper,
            openff.toolkit.utils.AmberToolsToolkitWrapper,
            openff.toolkit.utils.BuiltInToolkitWrapper,
        ]
    )


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def default_force_field() -> openff.toolkit.ForceField:
    """Returns the OpenFF 1.3.0 force field with constraints removed."""

    force_field = openff.toolkit.ForceField("openff-1.3.0.offxml")
    force_field.deregister_parameter_handler("Constraints")

    return force_field


@pytest.fixture(scope="module")
def ethanol() -> openff.toolkit.Molecule:
    """Returns an OpenFF ethanol molecule with a fixed atom order."""

    return openff.toolkit.Molecule.from_mapped_smiles(
        "[H:5][C:2]([H:6])([H:7])[C:3]([H:8])([H:9])[O:1][H:4]"
    )


@pytest.fixture(scope="module")
def ethanol_conformer(ethanol) -> torch.Tensor:
    """Returns a conformer [Å] of ethanol with an ordering which matches the
    ``ethanol`` fixture."""

    ethanol.generate_conformers(n_conformers=1)
    conformer = ethanol.conformers[0].m_as(_ANGSTROM)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def ethanol_interchange(
    ethanol, default_force_field, toolkit_registry_rdkit_first
) -> openff.interchange.Interchange:
    """Returns a parameterized system of ethanol."""
    return default_force_field.create_interchange(
        ethanol.to_topology(), toolkit_registry=toolkit_registry_rdkit_first
    )


@pytest.fixture(scope="module")
def formaldehyde() -> openff.toolkit.Molecule:
    """Returns an OpenFF formaldehyde molecule with a fixed atom order."""

    return openff.toolkit.Molecule.from_mapped_smiles("[H:3][C:1](=[O:2])[H:4]")


@pytest.fixture(scope="module")
def formaldehyde_conformer(formaldehyde) -> torch.Tensor:
    """Returns a conformer [Å] of formaldehyde with an ordering which matches the
    ``formaldehyde`` fixture."""

    formaldehyde.generate_conformers(n_conformers=1)
    conformer = formaldehyde.conformers[0].m_as(_ANGSTROM)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def formaldehyde_interchange(
    formaldehyde, default_force_field
) -> openff.interchange.Interchange:
    """Returns a parameterized system of formaldehyde."""

    return openff.interchange.Interchange.from_smirnoff(
        default_force_field, formaldehyde.to_topology()
    )


@pytest.fixture
def v_site_force_field() -> openff.toolkit.ForceField:
    force_field = openff.toolkit.ForceField()

    force_field.get_parameter_handler("Electrostatics")

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[*:1]",
            "epsilon": 0.0 * _KJ_PER_MOLE,
            "sigma": 1.0 * _ANGSTROM,
        }
    )

    charge_handler = force_field.get_parameter_handler("LibraryCharges")
    charge_handler.add_parameter(
        {"smirks": "[*:1]", "charge1": 0.0 * openff.units.unit.e}
    )

    vsite_handler = force_field.get_parameter_handler("VirtualSites")

    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[H][#6:2]([H])=[#8:1]",
            "name": "EP",
            "type": "BondCharge",
            "distance": 7.0 * _ANGSTROM,
            "match": "all_permutations",
            "charge_increment1": 0.2 * _E,
            "charge_increment2": 0.1 * _E,
            "sigma": 1.0 * _ANGSTROM,
            "epsilon": 2.0 / 4.184 * _KCAL_PER_MOLE,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#8:1]=[#6X3:2](-[#17])-[#1:3]",
            "name": "EP",
            "type": "MonovalentLonePair",
            "distance": 1.234 * _ANGSTROM,
            "outOfPlaneAngle": 25.67 * _DEGREES,
            "inPlaneAngle": 134.0 * _DEGREES,
            "match": "all_permutations",
            "charge_increment1": 0.0 * _E,
            "charge_increment2": 1.0552 * 0.5 * _E,
            "charge_increment3": 1.0552 * 0.5 * _E,
            "sigma": 0.0 * _NM,
            "epsilon": 0.5 * _KJ_PER_MOLE,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "name": "EP",
            "type": "DivalentLonePair",
            "distance": -3.21 * _NM,
            "outOfPlaneAngle": 37.43 * _DEGREES,
            "match": "all_permutations",
            "charge_increment1": 0.0 * _E,
            "charge_increment2": 1.0552 * 0.5 * _E,
            "charge_increment3": 1.0552 * 0.5 * _E,
            "sigma": 1.0 * _ANGSTROM,
            "epsilon": 0.5 * _KJ_PER_MOLE,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2][#7:1]([#1:3])[#1:4]",
            "name": "EP",
            "type": "TrivalentLonePair",
            "distance": 0.5 * _NM,
            "match": "once",
            "charge_increment1": 0.2 * _E,
            "charge_increment2": 0.0 * _E,
            "charge_increment3": 0.0 * _E,
            "charge_increment4": 0.0 * _E,
            "sigma": 1.0 * _ANGSTROM,
            "epsilon": 0.5 * _KJ_PER_MOLE,
        }
    )
    return force_field
