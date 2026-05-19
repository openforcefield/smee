import pytest

pytest.importorskip("openmm")
pytest.importorskip("rdkit")

import numpy  # noqa: E402
import openff.interchange  # noqa: E402
import openff.interchange.models  # noqa: E402
import openff.toolkit  # noqa: E402
import openmm.app  # noqa: E402
import openmm.unit  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem  # noqa: E402

import smee  # noqa: E402
import smee.converters  # noqa: E402
import smee.mm  # noqa: E402
import smee.mm._utils  # noqa: E402
import smee.tests.utils  # noqa: E402
from smee.mm._mm import (
    _apply_hmr,
    _approximate_box_size,
    _energy_minimize,
    _generate_packmol_input,
    _get_platform,
    _get_state_log,
    _run_simulation,
    _topology_to_xyz,
    generate_system_coords,
    simulate,
)


@pytest.fixture()
def mock_omm_topology() -> openmm.app.Topology:
    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("UNK", chain)

    for _ in range(2):
        topology.addAtom("Ar", openmm.app.Element.getByAtomicNumber(18), residue)

    return topology


@pytest.fixture()
def mock_omm_system() -> openmm.System:
    system = openmm.System()

    for _ in range(2):
        system.addParticle(18.0 * openmm.unit.amu)

    force = openmm.NonbondedForce()

    for _ in range(2):
        force.addParticle(
            0.0, 3.0 * openmm.unit.angstrom, 1.0 * openmm.unit.kilojoule_per_mole
        )

    system.addForce(force)

    return system


def test_apply_hmr():
    interchanges = [
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField("tip4p_fb.offxml", "openff-2.1.0.offxml"),
            openff.toolkit.Molecule.from_smiles(smiles).to_topology(),
        )
        for smiles in ["O", "CO"]
    ]

    force_field, [topology_water, topology_meoh] = smee.converters.convert_interchange(
        interchanges
    )

    system_smee = smee.TensorSystem(
        [topology_water, topology_meoh, topology_water],
        [1, 2, 1],
        False,
    )
    system_openmm = smee.converters.convert_to_openmm_system(force_field, system_smee)

    for i in range(system_openmm.getNumParticles()):
        # Round the masses to the nearest integer to make comparisons easier.
        mass = system_openmm.getParticleMass(i).value_in_unit(openmm.unit.amu)
        mass = float(int(mass + 0.5)) * openmm.unit.amu
        system_openmm.setParticleMass(i, mass)

    _apply_hmr(system_openmm, system_smee)

    masses = [
        system_openmm.getParticleMass(i).value_in_unit(openmm.unit.amu)
        for i in range(system_openmm.getNumParticles())
    ]

    expected_masses = [
        # Water 1
        16.0,
        1.0,
        1.0,
        0.0,
        # MeOH 1
        15.5,
        0.5,
        1.5,
        1.5,
        1.5,
        1.5,
        # MeOH 2
        15.5,
        0.5,
        1.5,
        1.5,
        1.5,
        1.5,
        # Water 2
        16.0,
        1.0,
        1.0,
        0.0,
    ]
    assert masses == expected_masses


def test_topology_to_rdkit():
    expected_atomic_nums = [1, 1, 8, 6, 8, 1, 1]

    topology = smee.TensorTopology(
        atomic_nums=torch.tensor(expected_atomic_nums),
        formal_charges=torch.tensor([0, 0, 0, 0, -1, 0, 0]),
        bond_idxs=torch.tensor([[2, 1], [2, 0], [3, 4], [3, 5], [3, 6]]),
        bond_orders=torch.tensor([1, 1, 1, 1, 1]),
        parameters={},
        v_sites=None,
        constraints=None,
    )

    mol = smee.mm._utils.topology_to_rdkit(topology)
    assert Chem.MolToSmiles(mol) == "[H]C([H])[O-].[H]O[H]"

    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    assert atomic_nums == expected_atomic_nums

    assert mol.GetNumConformers() == 1


def test_topology_to_xyz(mocker):
    mock_molecule: Chem.Mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    mock_molecule.RemoveAllConformers()

    conformer = Chem.Conformer(mock_molecule.GetNumAtoms())
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    conformer.SetAtomPosition(2, (0.0, 1.0, 0.0))

    mock_molecule.AddConformer(conformer)

    mocker.patch("smee.mm._utils.topology_to_rdkit", return_value=mock_molecule)

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("tip4p_fb.offxml"),
        openff.toolkit.Molecule.from_rdkit(mock_molecule).to_topology(),
    )
    force_field, [topology] = smee.converters.convert_interchange(interchange)

    xyz = _topology_to_xyz(topology, force_field)

    expected_xyz = (
        "4\n"
        "\n"
        "O 0.000000 0.000000 0.000000\n"
        "H 1.000000 0.000000 0.000000\n"
        "H 0.000000 1.000000 0.000000\n"
        "X 0.074440 0.074440 0.000000"
    )

    assert xyz == expected_xyz


def test_approximate_box_size():
    system = smee.TensorSystem(
        [smee.tests.utils.topology_from_smiles("O")], [256], True
    )

    config = smee.mm.GenerateCoordsConfig(scale_factor=2.0)

    box_size = _approximate_box_size(system, config)

    assert isinstance(box_size, openmm.unit.Quantity)
    assert box_size.unit.is_compatible(openmm.unit.angstrom)

    box_size = box_size.value_in_unit(openmm.unit.angstrom)
    assert isinstance(box_size, float)

    expected_length = (256.0 * 18.01528 / 6.02214076e23 * 1.0e24) ** (1.0 / 3.0) * 2.0
    assert numpy.isclose(box_size, expected_length, atol=3)


def test_generate_packmol_input():
    expected_tolerance = 0.1 * openmm.unit.nanometer
    expected_seed = 42

    config = smee.mm.GenerateCoordsConfig(
        tolerance=expected_tolerance, seed=expected_seed
    )

    actual_input_file = _generate_packmol_input(
        [1, 2, 3], 1.0 * openmm.unit.angstrom, config
    )

    expected_input_file = "\n".join(
        [
            "tolerance 1.000000",
            "filetype xyz",
            "output output.xyz",
            "seed 42",
            "structure 0.xyz",
            "  number 1",
            "  inside box 0. 0. 0. 1.0 1.0 1.0",
            "end structure",
            "structure 1.xyz",
            "  number 2",
            "  inside box 0. 0. 0. 1.0 1.0 1.0",
            "end structure",
            "structure 2.xyz",
            "  number 3",
            "  inside box 0. 0. 0. 1.0 1.0 1.0",
            "end structure",
        ]
    )
    assert actual_input_file == expected_input_file


def test_generate_system_coords():
    coords, box_vectors = generate_system_coords(
        smee.TensorSystem(
            [
                smee.tests.utils.topology_from_smiles("O"),
                smee.tests.utils.topology_from_smiles("CO"),
            ],
            [1, 2],
            True,
        ),
        None,
        smee.mm.GenerateCoordsConfig(),
    )

    assert isinstance(coords, openmm.unit.Quantity)
    coords = coords.value_in_unit(openmm.unit.angstrom)
    assert isinstance(coords, numpy.ndarray)
    assert coords.shape == (3 + 6 * 2, 3)
    assert not numpy.allclose(coords, 0.0)

    assert isinstance(box_vectors, openmm.unit.Quantity)
    box_vectors = box_vectors.value_in_unit(openmm.unit.angstrom)
    assert isinstance(box_vectors, numpy.ndarray)
    assert box_vectors.shape == (3, 3)
    assert not numpy.allclose(box_vectors, 0.0)


def test_generate_system_coords_with_v_sites():
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("tip4p_fb.offxml"),
        openff.toolkit.Molecule.from_mapped_smiles("[H:2][O:1][H:3]").to_topology(),
    )

    force_field, [topology] = smee.converters.convert_interchange(interchange)
    system = smee.TensorSystem([topology], [1], False)

    coords, box_vectors = generate_system_coords(system, force_field)
    assert isinstance(coords, openmm.unit.Quantity)
    coords = coords.value_in_unit(openmm.unit.angstrom)
    assert isinstance(coords, numpy.ndarray)
    assert coords.shape == (3 + 1, 3)
    assert not numpy.allclose(coords, 0.0)


def test_get_state_log(mocker):
    energy = 1.0 * openmm.unit.kilocalorie_per_mole
    box_vectors = numpy.eye(3) * 10.0 * openmm.unit.angstrom

    state = mocker.MagicMock()
    state.getPotentialEnergy.return_value = energy
    state.getPeriodicBoxVectors.return_value = box_vectors

    actual_log = _get_state_log(state)
    expected_log = "energy=4.1840 kcal / mol volume=1000.0000 Å^3"

    assert expected_log == actual_log


@pytest.mark.parametrize("is_periodic", [True, False])
def test_get_platform(is_periodic):
    platform = _get_platform(is_periodic)
    assert isinstance(platform, openmm.Platform)

    assert (platform.getName() == "Reference") == (not is_periodic)


def test_energy_minimize(mock_omm_system):
    state = (
        numpy.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]) * openmm.unit.angstrom,
        numpy.eye(3) * 1.0 * openmm.unit.nanometer,
    )

    state_new = _energy_minimize(
        mock_omm_system,
        state,
        openmm.Platform.getPlatformByName("Reference"),
        smee.mm.MinimizationConfig(),
    )

    coords = state_new.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    assert coords.shape == (2, 3)


def test_run_simulation(mock_omm_topology, mock_omm_system):
    state = (
        numpy.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]) * openmm.unit.angstrom,
        numpy.eye(3) * 2.0 * openmm.unit.nanometer,
    )

    force: openmm.NonbondedForce = mock_omm_system.getForce(0)
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)

    state_new = _run_simulation(
        mock_omm_system,
        mock_omm_topology,
        state,
        openmm.Platform.getPlatformByName("Reference"),
        smee.mm.SimulationConfig(
            temperature=86.0 * openmm.unit.kelvin,
            pressure=1.0 * openmm.unit.atmosphere,
            n_steps=1,
        ),
    )
    assert isinstance(state_new, openmm.State)

    coords = state_new.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    assert coords.shape == (2, 3)


def test_simulate(mocker, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors

    mock_coords = numpy.array([[0.0, 0.0, 0.0]]) * openmm.unit.angstrom
    mock_box = numpy.eye(3) * 2.0 * openmm.unit.nanometer

    mock_state = mocker.MagicMock()
    mock_state.getPotentialEnergy.return_value = 1.0 * openmm.unit.kilocalorie_per_mole
    mock_state.getPeriodicBoxVectors.return_value = mock_box

    spied_energy_minimize = mocker.spy(smee.mm._mm, "_energy_minimize")
    spied_run_simulation = mocker.spy(smee.mm._mm, "_run_simulation")

    reporter = mocker.MagicMock()
    reporter.describeNextReport.return_value = (1, True, False, False, True)

    simulate(
        tensor_top,
        tensor_ff,
        mock_coords,
        mock_box,
        [
            smee.mm.MinimizationConfig(),
            smee.mm.SimulationConfig(
                temperature=86.0 * openmm.unit.kelvin, pressure=None, n_steps=1
            ),
        ],
        smee.mm.SimulationConfig(
            temperature=86.0 * openmm.unit.kelvin, pressure=None, n_steps=2
        ),
        [reporter],
        True,
    )

    spied_energy_minimize.assert_called_once()
    assert spied_run_simulation.call_count == 2

    assert reporter.report.call_count == 2


def test_simulate_invalid_pressure(mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors

    with pytest.raises(
        ValueError, match="pressure cannot be specified for a non-periodic"
    ):
        simulate(
            tensor_top,
            tensor_ff,
            numpy.zeros((0, 3)) * openmm.unit.angstrom,
            numpy.eye(3) * openmm.unit.angstrom,
            [],
            smee.mm.SimulationConfig(
                temperature=1.0 * openmm.unit.kelvin,
                pressure=1.0 * openmm.unit.bar,
                n_steps=2,
            ),
            [],
        )
