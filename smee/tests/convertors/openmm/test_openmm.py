import numpy.random
import openff.interchange
import openff.toolkit
import openff.units
import openmm
import pytest
import torch

import smee
import smee.mm
import smee.potentials
import smee.tests.utils
from smee.converters.openmm import (
    convert_to_openmm_force,
    convert_to_openmm_system,
    convert_to_openmm_topology,
    create_openmm_system,
)

try:
    import smirnoff_plugins as _
    SMIRNOFF_PLUGINS_AVAILABLE = True
except ImportError:
    SMIRNOFF_PLUGINS_AVAILABLE = False


def _compute_energy(
    system: openmm.System,
    coords: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
) -> float:
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)

    context.setPositions(coords)

    state = context.getState(getEnergy=True)

    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalories_per_mole)


def _compare_smee_and_interchange(
    tensor_ff: smee.TensorForceField,
    tensor_system: smee.TensorSystem,
    interchange: openff.interchange.Interchange,
    coords: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
):
    system_smee = convert_to_openmm_system(tensor_ff, tensor_system)
    assert isinstance(system_smee, openmm.System)
    system_interchange = interchange.to_openmm(False, False)

    coords += (numpy.random.randn(*coords.shape) * 0.1) * openmm.unit.angstrom

    energy_smee = _compute_energy(system_smee, coords, box_vectors)
    energy_interchange = _compute_energy(system_interchange, coords, box_vectors)

    assert numpy.isclose(energy_smee, energy_interchange)


def test_create_openmm_system_v_sites(v_site_force_field):
    smiles = [
        "[H:3][C:2]([H:4])=[O:1]",
        "[Cl:3][C:2]([H:4])=[O:1]",
        "[H:2][O:1][H:3]",
        "[H:2][N:1]([H:3])[H:4]",
    ]

    interchange_full = openff.interchange.Interchange.from_smirnoff(
        v_site_force_field,
        openff.toolkit.Topology.from_molecules(
            [openff.toolkit.Molecule.from_mapped_smiles(pattern) for pattern in smiles]
        ),
    )

    system_interchange = interchange_full.to_openmm()
    n_particles = system_interchange.getNumParticles()

    force_field, topologies = smee.converters.convert_interchange(
        [
            openff.interchange.Interchange.from_smirnoff(
                v_site_force_field,
                openff.toolkit.Molecule.from_mapped_smiles(pattern).to_topology(),
            )
            for pattern in smiles
        ]
    )

    system_smee = create_openmm_system(
        smee.TensorSystem(topologies, [1] * len(smiles), False), force_field.v_sites
    )

    expected_v_site_idxs = [4, 9, 13, 14, 19]
    actual_v_site_idxs = [
        i for i in range(system_smee.getNumParticles()) if system_smee.isVirtualSite(i)
    ]
    assert actual_v_site_idxs == expected_v_site_idxs

    v_sites_interchange = [
        # interchange puts all v-sites at the end of a topology
        system_interchange.getVirtualSite(n_particles - 5 + i)
        for i in range(5)
    ]
    v_sites_smee = [system_smee.getVirtualSite(i) for i in expected_v_site_idxs]

    def compare_vec3(a: openmm.Vec3, b: openmm.Vec3):
        assert a.unit == b.unit
        assert numpy.allclose(
            numpy.array([*a.value_in_unit(a.unit)]),
            numpy.array([*b.value_in_unit(a.unit)]),
            atol=1.0e-5,
        )

    expected_particle_idxs = [
        [0, 1],
        [5, 6, 8],
        [10, 11, 12],
        [10, 12, 11],
        [15, 16, 17, 18],
    ]

    for i, (v_site_interchange, v_site_smee) in enumerate(
        zip(v_sites_interchange, v_sites_smee, strict=True)
    ):
        assert v_site_smee.getNumParticles() == v_site_interchange.getNumParticles()

        particles_smee = [
            v_site_smee.getParticle(i) for i in range(v_site_smee.getNumParticles())
        ]
        assert particles_smee == expected_particle_idxs[i]

        compare_vec3(
            v_site_smee.getLocalPosition(), v_site_interchange.getLocalPosition()
        )
        assert v_site_smee.getOriginWeights() == pytest.approx(
            v_site_interchange.getOriginWeights()
        )
        assert v_site_smee.getXWeights() == pytest.approx(
            v_site_interchange.getXWeights()
        )
        assert v_site_smee.getYWeights() == pytest.approx(
            v_site_interchange.getYWeights()
        )


@pytest.mark.parametrize("with_constraints", [True, False])
def test_convert_to_openmm_system_vacuum(with_constraints):
    # carbonic acid has impropers, 1-5 interactions so should test most convertors
    mol = openff.toolkit.Molecule.from_smiles("OC(=O)O")
    mol.generate_conformers(n_conformers=1)

    coords = mol.conformers[0].m_as(openff.units.unit.angstrom)
    coords = coords * openmm.unit.angstrom

    force_field = openff.toolkit.ForceField(
        "openff-2.0.0.offxml"
        if with_constraints
        else "openff_unconstrained-2.0.0.offxml"
    )
    interchange = openff.interchange.Interchange.from_smirnoff(
        force_field, mol.to_topology()
    )

    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    _compare_smee_and_interchange(tensor_ff, tensor_top, interchange, coords, None)


@pytest.mark.parametrize("with_constraints", [True, False])
def test_convert_to_openmm_system_periodic(with_constraints):
    ff = openff.toolkit.ForceField(
        "openff-2.0.0.offxml"
        if with_constraints
        else "openff_unconstrained-2.0.0.offxml"
    )
    top = openff.toolkit.Topology()

    interchanges = []

    n_copies_per_mol = [5, 5]

    # carbonic acid has impropers, 1-5 interactions so should test most convertors
    for smiles, n_copies in zip(["OC(=O)O", "O"], n_copies_per_mol, strict=True):
        mol = openff.toolkit.Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )
        interchanges.append(interchange)

        for _ in range(n_copies):
            top.add_molecule(mol)

    tensor_ff, tensor_tops = smee.converters.convert_interchange(interchanges)
    tensor_system = smee.TensorSystem(tensor_tops, n_copies_per_mol, True)

    coords, _ = smee.mm.generate_system_coords(
        tensor_system, None, smee.mm.GenerateCoordsConfig()
    )
    box_vectors = numpy.eye(3) * 20.0 * openmm.unit.angstrom

    top.box_vectors = box_vectors

    interchange_top = openff.interchange.Interchange.from_smirnoff(ff, top)

    _compare_smee_and_interchange(
        tensor_ff, tensor_system, interchange_top, coords, box_vectors
    )


@pytest.mark.parametrize("with_exception", [True, False])
def test_convert_lj_potential_with_exceptions(with_exception):
    system, vdw_potential, _ = smee.tests.utils.system_with_exceptions()

    vdw_potential.exceptions = {} if not with_exception else vdw_potential.exceptions

    forces = convert_to_openmm_force(vdw_potential, system)

    assert len(forces) == 2

    assert isinstance(forces[0], openmm.CustomNonbondedForce)
    assert isinstance(forces[1], openmm.CustomBondForce)

    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    expected_energy = smee.compute_energy_potential(system, vdw_potential, coords)

    omm_system = openmm.System()
    for _ in range(system.n_atoms):
        omm_system.addParticle(1.0)
    for force in forces:
        omm_system.addForce(force)

    context = openmm.Context(
        omm_system,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(coords.numpy() * openmm.unit.angstrom)

    energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(openmm.unit.kilocalorie_per_mole)
    )
    assert torch.isclose(
        torch.tensor(energy, dtype=expected_energy.dtype), expected_energy
    )


@pytest.mark.skipif(
    not SMIRNOFF_PLUGINS_AVAILABLE,
    reason="Requires SMIRNOFF plugins to be installed.",
)
def test_convert_to_openmm_system_dexp_periodic(test_data_dir):
    ff = openff.toolkit.ForceField(
        str(test_data_dir / "de-ff.offxml"), load_plugins=True
    )
    top = openff.toolkit.Topology()

    interchanges = []

    n_copies_per_mol = [5, 5]

    for smiles, n_copies in zip(["OCCO", "O"], n_copies_per_mol, strict=True):
        mol = openff.toolkit.Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )
        interchanges.append(interchange)

        for _ in range(n_copies):
            top.add_molecule(mol)

    tensor_ff, tensor_tops = smee.converters.convert_interchange(interchanges)
    tensor_system = smee.TensorSystem(tensor_tops, n_copies_per_mol, True)

    coords, _ = smee.mm.generate_system_coords(
        tensor_system, None, smee.mm.GenerateCoordsConfig()
    )
    box_vectors = numpy.eye(3) * 20.0 * openmm.unit.angstrom

    top.box_vectors = box_vectors

    interchange_top = openff.interchange.Interchange.from_smirnoff(ff, top)

    _compare_smee_and_interchange(
        tensor_ff, tensor_system, interchange_top, coords, box_vectors
    )


def test_convert_to_openmm_topology():
    formaldehyde_interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-2.0.0.offxml"),
        openff.toolkit.Molecule.from_smiles("C=O").to_topology(),
    )
    water_interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-2.0.0.offxml"),
        openff.toolkit.Molecule.from_smiles("O").to_topology(),
    )

    tensor_ff, [methane_top, water_top] = smee.converters.convert_interchange(
        [formaldehyde_interchange, water_interchange]
    )
    tensor_system = smee.TensorSystem([methane_top, water_top], [1, 2], True)

    openmm_topology = convert_to_openmm_topology(tensor_system)

    assert openmm_topology.getNumChains() == 2
    assert openmm_topology.getNumResidues() == 3  # 1 methane, 2 water

    residue_names = [residue.name for residue in openmm_topology.residues()]
    assert residue_names == ["UNK", "HOH", "HOH"]

    atom_names = [atom.name for atom in openmm_topology.atoms()]
    expected_atom_names = [
        "C",
        "O",
        "H1",
        "H2",
        "O",
        "H1",
        "H2",
        "O",
        "H1",
        "H2",
    ]
    assert atom_names == expected_atom_names

    bond_idxs = [
        (bond.atom1.index, bond.atom2.index, bond.order)
        for bond in openmm_topology.bonds()
    ]
    expected_bond_idxs = [
        (0, 1, 2),
        (0, 2, 1),
        (0, 3, 1),
        (4, 5, 1),
        (4, 6, 1),
        (7, 8, 1),
        (7, 9, 1),
    ]
    assert bond_idxs == expected_bond_idxs
