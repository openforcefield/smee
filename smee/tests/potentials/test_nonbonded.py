import copy
import math

import numpy
import openff
import openmm.unit
import pytest
import torch

from tholedipoleplugin import TholeDipoleForce

import smee
import smee.converters
import smee.converters.openmm
import smee.mm
import smee.tests.utils
import smee.utils
from smee.potentials.nonbonded import (
    _COULOMB_PRE_FACTOR,
    _compute_dexp_lrc,
    _compute_lj_lrc,
    _compute_pme_exclusions,
    compute_coulomb_energy,
    compute_dampedexp6810_energy,
    compute_dexp_energy,
    compute_lj_energy,
    compute_pairwise,
    compute_pairwise_scales,
    prepare_lrc_types,
)
from smee.potentials.multipole import compute_multipole_energy


def _compute_openmm_energy(
        system: smee.TensorSystem,
        coords: torch.Tensor,
        box_vectors: torch.Tensor | None,
        potential: smee.TensorPotential,
        polarization_type: str | None = None,
        return_omm_forces: bool | None = None,
) -> torch.Tensor:
    coords = coords.numpy() * openmm.unit.angstrom

    if box_vectors is not None:
        box_vectors = box_vectors.numpy() * openmm.unit.angstrom

    omm_forces = smee.converters.convert_to_openmm_force(potential, system)
    omm_system = smee.converters.openmm.create_openmm_system(system, None)

    # Handle polarization type for TholeDipoleForce
    if polarization_type is not None:
        for omm_force in omm_forces:
            if isinstance(omm_force, TholeDipoleForce):
                if polarization_type == "direct":
                    omm_force.setPolarizationType(TholeDipoleForce.Direct)
                elif polarization_type == "mutual":
                    omm_force.setPolarizationType(TholeDipoleForce.Mutual)
                elif polarization_type == "extrapolated":
                    omm_force.setPolarizationType(TholeDipoleForce.Extrapolated)
                else:
                    raise ValueError(f"Unknown polarization_type: {polarization_type}")

    for omm_force in omm_forces:
        omm_system.addForce(omm_force)

    if box_vectors is not None:
        omm_system.setDefaultPeriodicBoxVectors(*box_vectors)

    omm_integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    # Use Reference platform for TholeDipole (OpenCL may segfault)
    platform = openmm.Platform.getPlatformByName('Reference')
    omm_context = openmm.Context(omm_system, omm_integrator, platform)

    if box_vectors is not None:
        omm_context.setPeriodicBoxVectors(*box_vectors)

    omm_context.setPositions(coords)

    omm_energy = omm_context.getState(getEnergy=True).getPotentialEnergy()
    omm_energy = omm_energy.value_in_unit(openmm.unit.kilocalories_per_mole)

    # Get induced dipoles from TholeDipoleForce
    try:
        thole_force = None
        for force in omm_forces:
            if isinstance(force, TholeDipoleForce):
                thole_force = force
                break

        if thole_force:
            induced_dipoles = thole_force.getInducedDipoles(omm_context)

            # Convert from e·nm to e·Å (multiply by 10)
            conversion_factor = 10.0
            induced_dipoles_angstrom = [[d * conversion_factor for d in dipole] for dipole in induced_dipoles]
            print(f"\nOpenMM induced dipoles (e·Å):")
            for i, dipole in enumerate(induced_dipoles_angstrom):
                print(f"  Particle {i}: [{dipole[0]:.10f}, {dipole[1]:.10f}, {dipole[2]:.10f}]")

    except Exception as e:
        print(f"Could not get induced dipoles: {e}")

    if return_omm_forces:
        return torch.tensor(omm_energy, dtype=torch.float64), omm_forces
    else:
        return torch.tensor(omm_energy, dtype=torch.float64)


def _parameter_key_to_idx(potential: smee.TensorPotential, key: str):
    return next(iter(i for i, k in enumerate(potential.parameter_keys) if k.id == key))


def test_compute_pairwise_scales():
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [2, 3])

    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.attributes = torch.tensor(
        [0.01, 0.02, 0.5, 1.0, 9.0, 2.0], dtype=torch.float64
    )

    scales = compute_pairwise_scales(system, vdw_potential)

    # fmt: off
    expected_scale_matrix = torch.tensor(
        [
            [1.0, 0.01, 0.01, 0.01, 0.01] + [1.0] * (system.n_particles - 5),
            [0.01, 1.0, 0.02, 0.02, 0.02] + [1.0] * (system.n_particles - 5),
            [0.01, 0.02, 1.0, 0.02, 0.02] + [1.0] * (system.n_particles - 5),
            [0.01, 0.02, 0.02, 1.0, 0.02] + [1.0] * (system.n_particles - 5),
            [0.01, 0.02, 0.02, 0.02, 1.0] + [1.0] * (system.n_particles - 5),
            #
            [1.0] * 5 + [1.0, 0.01, 0.01, 0.01, 0.01] + [1.0] * (system.n_particles - 10),  # noqa: E501
            [1.0] * 5 + [0.01, 1.0, 0.02, 0.02, 0.02] + [1.0] * (system.n_particles - 10),  # noqa: E501
            [1.0] * 5 + [0.01, 0.02, 1.0, 0.02, 0.02] + [1.0] * (system.n_particles - 10),  # noqa: E501
            [1.0] * 5 + [0.01, 0.02, 0.02, 1.0, 0.02] + [1.0] * (system.n_particles - 10),  # noqa: E501
            [1.0] * 5 + [0.01, 0.02, 0.02, 0.02, 1.0] + [1.0] * (system.n_particles - 10),  # noqa: E501
            #
            [1.0] * 10 + [1.0, 0.01, 0.01] + [1.0] * (system.n_particles - 13),
            [1.0] * 10 + [0.01, 1.0, 0.02] + [1.0] * (system.n_particles - 13),
            [1.0] * 10 + [0.01, 0.02, 1.0] + [1.0] * (system.n_particles - 13),
            #
            [1.0] * 13 + [1.0, 0.01, 0.01] + [1.0] * (system.n_particles - 16),
            [1.0] * 13 + [0.01, 1.0, 0.02] + [1.0] * (system.n_particles - 16),
            [1.0] * 13 + [0.01, 0.02, 1.0] + [1.0] * (system.n_particles - 16),
            #
            [1.0] * 16 + [1.0, 0.01, 0.01],
            [1.0] * 16 + [0.01, 1.0, 0.02],
            [1.0] * 16 + [0.01, 0.02, 1.0],
        ],
        dtype=torch.float64
    )
    # fmt: on

    i, j = torch.triu_indices(system.n_particles, system.n_particles, 1)
    expected_scales = expected_scale_matrix[i, j]

    assert scales.shape == expected_scales.shape
    assert torch.allclose(scales, expected_scales)


def test_compute_pairwise_periodic():
    system = smee.TensorSystem(
        [
            smee.tests.utils.topology_from_smiles("[Ar]"),
            smee.tests.utils.topology_from_smiles("[Ne]"),
        ],
        [2, 3],
        True,
    )

    coords = torch.tensor(
        [
            [+0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [+4.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
            [+8.0, 0.0, 0.0],
        ]
    )
    box_vectors = torch.eye(3) * 24.0

    cutoff = torch.tensor(9.0)

    pairwise = compute_pairwise(system, coords, box_vectors, cutoff)

    expected_idxs = torch.tensor(
        [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4]],
        dtype=torch.int32,
    )
    n_expected_pairs = len(expected_idxs)

    assert pairwise.idxs.shape == (n_expected_pairs, 2)
    assert torch.allclose(pairwise.idxs, expected_idxs)
    assert pairwise.idxs.dtype == torch.int32

    assert pairwise.deltas.shape == (n_expected_pairs, 3)
    assert pairwise.deltas.dtype == torch.float32

    expected_distances = torch.tensor([4.0, 4.0, 8.0, 8.0, 4.0, 8.0, 4.0, 8.0])
    assert torch.allclose(pairwise.distances, expected_distances)
    assert pairwise.distances.shape == (n_expected_pairs,)
    assert pairwise.distances.dtype == torch.float32

    assert torch.isclose(cutoff, pairwise.cutoff)


@pytest.mark.parametrize("with_batch", [True, False])
def test_compute_pairwise_non_periodic(with_batch):
    system = smee.TensorSystem(
        [
            smee.tests.utils.topology_from_smiles("[Ar]"),
            smee.tests.utils.topology_from_smiles("[Ne]"),
        ],
        [2, 3],
        False,
    )

    coords = torch.tensor(
        [
            [+0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [+4.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
            [+8.0, 0.0, 0.0],
        ]
    )
    coords = coords if not with_batch else coords.unsqueeze(0)

    pairwise = compute_pairwise(system, coords, None, torch.tensor(9.0))

    expected_idxs = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
        ],
        dtype=torch.int32,
    )
    n_expected_pairs = len(expected_idxs)

    expected_batch_size = () if not with_batch else (1,)

    assert pairwise.idxs.shape == (n_expected_pairs, 2)
    assert torch.allclose(pairwise.idxs, expected_idxs)
    assert pairwise.idxs.dtype == torch.int32

    assert pairwise.deltas.shape == (*expected_batch_size, n_expected_pairs, 3)
    assert pairwise.deltas.dtype == torch.float32

    expected_distances = torch.tensor(
        [4.0, 4.0, 8.0, 8.0, 8.0, 4.0, 12.0, 12.0, 4.0, 16.0]
    )
    expected_distances = (
        expected_distances if not with_batch else expected_distances.unsqueeze(0)
    )

    assert torch.allclose(pairwise.distances, expected_distances)
    assert pairwise.distances.shape == (*expected_batch_size, n_expected_pairs)
    assert pairwise.distances.dtype == torch.float32

    assert pairwise.cutoff is None


@pytest.mark.parametrize("with_exceptions", [True, False])
def test_prepare_lrc_types(with_exceptions):
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [2, 3])

    vdw_potential = force_field.potentials_by_type["vdW"]
    assert len(vdw_potential.parameters) == 4

    expected_idxs_i = torch.tensor([0, 0, 0, 1, 1, 2, 0, 1, 2, 3])
    expected_idxs_j = torch.tensor([1, 2, 3, 2, 3, 3, 0, 1, 2, 3])

    # CH - HC -- 2 C, 8 H
    # CH - OH -- 2 C, 3 O
    # CH - HO -- 2 C, 6 H
    # HC - OH -- 8 H, 3 O
    # HC - HO -- 8 H, 6 H
    # OH - HO -- 3 O, 6 H
    # CH - CH -- 2 C  - count self interactions as well
    # HC - HC -- 8 H
    # OH - OH -- 3 O
    # HO - HO -- 6 H
    expected_n_ij = torch.tensor([16, 6, 12, 24, 48, 18, 3, 36, 6, 21]).double()

    param_idxs = {
        "CH": _parameter_key_to_idx(vdw_potential, "[#6X4:1]"),
        "HC": _parameter_key_to_idx(vdw_potential, "[#1:1]-[#6X4]"),
        "OH": _parameter_key_to_idx(vdw_potential, "[#1]-[#8X2H2+0:1]-[#1]"),
        "HO": _parameter_key_to_idx(vdw_potential, "[#1:1]-[#8X2H2+0]-[#1]"),
    }
    # sanity check that the order is the same as when the test was written
    assert [*param_idxs.values()] == [0, 1, 2, 3]

    if with_exceptions:
        smee.tests.utils.add_explicit_lb_exceptions(vdw_potential, system)
        assert len(vdw_potential.exceptions) == 10

    idxs_i, idxs_j, n_ij_interactions = prepare_lrc_types(system, vdw_potential)

    if with_exceptions:
        subset_idxs = [0, 1, 2, 13, 14, 25, 91, 92, 93, 94]

        idxs_i = idxs_i[subset_idxs]
        idxs_j = idxs_j[subset_idxs]

        n_ij_interactions = n_ij_interactions[subset_idxs]

    assert idxs_i.shape == expected_idxs_i.shape
    assert torch.allclose(idxs_i, expected_idxs_i)

    assert idxs_j.shape == expected_idxs_j.shape
    assert torch.allclose(idxs_j, expected_idxs_j)

    assert n_ij_interactions.shape == expected_n_ij.shape
    assert torch.allclose(n_ij_interactions, expected_n_ij)


@pytest.mark.parametrize(
    "lrc_fn, convert_fn",
    [
        (_compute_lj_lrc, lambda p: p),
        (_compute_dexp_lrc, smee.tests.utils.convert_lj_to_dexp),
    ],
)
def test_compute_xxx_lrc_with_exceptions(lrc_fn, convert_fn):
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [50, 30])

    vdw_potential_no_exceptions = convert_fn(
        copy.deepcopy(force_field.potentials_by_type["vdW"])
    )
    assert vdw_potential_no_exceptions.exceptions is None

    rs = torch.tensor(8.0)
    rc = torch.tensor(9.0)

    volume = torch.tensor(18.0 ** 3)

    lrc_no_exceptions = lrc_fn(system, vdw_potential_no_exceptions, rs, rc, volume)

    vdw_potential_with_exceptions = convert_fn(
        copy.deepcopy(force_field.potentials_by_type["vdW"])
    )
    smee.tests.utils.add_explicit_lb_exceptions(vdw_potential_with_exceptions, system)
    assert len(vdw_potential_with_exceptions.exceptions) == 10

    lrc_with_exceptions = lrc_fn(system, vdw_potential_with_exceptions, rs, rc, volume)

    assert torch.isclose(lrc_no_exceptions, lrc_with_exceptions)


@pytest.mark.parametrize(
    "energy_fn,convert_fn,with_exceptions",
    [
        (compute_lj_energy, lambda p: p, False),
        (compute_lj_energy, lambda p: p, True),
        (compute_dexp_energy, smee.tests.utils.convert_lj_to_dexp, False),
        (compute_dexp_energy, smee.tests.utils.convert_lj_to_dexp, True),
    ],
)
def test_compute_xxx_energy_periodic(
        energy_fn, convert_fn, etoh_water_system, with_exceptions
):
    tensor_sys, tensor_ff, coords, box_vectors = etoh_water_system

    if with_exceptions:
        smee.tests.utils.add_explicit_lb_exceptions(
            tensor_ff.potentials_by_type["vdW"], tensor_sys
        )

    vdw_potential = convert_fn(tensor_ff.potentials_by_type["vdW"])
    vdw_potential.parameters.requires_grad = True

    energy = energy_fn(tensor_sys, vdw_potential, coords.float(), box_vectors.float())
    energy.backward()

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, box_vectors, vdw_potential
    )

    assert torch.isclose(energy, expected_energy, atol=1.0e-3)


@pytest.mark.parametrize(
    "energy_fn,convert_fn,with_exceptions",
    [
        (compute_lj_energy, lambda p: p, False),
        (compute_lj_energy, lambda p: p, True),
        (compute_dexp_energy, smee.tests.utils.convert_lj_to_dexp, False),
        (compute_dexp_energy, smee.tests.utils.convert_lj_to_dexp, True),
    ],
)
def test_compute_xxx_energy_non_periodic(energy_fn, convert_fn, with_exceptions):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(["CCC", "O"], [2, 3])
    tensor_sys.is_periodic = False

    if with_exceptions:
        smee.tests.utils.add_explicit_lb_exceptions(
            tensor_ff.potentials_by_type["vdW"], tensor_sys
        )

    coords, _ = smee.mm.generate_system_coords(tensor_sys, None)
    coords = torch.tensor(coords.value_in_unit(openmm.unit.angstrom))

    vdw_potential = convert_fn(tensor_ff.potentials_by_type["vdW"])
    vdw_potential.parameters.requires_grad = True

    energy = energy_fn(tensor_sys, vdw_potential, coords.float(), None)
    energy.backward()

    expected_energy = _compute_openmm_energy(tensor_sys, coords, None, vdw_potential)

    assert torch.isclose(energy, expected_energy, atol=1.0e-5)


def _expected_energy_lj_exceptions(params: dict[str, smee.tests.utils.LJParam]):
    sqrt_2 = math.sqrt(2)

    expected_energy = 4.0 * (
            params["oh"].eps * (params["oh"].sig ** 12 - params["oh"].sig ** 6)
            + params["oh"].eps * (params["oh"].sig ** 12 - params["oh"].sig ** 6)
            #
            + params["ah"].eps * (params["ah"].sig ** 12 - params["ah"].sig ** 6)
            + params["ah"].eps * (params["ah"].sig ** 12 - params["ah"].sig ** 6)
            #
            + params["hh"].eps
            * ((params["hh"].sig / sqrt_2) ** 12 - (params["hh"].sig / sqrt_2) ** 6)
            #
            + params["oa"].eps
            * ((params["oa"].sig / sqrt_2) ** 12 - (params["oa"].sig / sqrt_2) ** 6)
    )
    return expected_energy


def _expected_energy_dexp_exceptions(params: dict[str, smee.tests.utils.LJParam]):
    alpha = 16.5
    beta = 5.0

    def _dexp(eps, sig, dist):
        r_min = 2 ** (1 / 6) * sig

        x = dist / r_min

        rep = beta / (alpha - beta) * math.exp(alpha * (1.0 - x))
        atr = alpha / (alpha - beta) * math.exp(beta * (1.0 - x))

        return eps * (rep - atr)

    sqrt_2 = math.sqrt(2)

    expected_energy = (
            _dexp(params["oh"].eps, params["oh"].sig, 1.0)
            + _dexp(params["oh"].eps, params["oh"].sig, 1.0)
            #
            + _dexp(params["ah"].eps, params["ah"].sig, 1.0)
            + _dexp(params["ah"].eps, params["ah"].sig, 1.0)
            #
            + _dexp(params["hh"].eps, params["hh"].sig, sqrt_2)
            #
            + _dexp(params["oa"].eps, params["oa"].sig, sqrt_2)
    )
    return expected_energy


@pytest.mark.parametrize(
    "energy_fn,convert_fn,expected_fn",
    [
        (compute_lj_energy, lambda p: p, _expected_energy_lj_exceptions),
        (
                compute_dexp_energy,
                smee.tests.utils.convert_lj_to_dexp,
                _expected_energy_dexp_exceptions,
        ),
    ],
)
def test_compute_energy_exceptions_non_periodic(energy_fn, convert_fn, expected_fn):
    system, lj_potential, params = smee.tests.utils.system_with_exceptions()

    vdw_potential = convert_fn(lj_potential)

    # O --- H
    # |     |
    # H --- Ar
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    expected_energy = expected_fn(params)
    energy = energy_fn(system, vdw_potential, coords, None)

    assert torch.isclose(energy, torch.tensor(expected_energy, dtype=energy.dtype))


def test_coulomb_pre_factor():
    # Compare against a value computed directly from C++ using the OpenMM 7.5.1
    # ONE_4PI_EPS0 define constant multiplied by 10 for nm -> A
    _KCAL_TO_KJ = 4.184

    assert numpy.isclose(_COULOMB_PRE_FACTOR * _KCAL_TO_KJ, 1389.3545764, atol=1.0e-7)


def test_compute_pme_exclusions():
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [2, 3])

    coulomb_potential = force_field.potentials_by_type["Electrostatics"]
    exclusions = _compute_pme_exclusions(system, coulomb_potential)

    expected_exclusions = torch.tensor(
        [
            # C #1
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
            # C #2
            [6, 7, 8, 9],
            [5, 7, 8, 9],
            [5, 6, 8, 9],
            [5, 6, 7, 9],
            [5, 6, 7, 8],
            # O #1
            [11, 12, -1, -1],
            [10, 12, -1, -1],
            [10, 11, -1, -1],
            # O #2
            [14, 15, -1, -1],
            [13, 15, -1, -1],
            [13, 14, -1, -1],
            # O #3
            [17, 18, -1, -1],
            [16, 18, -1, -1],
            [16, 17, -1, -1],
        ]
    )

    assert exclusions.shape == expected_exclusions.shape
    assert torch.allclose(exclusions, expected_exclusions)


def test_compute_coulomb_energy_periodic(etoh_water_system):
    tensor_sys, tensor_ff, coords, box_vectors = etoh_water_system

    coulomb_potential = tensor_ff.potentials_by_type["Electrostatics"]
    coulomb_potential.parameters.requires_grad = True

    energy = compute_coulomb_energy(tensor_sys, coulomb_potential, coords, box_vectors)
    energy.backward()

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, box_vectors, coulomb_potential
    )

    assert torch.isclose(energy, expected_energy, atol=1.0e-2)


def test_compute_coulomb_energy_non_periodic():
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(["CCC", "O"], [2, 3])
    tensor_sys.is_periodic = False

    coords, _ = smee.mm.generate_system_coords(tensor_sys, None)
    coords = torch.tensor(coords.value_in_unit(openmm.unit.angstrom))

    coulomb_potential = tensor_ff.potentials_by_type["Electrostatics"]
    coulomb_potential.parameters.requires_grad = True

    energy = compute_coulomb_energy(tensor_sys, coulomb_potential, coords.float(), None)
    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, None, coulomb_potential
    )

    assert torch.isclose(energy, expected_energy, atol=1.0e-4)


def test_compute_dampedexp6810_energy_ne_scan_non_periodic(test_data_dir):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["[Ne]", "[Ne]"],
        [1, 1],
        openff.toolkit.ForceField(
            str(test_data_dir / "PHAST-H2CNO-nonpolar-2.0.0.offxml"), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = torch.stack(
        [
            torch.vstack(
                [
                    torch.tensor([0, 0, 0]),
                    torch.tensor([0, 0, 0]) + torch.tensor([0, 0, 1.5 + i * 0.5]),
                ]
            )
            for i in range(20)
        ]
    )

    energies = smee.compute_energy(tensor_sys, tensor_ff, coords)
    expected_energies = []
    for coord in coords:
        expected_energies.append(
            _compute_openmm_energy(
                tensor_sys, coord, None, tensor_ff.potentials_by_type["vdW"]
            )
        )
    expected_energies = torch.tensor(expected_energies)
    assert torch.allclose(energies, expected_energies.float(), atol=1.0e-4)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated")
    ]
)
def test_compute_multipole_energy_CC_O_non_periodic(test_data_dir, forcefield_name, polarization_type):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["CC", "O"],
        [3, 2],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    # Use fixed coordinates to ensure reproducibility
    coords = torch.tensor([
        [5.9731, 4.8234, 5.1358],
        [5.6308, 3.4725, 5.7007],
        [5.0358, 5.2467, 4.7020],
        [6.2522, 5.4850, 5.9780],
        [6.7136, 4.7967, 4.3256],
        [4.9936, 3.6648, 6.6100],
        [6.5061, 2.9131, 6.0617],
        [5.0991, 2.8173, 4.9849],
        [0.9326, 2.8105, 5.2711],
        [0.9434, 1.3349, 5.5607],
        [1.1295, 2.9339, 4.1794],
        [-0.0939, 3.1853, 5.4460],
        [1.7103, 3.3774, 5.7996],
        [0.0123, 0.9149, 5.0849],
        [0.8655, 1.0972, 6.6316],
        [1.8432, 0.8172, 5.1776],
        [3.2035, 0.7561, 3.0346],
        [3.4468, 1.0277, 1.5757],
        [4.1522, 0.3467, 3.4566],
        [3.0323, 1.7263, 3.5387],
        [2.4222, 0.0093, 3.2280],
        [4.1461, 1.9103, 1.5332],
        [2.5430, 1.3356, 1.0299],
        [3.8762, 0.1647, 1.0324],
        [6.3764, 1.9600, 2.9162],
        [6.1056, 1.3456, 3.6328],
        [6.6023, 2.8357, 3.3122],
        [3.0792, 6.2544, 4.6979],
        [3.5093, 6.6131, 5.5045],
        [3.5237, 6.6324, 3.9016]], dtype=torch.float64)

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]
    es_potential.parameters.requires_grad = True

    energy = compute_multipole_energy(tensor_sys, es_potential, coords.float(), None,
                                      polarization_type=polarization_type)
    energy.backward()
    expected_energy, omm_forces = _compute_openmm_energy(tensor_sys, coords, None, es_potential,
                                                         polarization_type=polarization_type, return_omm_forces=True)
    print_debug_info_multipole(energy, expected_energy, tensor_sys, es_potential, omm_forces)
    assert torch.allclose(energy, expected_energy, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        #("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        #("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        #("PHAST-H2CNO-2.0.0.offxml", "extrapolated")
    ]
)
def test_compute_multipole_energy_charged_ne_non_periodic(test_data_dir, forcefield_name, polarization_type):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["[Ne]", "[Ne]"],
        [1, 1],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = torch.vstack([torch.tensor([0, 0, 0]), torch.tensor([0, 0, 3.0])])

    # give each atom a charge otherwise the system is neutral
    es_potential = tensor_ff.potentials_by_type["Electrostatics"]
    es_potential.parameters[0, 0] = 1

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords, None, polarization_type=polarization_type
    )

    expected_energy, omm_forces = _compute_openmm_energy(tensor_sys, coords, None, es_potential,
                                                         polarization_type=polarization_type, return_omm_forces=True)
    print_debug_info_multipole(energy, expected_energy, tensor_sys, es_potential, omm_forces)
    assert torch.allclose(energy, expected_energy, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated")
    ]
)
def test_compute_multipole_energy_c_xe_non_periodic(test_data_dir, forcefield_name, polarization_type):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["C", "[Xe]"],
        [1, 1],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = torch.tensor(
        [
            [+0.00000, +0.00000, +0.00000],
            [+0.00000, +0.00000, +1.08900],
            [+1.02672, +0.00000, -0.36300],
            [-0.51336, -0.88916, -0.36300],
            [-0.51336, +0.88916, -0.36300],
            [+3.00000, +0.00000, +0.00000],
        ]
    )

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords, None, polarization_type=polarization_type
    )

    expected_energy, omm_forces = _compute_openmm_energy(tensor_sys, coords, None, es_potential,
                                                         polarization_type=polarization_type, return_omm_forces=True)
    print_debug_info_multipole(energy, expected_energy, tensor_sys, es_potential, omm_forces)
    assert torch.allclose(energy, expected_energy, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated")
    ]
)
def test_compute_phast2_energy_water_conformers_non_periodic(test_data_dir, forcefield_name, polarization_type):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["O"],
        [2],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = torch.tensor([[[-5.5964e-02, 8.1693e-01, -5.3445e-01],
                            [+2.5174e-01, -5.8659e-01, -8.1979e-01],
                            [+0.0000e+00, 0.0000e+00, 0.0000e+00],
                            [+7.6271e+00, -6.6103e-01, -5.7262e-01],
                            [+7.7119e+00, -4.1601e-01, 9.3098e-01],
                            [+7.9377e+00, 0.0000e+00, 0.0000e+00]],
                           [[+7.1041e-01, 4.7487e-01, -1.5602e-01],
                            [-4.8097e-01, 7.2769e-01, -2.2119e-01],
                            [+0.0000e+00, 0.0000e+00, 0.0000e+00],
                            [+8.1144e+00, -8.7009e-01, -3.9085e-01],
                            [+8.1329e+00, 9.2279e-01, -4.4597e-01],
                            [+7.9377e+00, 0.0000e+00, 0.0000e+00]],
                           [[+2.1348e-01, 3.6725e-01, 8.3273e-01],
                            [-5.7851e-01, -6.4377e-01, 5.4664e-01],
                            [+0.0000e+00, 0.0000e+00, 0.0000e+00],
                            [+7.2758e+00, 3.1414e-01, -5.9182e-01],
                            [+7.7279e+00, -5.7537e-01, 7.6088e-01],
                            [+7.9377e+00, 0.0000e+00, 0.0000e+00]]])

    multipole_potential = tensor_ff.potentials_by_type["Electrostatics"]
    vdw_potential = tensor_ff.potentials_by_type["vdW"]

    multipole_energy = compute_multipole_energy(
        tensor_sys, multipole_potential, coords, None, polarization_type=polarization_type
    )

    multipole_expected_energy = torch.tensor(
        [_compute_openmm_energy(tensor_sys, coord, None, multipole_potential, polarization_type=polarization_type) for coord in coords]
    )

    vdw_energy = compute_dampedexp6810_energy(
        tensor_sys, vdw_potential, coords, None
    )

    vdw_expected_energy = torch.tensor(
        [_compute_openmm_energy(tensor_sys, coord, None, vdw_potential) for coord in coords]
    )

    assert torch.allclose(multipole_energy, multipole_expected_energy, atol=1.0e-3)
    assert torch.allclose(vdw_energy, vdw_expected_energy, atol=1.0e-4)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated")
    ]
)
@pytest.mark.parametrize("smiles", ["CC", "CCC", "CCCC", "CCCCC"])
def test_compute_multipole_energy_isolated_non_periodic(test_data_dir, forcefield_name, polarization_type, smiles):

    print(f"\n{forcefield_name} - {polarization_type} - {smiles}")

    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        [smiles],
        [1],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords, _ = smee.mm.generate_system_coords(tensor_sys, None)
    coords = torch.tensor(coords.value_in_unit(openmm.unit.angstrom))

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]
    es_potential.parameters.requires_grad = True

    energy = compute_multipole_energy(tensor_sys, es_potential, coords.float(), None,
                                      polarization_type=polarization_type)
    energy.backward()
    expected_energy, omm_forces = _compute_openmm_energy(tensor_sys, coords, None, es_potential,
                                                         polarization_type=polarization_type, return_omm_forces=True)
    print_debug_info_multipole(energy, expected_energy, tensor_sys, es_potential, omm_forces)
    assert torch.allclose(energy, expected_energy, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated")
    ]
)
def test_compute_multipole_energy_CC_O_periodic(test_data_dir, forcefield_name, polarization_type):
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["CC", "O"],
        [10, 15],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = True

    # Use lower density to ensure box size >= 2*cutoff (18.0 Å)
    config = smee.mm.GenerateCoordsConfig(
        target_density=0.4 * openmm.unit.gram / openmm.unit.milliliter,
        scale_factor=1.3,
        padding=3.0 * openmm.unit.angstrom,
    )
    coords, box_vectors = smee.mm.generate_system_coords(tensor_sys, None, config)
    coords = torch.tensor(coords.value_in_unit(openmm.unit.angstrom))
    box_vectors = torch.tensor(box_vectors.value_in_unit(openmm.unit.angstrom))

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]
    es_potential.parameters.requires_grad = True

    energy = compute_multipole_energy(tensor_sys, es_potential, coords.float(), box_vectors.float(),
                                      polarization_type=polarization_type)
    energy.backward()
    expected_energy, omm_forces = _compute_openmm_energy(tensor_sys, coords, box_vectors, es_potential,
                                             polarization_type=polarization_type, return_omm_forces=True)
    print_debug_info_multipole(energy, expected_energy, tensor_sys, es_potential, omm_forces)
    assert torch.allclose(energy, expected_energy, atol=1e-3)


def print_debug_info_multipole(energy: torch.Tensor,
                               expected_energy: torch.Tensor,
                               tensor_sys: smee.TensorSystem,
                               es_potential: smee.TensorPotential,
                               omm_forces: list[openmm.Force]):
    print(f"Energy\nSMEE {energy} OpenMM {expected_energy}")

    print(f"SMEE Parameters {es_potential.parameters}")

    for idx, topology in enumerate(tensor_sys.topologies):
        print(f"SMEE Topology {idx}")
        print(f"Assignment Matrix {topology.parameters[es_potential.type].assignment_matrix.to_dense()}")

    thole_force = None
    for force in omm_forces:
        if isinstance(force, TholeDipoleForce):
            thole_force = force
            break

    print(thole_force)


# =============================================================================
# Water Model Tests - Testing SMEE with water force field from water_fitting
# =============================================================================

def _get_water_model_forcefield():
    """Load the water model force field."""
    import pathlib
    water_ff_path = pathlib.Path(__file__).parent.parent.parent.parent / "water_fitting" / "water_model.offxml"
    if not water_ff_path.exists():
        pytest.skip(f"Water model not found at {water_ff_path}")
    return openff.toolkit.ForceField(str(water_ff_path), load_plugins=True)


def _get_water_dimer_coords():
    """Return coordinates for a water dimer in Angstroms.

    First water at origin, second water displaced along z-axis.
    Standard water geometry: O-H bond length ~0.9572 Å, H-O-H angle ~104.5°
    """
    # Water 1 (near origin)
    o1 = [0.0, 0.0, 0.0]
    h1a = [0.9572, 0.0, 0.0]
    h1b = [-0.2399, 0.9270, 0.0]

    # Water 2 (displaced by ~2.8 Å along z-axis - typical H-bond distance)
    z_offset = 2.8
    o2 = [0.0, 0.0, z_offset]
    h2a = [0.9572, 0.0, z_offset]
    h2b = [-0.2399, 0.9270, z_offset]

    coords = torch.tensor([o1, h1a, h1b, o2, h2a, h2b], dtype=torch.float64)
    return coords


def _get_water_trimer_coords():
    """Return coordinates for a water trimer in Angstroms.

    Three water molecules in a triangular arrangement.
    """
    import math

    # Water 1 at origin
    o1 = [0.0, 0.0, 0.0]
    h1a = [0.9572, 0.0, 0.0]
    h1b = [-0.2399, 0.9270, 0.0]

    # Water 2 displaced along x
    x_offset = 3.0
    o2 = [x_offset, 0.0, 0.0]
    h2a = [x_offset + 0.9572, 0.0, 0.0]
    h2b = [x_offset - 0.2399, 0.9270, 0.0]

    # Water 3 displaced to form triangle
    angle = math.pi / 3  # 60 degrees
    r = 3.0
    o3 = [r * math.cos(angle), r * math.sin(angle), 0.0]
    h3a = [o3[0] + 0.9572, o3[1], 0.0]
    h3b = [o3[0] - 0.2399, o3[1] + 0.9270, 0.0]

    coords = torch.tensor([o1, h1a, h1b, o2, h2a, h2b, o3, h3a, h3b], dtype=torch.float64)
    return coords


@pytest.mark.parametrize("polarization_type", ["direct", "mutual", "extrapolated"])
def test_water_dimer_energy(polarization_type):
    """Test water dimer energy matches OpenMM/TholeDipoleForce."""
    force_field = _get_water_model_forcefield()

    water1 = openff.toolkit.Molecule.from_smiles("O")
    water2 = openff.toolkit.Molecule.from_smiles("O")
    topology = openff.toolkit.Topology.from_molecules([water1, water2])

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)
    coords = _get_water_dimer_coords()

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Compute SMEE energy
    energy = compute_multipole_energy(
        system, es_potential, coords.float(), None, polarization_type=polarization_type
    )

    # Compute OpenMM reference energy
    expected_energy = _compute_openmm_energy(
        system, coords, None, es_potential, polarization_type=polarization_type
    )

    print(f"\nWater dimer ({polarization_type}):")
    print(f"  SMEE Energy: {energy.item():.6f} kcal/mol")
    print(f"  OpenMM Energy: {expected_energy.item():.6f} kcal/mol")
    print(f"  Difference: {abs(energy.item() - expected_energy.item()):.2e} kcal/mol")

    assert torch.allclose(energy, expected_energy, atol=1e-4)


@pytest.mark.parametrize("polarization_type", ["direct", "mutual", "extrapolated"])
def test_water_dimer_forces(polarization_type):
    """Test water dimer forces via autograd match finite difference."""
    force_field = _get_water_model_forcefield()

    water1 = openff.toolkit.Molecule.from_smiles("O")
    water2 = openff.toolkit.Molecule.from_smiles("O")
    topology = openff.toolkit.Topology.from_molecules([water1, water2])

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)
    coords = _get_water_dimer_coords().float()
    coords.requires_grad = True

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Compute SMEE energy and forces
    energy = compute_multipole_energy(
        system, es_potential, coords, None, polarization_type=polarization_type
    )
    energy.backward()
    smee_forces = -coords.grad.detach()

    # Compute numerical forces via finite difference
    # Use h=1e-2 which is optimal for float32 precision (smaller h causes precision loss)
    h = 1e-2
    numerical_forces = torch.zeros_like(coords)

    for i in range(coords.shape[0]):
        for j in range(3):
            coords_plus = coords.detach().clone()
            coords_plus[i, j] += h
            e_plus = compute_multipole_energy(
                system, es_potential, coords_plus, None, polarization_type=polarization_type
            )

            coords_minus = coords.detach().clone()
            coords_minus[i, j] -= h
            e_minus = compute_multipole_energy(
                system, es_potential, coords_minus, None, polarization_type=polarization_type
            )

            numerical_forces[i, j] = -(e_plus - e_minus) / (2 * h)

    print(f"\nWater dimer forces ({polarization_type}):")
    print(f"  Max force difference (autograd vs numerical): {(smee_forces - numerical_forces).abs().max():.2e}")

    assert torch.allclose(smee_forces, numerical_forces, atol=1e-3)


@pytest.mark.parametrize("polarization_type", ["direct", "mutual", "extrapolated"])
def test_water_trimer_energy(polarization_type):
    """Test water trimer energy matches OpenMM/TholeDipoleForce."""
    force_field = _get_water_model_forcefield()

    waters = [openff.toolkit.Molecule.from_smiles("O") for _ in range(3)]
    topology = openff.toolkit.Topology.from_molecules(waters)

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)
    coords = _get_water_trimer_coords()

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Compute SMEE energy
    energy = compute_multipole_energy(
        system, es_potential, coords.float(), None, polarization_type=polarization_type
    )

    # Compute OpenMM reference energy
    expected_energy = _compute_openmm_energy(
        system, coords, None, es_potential, polarization_type=polarization_type
    )

    print(f"\nWater trimer ({polarization_type}):")
    print(f"  SMEE Energy: {energy.item():.6f} kcal/mol")
    print(f"  OpenMM Energy: {expected_energy.item():.6f} kcal/mol")
    print(f"  Difference: {abs(energy.item() - expected_energy.item()):.2e} kcal/mol")

    assert torch.allclose(energy, expected_energy, atol=1e-4)


def test_single_water_zero_energy():
    """Test that a single isolated water molecule has zero intermolecular energy."""
    force_field = _get_water_model_forcefield()

    water = openff.toolkit.Molecule.from_smiles("O")
    topology = openff.toolkit.Topology.from_molecules([water])

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)

    # Standard water geometry
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.2399, 0.9270, 0.0],
    ], dtype=torch.float32)

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Compute SMEE energy
    energy = compute_multipole_energy(
        system, es_potential, coords, None, polarization_type="direct"
    )

    # Single molecule should have zero intermolecular energy
    # (intramolecular terms are excluded via covalent maps)
    print(f"\nSingle water molecule energy: {energy.item():.6e} kcal/mol")

    assert torch.allclose(energy, torch.tensor(0.0, dtype=energy.dtype), atol=1e-6)


@pytest.mark.parametrize("polarization_type", ["direct", "mutual", "extrapolated"])
def test_water_dimer_multiple_conformers(polarization_type):
    """Test water dimer with multiple conformers (batched computation)."""
    force_field = _get_water_model_forcefield()

    water1 = openff.toolkit.Molecule.from_smiles("O")
    water2 = openff.toolkit.Molecule.from_smiles("O")
    topology = openff.toolkit.Topology.from_molecules([water1, water2])

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)

    # Create multiple conformers with different separations
    base_coords = _get_water_dimer_coords()
    conformers = []
    for z_offset in [2.5, 3.0, 3.5, 4.0]:
        coords = base_coords.clone()
        # Shift second water
        coords[3:, 2] = z_offset
        conformers.append(coords)

    coords_batch = torch.stack(conformers).float()

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Compute SMEE energies for batch
    energies = compute_multipole_energy(
        system, es_potential, coords_batch, None, polarization_type=polarization_type
    )

    # Compute OpenMM reference energies individually
    expected_energies = torch.tensor([
        _compute_openmm_energy(system, coords, None, es_potential, polarization_type=polarization_type)
        for coords in conformers
    ])

    print(f"\nWater dimer multiple conformers ({polarization_type}):")
    for i, (e, exp_e) in enumerate(zip(energies, expected_energies)):
        print(f"  Conformer {i}: SMEE={e.item():.6f}, OpenMM={exp_e.item():.6f}, diff={abs(e.item()-exp_e.item()):.2e}")

    assert torch.allclose(energies, expected_energies, atol=1e-4)


@pytest.mark.parametrize("polarization_type", ["direct", "mutual", "extrapolated"])
def test_water_dimer_axis_resolution(polarization_type):
    """Test that axis atoms are correctly resolved from SMIRKS indices to topology indices."""
    force_field = _get_water_model_forcefield()

    water1 = openff.toolkit.Molecule.from_smiles("O")
    water2 = openff.toolkit.Molecule.from_smiles("O")
    topology = openff.toolkit.Topology.from_molecules([water1, water2])

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]
    param_map = tensor_top.parameters[es_potential.type]
    assigned_params = param_map.assignment_matrix @ es_potential.parameters

    print(f"\nAxis resolution test ({polarization_type}):")
    print("Checking that axis atoms are actual topology indices (not SMIRKS indices):")

    # For 6 atoms (2 waters), axis atoms should be 0-5 (topology indices)
    # NOT 1-3 (SMIRKS indices)
    for i in range(6):
        axis_type = int(assigned_params[i, 13])
        atom_z = int(assigned_params[i, 14])
        atom_x = int(assigned_params[i, 15])
        print(f"  Atom {i}: axisType={axis_type}, atomZ={atom_z}, atomX={atom_x}")

        # Verify axis atoms are valid topology indices
        if atom_z >= 0:
            assert atom_z < 6, f"atomZ={atom_z} exceeds topology size (6 atoms)"
        if atom_x >= 0:
            assert atom_x < 6, f"atomX={atom_x} exceeds topology size (6 atoms)"

    # Verify we can compute energy without errors
    coords = _get_water_dimer_coords()
    energy = compute_multipole_energy(
        system, es_potential, coords.float(), None, polarization_type=polarization_type
    )
    assert torch.isfinite(energy)


@pytest.mark.parametrize("n_waters", [2, 4, 8])
def test_water_cluster_energy(n_waters):
    """Test water clusters of various sizes."""
    force_field = _get_water_model_forcefield()

    waters = [openff.toolkit.Molecule.from_smiles("O") for _ in range(n_waters)]
    topology = openff.toolkit.Topology.from_molecules(waters)

    interchange = openff.interchange.Interchange.from_smirnoff(force_field, topology)
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    system = smee.TensorSystem([tensor_top], [1], is_periodic=False)

    # Generate coordinates for water cluster
    import numpy as np
    np.random.seed(42)

    coords_list = []
    spacing = 3.0  # Å between water molecules

    for i in range(n_waters):
        # Place water molecules in a grid
        x = (i % 2) * spacing
        y = ((i // 2) % 2) * spacing
        z = (i // 4) * spacing

        # Standard water geometry
        coords_list.append([x, y, z])  # O
        coords_list.append([x + 0.9572, y, z])  # H
        coords_list.append([x - 0.2399, y + 0.9270, z])  # H

    coords = torch.tensor(coords_list, dtype=torch.float64)

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Compute SMEE energy
    energy = compute_multipole_energy(
        system, es_potential, coords.float(), None, polarization_type="mutual"
    )

    # Compute OpenMM reference energy
    expected_energy = _compute_openmm_energy(
        system, coords, None, es_potential, polarization_type="mutual"
    )

    print(f"\nWater cluster (n={n_waters}):")
    print(f"  SMEE Energy: {energy.item():.6f} kcal/mol")
    print(f"  OpenMM Energy: {expected_energy.item():.6f} kcal/mol")
    print(f"  Difference: {abs(energy.item() - expected_energy.item()):.2e} kcal/mol")

    assert torch.allclose(energy, expected_energy, atol=1e-3)


# =============================================================================
# Ammonia Tests - Testing ThreeFold axis type and different molecular geometry
# =============================================================================

def _get_ammonia_dimer_coords():
    """Return coordinates for an ammonia dimer in Angstroms.

    Ammonia geometry: N-H bond length ~1.012 Å, H-N-H angle ~106.7°
    Tetrahedral-like geometry with lone pair.
    """
    import math

    # Ammonia 1 at origin
    # N at origin, 3 H atoms in tetrahedral-like arrangement
    n1 = [0.0, 0.0, 0.0]
    # H atoms roughly 1.012 Å from N, ~106.7° H-N-H angle
    h1a = [0.9377, 0.0, -0.3816]
    h1b = [-0.4689, 0.8121, -0.3816]
    h1c = [-0.4689, -0.8121, -0.3816]

    # Ammonia 2 displaced along z-axis
    z_offset = 3.5
    n2 = [0.0, 0.0, z_offset]
    h2a = [0.9377, 0.0, z_offset - 0.3816]
    h2b = [-0.4689, 0.8121, z_offset - 0.3816]
    h2c = [-0.4689, -0.8121, z_offset - 0.3816]

    coords = torch.tensor([n1, h1a, h1b, h1c, n2, h2a, h2b, h2c], dtype=torch.float64)
    return coords


def _get_ammonia_trimer_coords():
    """Return coordinates for an ammonia trimer in Angstroms."""
    import math

    # Ammonia 1 at origin
    n1 = [0.0, 0.0, 0.0]
    h1a = [0.9377, 0.0, -0.3816]
    h1b = [-0.4689, 0.8121, -0.3816]
    h1c = [-0.4689, -0.8121, -0.3816]

    # Ammonia 2 displaced along x
    x_offset = 3.5
    n2 = [x_offset, 0.0, 0.0]
    h2a = [x_offset + 0.9377, 0.0, -0.3816]
    h2b = [x_offset - 0.4689, 0.8121, -0.3816]
    h2c = [x_offset - 0.4689, -0.8121, -0.3816]

    # Ammonia 3 displaced to form triangle
    angle = math.pi / 3
    r = 3.5
    n3 = [r * math.cos(angle), r * math.sin(angle), 0.0]
    h3a = [n3[0] + 0.9377, n3[1], -0.3816]
    h3b = [n3[0] - 0.4689, n3[1] + 0.8121, -0.3816]
    h3c = [n3[0] - 0.4689, n3[1] - 0.8121, -0.3816]

    coords = torch.tensor([
        n1, h1a, h1b, h1c,
        n2, h2a, h2b, h2c,
        n3, h3a, h3b, h3c
    ], dtype=torch.float64)
    return coords


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated"),
    ]
)
def test_ammonia_dimer_energy(test_data_dir, forcefield_name, polarization_type):
    """Test ammonia dimer energy matches OpenMM/TholeDipoleForce.

    Ammonia uses ThreeFold axis type for N and ZOnly for H atoms.
    """
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["N", "N"],
        [1, 1],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = _get_ammonia_dimer_coords()

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords.float(), None, polarization_type=polarization_type
    )

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, None, es_potential, polarization_type=polarization_type
    )

    print(f"\nAmmonia dimer ({forcefield_name}, {polarization_type}):")
    print(f"  SMEE Energy: {energy.item():.6f} kcal/mol")
    print(f"  OpenMM Energy: {expected_energy.item():.6f} kcal/mol")
    print(f"  Difference: {abs(energy.item() - expected_energy.item()):.2e} kcal/mol")

    assert torch.allclose(energy, expected_energy, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated"),
    ]
)
def test_ammonia_dimer_forces(test_data_dir, forcefield_name, polarization_type):
    """Test ammonia dimer forces via autograd match finite difference."""
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["N", "N"],
        [1, 1],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = _get_ammonia_dimer_coords().float()
    coords.requires_grad = True

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords, None, polarization_type=polarization_type
    )
    energy.backward()
    smee_forces = -coords.grad.detach()

    # Use h=1e-2 for optimal float32 finite difference precision
    h = 1e-2
    numerical_forces = torch.zeros_like(coords)

    for i in range(coords.shape[0]):
        for j in range(3):
            coords_plus = coords.detach().clone()
            coords_plus[i, j] += h
            e_plus = compute_multipole_energy(
                tensor_sys, es_potential, coords_plus, None, polarization_type=polarization_type
            )

            coords_minus = coords.detach().clone()
            coords_minus[i, j] -= h
            e_minus = compute_multipole_energy(
                tensor_sys, es_potential, coords_minus, None, polarization_type=polarization_type
            )

            numerical_forces[i, j] = -(e_plus - e_minus) / (2 * h)

    print(f"\nAmmonia dimer forces ({forcefield_name}, {polarization_type}):")
    print(f"  Max force difference (autograd vs numerical): {(smee_forces - numerical_forces).abs().max():.2e}")

    assert torch.allclose(smee_forces, numerical_forces, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-nonpolar-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated"),
    ]
)
def test_ammonia_trimer_energy(test_data_dir, forcefield_name, polarization_type):
    """Test ammonia trimer energy matches OpenMM/TholeDipoleForce."""
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["N"],
        [3],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    coords = _get_ammonia_trimer_coords()

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords.float(), None, polarization_type=polarization_type
    )

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, None, es_potential, polarization_type=polarization_type
    )

    print(f"\nAmmonia trimer ({forcefield_name}, {polarization_type}):")
    print(f"  SMEE Energy: {energy.item():.6f} kcal/mol")
    print(f"  OpenMM Energy: {expected_energy.item():.6f} kcal/mol")
    print(f"  Difference: {abs(energy.item() - expected_energy.item()):.2e} kcal/mol")

    assert torch.allclose(energy, expected_energy, atol=1e-3)


@pytest.mark.parametrize(
    "forcefield_name",
    ["PHAST-H2CNO-nonpolar-2.0.0.offxml", "PHAST-H2CNO-2.0.0.offxml"]
)
def test_single_ammonia_zero_energy(test_data_dir, forcefield_name):
    """Test that a single isolated ammonia molecule has zero intermolecular energy."""
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["N"],
        [1],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    # Standard ammonia geometry
    coords = torch.tensor([
        [0.0, 0.0, 0.0],        # N
        [0.9377, 0.0, -0.3816], # H
        [-0.4689, 0.8121, -0.3816], # H
        [-0.4689, -0.8121, -0.3816], # H
    ], dtype=torch.float32)

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords, None, polarization_type="direct"
    )

    print(f"\nSingle ammonia molecule energy ({forcefield_name}): {energy.item():.6e} kcal/mol")

    assert torch.allclose(energy, torch.tensor(0.0, dtype=energy.dtype), atol=1e-6)


@pytest.mark.parametrize(
    "forcefield_name,polarization_type",
    [
        ("PHAST-H2CNO-2.0.0.offxml", "direct"),
        ("PHAST-H2CNO-2.0.0.offxml", "mutual"),
        ("PHAST-H2CNO-2.0.0.offxml", "extrapolated"),
    ]
)
def test_ammonia_axis_types(test_data_dir, forcefield_name, polarization_type):
    """Test that ammonia axis types are correctly assigned."""
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["N"],
        [2],
        openff.toolkit.ForceField(
            str(test_data_dir / forcefield_name), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    # Get assigned parameters for the single topology (4 atoms)
    topology = tensor_sys.topologies[0]
    param_map = topology.parameters[es_potential.type]
    assigned_params = param_map.assignment_matrix @ es_potential.parameters

    n_atoms_per_mol = topology.n_particles
    print(f"\nAmmonia axis types ({forcefield_name}, {polarization_type}):")
    print(f"  Atoms per molecule: {n_atoms_per_mol}")

    # Check axis types for each atom in the topology template
    for i in range(n_atoms_per_mol):
        axis_type = int(assigned_params[i, 13])
        atom_z = int(assigned_params[i, 14])
        atom_x = int(assigned_params[i, 15])
        atom_y = int(assigned_params[i, 16])

        atom_type = "N" if i == 0 else "H"
        print(f"  Atom {i} ({atom_type}): axisType={axis_type}, atomZ={atom_z}, atomX={atom_x}, atomY={atom_y}")

        # Verify axis atoms are valid topology indices (or -1)
        if atom_z >= 0:
            assert atom_z < n_atoms_per_mol, f"atomZ={atom_z} exceeds topology size"
        if atom_x >= 0:
            assert atom_x < n_atoms_per_mol, f"atomX={atom_x} exceeds topology size"
        if atom_y >= 0:
            assert atom_y < n_atoms_per_mol, f"atomY={atom_y} exceeds topology size"

    # Verify we can compute energy without errors
    coords = _get_ammonia_dimer_coords()
    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords.float(), None, polarization_type=polarization_type
    )
    assert torch.isfinite(energy)


@pytest.mark.parametrize("n_ammonia", [2, 4, 6])
def test_ammonia_cluster_energy(test_data_dir, n_ammonia):
    """Test ammonia clusters of various sizes."""
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(
        ["N"],
        [n_ammonia],
        openff.toolkit.ForceField(
            str(test_data_dir / "PHAST-H2CNO-2.0.0.offxml"), load_plugins=True
        ),
    )
    tensor_sys.is_periodic = False

    # Generate coordinates for ammonia cluster
    coords_list = []
    spacing = 4.0  # Å between molecules

    for i in range(n_ammonia):
        # Place ammonia molecules in a grid
        x = (i % 2) * spacing
        y = ((i // 2) % 2) * spacing
        z = (i // 4) * spacing

        # Standard ammonia geometry
        coords_list.append([x, y, z])  # N
        coords_list.append([x + 0.9377, y, z - 0.3816])  # H
        coords_list.append([x - 0.4689, y + 0.8121, z - 0.3816])  # H
        coords_list.append([x - 0.4689, y - 0.8121, z - 0.3816])  # H

    coords = torch.tensor(coords_list, dtype=torch.float64)

    es_potential = tensor_ff.potentials_by_type["Electrostatics"]

    energy = compute_multipole_energy(
        tensor_sys, es_potential, coords.float(), None, polarization_type="mutual"
    )

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, None, es_potential, polarization_type="mutual"
    )

    print(f"\nAmmonia cluster (n={n_ammonia}):")
    print(f"  SMEE Energy: {energy.item():.6f} kcal/mol")
    print(f"  OpenMM Energy: {expected_energy.item():.6f} kcal/mol")
    print(f"  Difference: {abs(energy.item() - expected_energy.item()):.2e} kcal/mol")

    assert torch.allclose(energy, expected_energy, atol=1e-3)
