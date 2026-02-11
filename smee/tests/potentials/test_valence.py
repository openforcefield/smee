import pytest
import torch

import smee
from smee.potentials.valence import (
    _compute_cosine_torsion_energy,
    compute_cosine_improper_torsion_energy,
    compute_cosine_proper_torsion_energy,
    compute_harmonic_angle_energy,
    compute_harmonic_bond_energy,
    compute_linear_angle_energy,
    compute_linear_bond_energy,
)


def _mock_models(
    particle_idxs: torch.Tensor,
    parameters: torch.Tensor,
    parameter_cols: tuple[str, ...],
) -> tuple[smee.TensorPotential, smee.TensorSystem]:
    potential = smee.TensorPotential(
        type="mock",
        fn="mock-fn",
        parameters=parameters,
        parameter_keys=[None] * len(parameters),
        parameter_cols=parameter_cols,
        parameter_units=[None] * len(parameters),
        attributes=None,
        attribute_cols=None,
        attribute_units=None,
    )

    n_atoms = int(particle_idxs.max())

    parameter_map = smee.ValenceParameterMap(
        particle_idxs=particle_idxs,
        assignment_matrix=torch.eye(len(particle_idxs)),
    )
    topology = smee.TensorTopology(
        atomic_nums=torch.zeros(n_atoms, dtype=torch.long),
        formal_charges=torch.zeros(n_atoms, dtype=torch.long),
        bond_idxs=torch.zeros((0, 2), dtype=torch.long),
        bond_orders=torch.zeros(0, dtype=torch.long),
        parameters={potential.type: parameter_map},
        v_sites=None,
        constraints=None,
    )

    return potential, smee.TensorSystem([topology], [1], False)


@pytest.mark.parametrize(
    "conformer, expected_shape",
    [
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.Size([]),
        ),
        (torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1,)),
    ],
)
def test_compute_harmonic_bond_energy(conformer, expected_shape):
    atom_indices = torch.tensor([[0, 1], [0, 2]])
    parameters = torch.tensor([[2.0, 0.95], [0.5, 1.01]], requires_grad=True)

    potential, system = _mock_models(atom_indices, parameters, ("k", "length"))

    energy = compute_harmonic_bond_energy(system, potential, conformer)
    energy.backward()

    assert energy.shape == expected_shape

    assert torch.isclose(energy, torch.tensor(1.0 * 0.05**2 + 0.25 * 0.01**2))
    assert not torch.allclose(parameters.grad, torch.tensor(0.0))


@pytest.mark.parametrize(
    "conformer, expected_shape",
    [
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.Size([]),
        ),
        (torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1,)),
    ],
)
def test_compute_harmonic_angle_energy(conformer, expected_shape):
    atom_indices = torch.tensor([[0, 1, 2]])
    # 1.67 radians ≈ 95.7 degrees, compared to equilibrium value of 90 degrees (1.57 radians)
    parameters = torch.tensor([[200.0, 1.67]], requires_grad=True)

    potential, system = _mock_models(atom_indices, parameters, ("k", "angle"))

    energy = compute_harmonic_angle_energy(system, potential, conformer)
    energy.backward()

    assert energy.shape == expected_shape

    expected_energy = 0.5 * parameters[0, 0] * (torch.pi / 2.0 - parameters[0, 1]) ** 2
    expected_gradient = torch.tensor(
        [
            0.5 * (torch.pi / 2.0 - parameters[0, 1]) ** 2,
            parameters[0, 0] * (parameters[0, 1] - torch.pi / 2.0),
        ]
    )

    assert torch.isclose(energy, expected_energy)
    assert torch.allclose(parameters.grad, expected_gradient)


@pytest.mark.parametrize("expected_shape", [torch.Size([]), (1,)])
@pytest.mark.parametrize(
    "energy_function",
    [
        _compute_cosine_torsion_energy,
        compute_cosine_proper_torsion_energy,
        compute_cosine_improper_torsion_energy,
    ],
)
@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_compute_cosine_torsion_energy(expected_shape, energy_function, phi_sign):
    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )

    if expected_shape == (1,):
        conformer = torch.unsqueeze(conformer, 0)

    atom_indices = torch.tensor([[0, 1, 2, 3]])
    parameters = torch.tensor([[2.0, 2.0, 20.0, 1.5]], requires_grad=True)

    potential, system = _mock_models(
        atom_indices, parameters, ("k", "periodicity", "phase", "idivf")
    )

    energy = energy_function(system, potential, conformer)
    energy.backward()

    expected_energy = (
        parameters[0, 0]
        / parameters[0, 3]
        * (
            1.0
            + torch.cos(
                torch.tensor(
                    [
                        parameters[0, 1] * torch.tensor(phi_sign * torch.pi / 4.0)
                        - parameters[0, 2]
                    ]
                )
            )
        )
    )

    assert torch.isclose(energy, expected_energy)
    assert energy.shape == expected_shape


@pytest.mark.parametrize(
    "conformer, expected_shape",
    [
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.Size([]),
        ),
        (torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1,)),
    ],
)
def test_compute_linear_bond_energy(conformer, expected_shape):
    """Test linear bond energy gives same result as harmonic when parameters are derived
    from harmonic k and length with two basis functions.

    Linear form: 0.5 * (k1+k2) * (r - (k1*b1 + k2*b2)/(k1+k2))^2
    Should equal harmonic: 0.5 * k * (r - length)^2 when k=k1+k2 and length=(k1*b1+k2*b2)/(k1+k2)
    """
    atom_indices = torch.tensor([[0, 1], [0, 2]])

    # Original harmonic parameters from test_compute_harmonic_bond_energy
    k_harmonic = torch.tensor([2.0, 0.5])
    length_harmonic = torch.tensor([0.95, 1.01])

    # Choose basis functions with reasonable spacing around equilibrium
    # For bond 1: b1 = 0.90, b2 = 1.00 (span 0.1 Å around length=0.95)
    # For bond 2: b1 = 0.96, b2 = 1.06 (span 0.1 Å around length=1.01)
    b1 = torch.tensor([0.90, 0.96])
    b2 = torch.tensor([1.00, 1.06])

    # Solve for k1, k2 so that:
    # k1 + k2 = k_harmonic
    # (k1*b1 + k2*b2)/(k1+k2) = length_harmonic
    k1 = k_harmonic * (length_harmonic - b2) / (b1 - b2)
    k2 = k_harmonic - k1

    parameters = torch.stack([k1, k2, b1, b2], dim=1)
    parameters.requires_grad_(True)

    potential, system = _mock_models(atom_indices, parameters, ("k1", "k2", "b1", "b2"))

    energy = compute_linear_bond_energy(system, potential, conformer)
    energy.backward()

    assert energy.shape == expected_shape

    # Should give same energy as harmonic bond test
    expected_energy = torch.tensor(1.0 * 0.05**2 + 0.25 * 0.01**2)

    assert torch.isclose(energy, expected_energy, atol=1e-6)
    assert not torch.allclose(parameters.grad, torch.tensor(0.0))


@pytest.mark.parametrize(
    "conformer, expected_shape",
    [
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.Size([]),
        ),
        (torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1,)),
    ],
)
def test_compute_linear_angle_energy(conformer, expected_shape):
    """Test linear angle energy gives same result as harmonic when parameters are derived
    from harmonic k and angle with two basis functions.

    Linear form: 0.5 * (k1+k2) * (theta - (k1*a1 + k2*a2)/(k1+k2))^2
    Should equal harmonic: 0.5 * k * (theta - angle)^2 when k=k1+k2 and angle=(k1*a1+k2*a2)/(k1+k2)
    """
    atom_indices = torch.tensor([[0, 1, 2]])

    # Original harmonic parameters from test_compute_harmonic_angle_energy
    k_harmonic = torch.tensor([200.0])
    angle_harmonic = torch.tensor([1.67])  # radians

    # Choose basis functions with reasonable spacing around equilibrium (±0.2 radians ≈ ±11.5 degrees)
    angle1 = angle_harmonic - 0.2
    angle2 = angle_harmonic + 0.2

    # Solve for k1, k2 so that:
    # k1 + k2 = k_harmonic
    # (k1*angle1 + k2*angle2)/(k1+k2) = angle_harmonic
    k1 = k_harmonic * (angle_harmonic - angle2) / (angle1 - angle2)
    k2 = k_harmonic - k1

    parameters = torch.stack([k1, k2, angle1, angle2], dim=1)
    parameters.requires_grad_(True)

    potential, system = _mock_models(
        atom_indices, parameters, ("k1", "k2", "angle1", "angle2")
    )

    energy = compute_linear_angle_energy(system, potential, conformer)
    energy.backward()

    assert energy.shape == expected_shape

    # Should give same energy as harmonic angle test
    expected_energy = 0.5 * k_harmonic[0] * (torch.pi / 2.0 - angle_harmonic[0]) ** 2

    assert torch.isclose(energy, expected_energy, atol=1e-6)
    assert not torch.allclose(parameters.grad, torch.tensor(0.0))
