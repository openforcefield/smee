import pytest

pytest.importorskip("openmm")
pytest.importorskip("msgpack")

import msgpack
import numpy
import openff.interchange.models
import openff.units
import openmm.unit
import torch

import smee
import smee.tests.utils
from smee.mm._ops import (
    _compute_frame_observables,
    _compute_mass,
    _compute_observables,
    _pack_force_field,
    _unpack_force_field,
    compute_ensemble_averages,
    reweight_ensemble_averages,
)
from smee.mm._reporters import _encoder


def _mock_potential(type_, parameters, attributes) -> smee.TensorPotential:
    return smee.TensorPotential(
        type_,
        f"{type_}-fn",
        parameters,
        [
            openff.interchange.models.PotentialKey(id=f"[#{i}:1]")
            for i in range(len(parameters))
        ],
        tuple(f"param{i}" for i in range(parameters.shape[1])),
        tuple(openff.units.unit.angstrom for _ in range(parameters.shape[1])),
        attributes,
        tuple(f"attr-{i}" for i in range(len(attributes))),
        tuple(openff.units.unit.angstrom for _ in range(len(attributes))),
    )


def test_pack_unpack_force_field(mocker):
    parameters_a = torch.randn(4).reshape(2, 2)
    attributes_a = torch.randn(2)

    parameters_b = torch.randn(6).reshape(2, 3)
    attributes_b = torch.randn(3)

    v_site_keys = [mocker.Mock()]
    v_site_weights = [mocker.Mock()]
    v_site_parameters = torch.randn(12).reshape(4, 3)

    force_field = smee.TensorForceField(
        [
            _mock_potential("vdw", parameters_a, attributes_a),
            _mock_potential("bond", parameters_b, attributes_b),
        ],
        smee.TensorVSites(v_site_keys, v_site_weights, v_site_parameters),
    )

    tensors, param_lookup, attr_lookup, has_v_sites = _pack_force_field(force_field)

    expected_tensors = (
        parameters_a,
        parameters_b,
        attributes_a,
        attributes_b,
        v_site_parameters,
    )
    assert tensors == expected_tensors
    assert has_v_sites is True

    expected_param_lookup = {"vdw": 0, "bond": 1}
    assert param_lookup == expected_param_lookup
    expected_attr_lookup = {"vdw": 2, "bond": 3}
    assert attr_lookup == expected_attr_lookup

    updated_tensors = tuple(v + 1.0 for v in tensors)

    unpacked_force_field = _unpack_force_field(
        updated_tensors, param_lookup, attr_lookup, has_v_sites, force_field
    )

    assert len(unpacked_force_field.potentials) == 2

    expected_params = updated_tensors[:2]
    expected_attrs = updated_tensors[2:-1]
    expected_v_site = updated_tensors[-1]

    for i, (original, unpacked) in enumerate(
        zip(force_field.potentials, unpacked_force_field.potentials, strict=True)
    ):
        assert original.type == unpacked.type
        assert original.fn == unpacked.fn

        assert original.parameters.shape == unpacked.parameters.shape
        assert torch.allclose(original.parameters, tensors[i])
        assert torch.allclose(unpacked.parameters, expected_params[i])

        assert original.parameter_keys == unpacked.parameter_keys
        assert original.parameter_cols == unpacked.parameter_cols
        assert original.parameter_units == unpacked.parameter_units

        assert original.attributes.shape == unpacked.attributes.shape
        assert torch.allclose(original.attributes, tensors[i + 2])
        assert torch.allclose(unpacked.attributes, expected_attrs[i])

        assert original.attribute_cols == unpacked.attribute_cols
        assert original.attribute_units == unpacked.attribute_units

    assert unpacked_force_field.v_sites.keys == v_site_keys
    assert unpacked_force_field.v_sites.weights == v_site_weights
    assert force_field.v_sites.parameters.shape == v_site_parameters.shape
    assert torch.allclose(force_field.v_sites.parameters, tensors[-1])
    assert unpacked_force_field.v_sites.parameters.shape == v_site_parameters.shape
    assert torch.allclose(unpacked_force_field.v_sites.parameters, expected_v_site)


def test_compute_mass():
    water = smee.tests.utils.topology_from_smiles("O")
    methane = smee.tests.utils.topology_from_smiles("C")

    system = smee.TensorSystem([water, methane], [2, 3], True)

    mass_o = 15.99943
    mass_c = 12.01078
    mass_h = 1.007947

    expected_mass = (mass_o + 2.0 * mass_h) * 2 + (mass_c + 4.0 * mass_h) * 3
    mass = _compute_mass(system)

    assert mass == pytest.approx(expected_mass)


def test_compute_frame_observables_non_periodic(mocker):
    system = smee.TensorSystem(
        [smee.tests.utils.topology_from_smiles("[Ar]")], [1], False
    )

    beta = 2.0

    expected_potential = 1.2345

    values = _compute_frame_observables(
        system, mocker.MagicMock(), expected_potential, mocker.MagicMock(), beta, None
    )
    assert values == {
        "potential_energy": expected_potential,
        "potential_energy^2": expected_potential**2,
        "reduced_potential": beta * expected_potential,
    }


def test_compute_frame_observables():
    system = smee.TensorSystem(
        [smee.tests.utils.topology_from_smiles("[Ar]")], [1], True
    )

    box_length = 20.0
    expected_volume = box_length**3

    box_vectors = torch.eye(3) * box_length

    expected_potential = 1.2345
    expected_kinetic = 5.4321

    beta = 2.0 / openmm.unit.kilocalories_per_mole
    pressure = 1.0 * openmm.unit.bar

    mass_ar = 39.9481 * openmm.unit.daltons
    expected_density = (
        mass_ar / (expected_volume * openmm.unit.angstrom**3)
    ).value_in_unit(openmm.unit.gram / openmm.unit.item / openmm.unit.milliliter)

    expected_enthalpy = (
        expected_potential * openmm.unit.kilocalorie_per_mole
        + expected_kinetic * openmm.unit.kilocalorie_per_mole
        + pressure
        * (expected_volume * openmm.unit.angstrom**3)
        * openmm.unit.AVOGADRO_CONSTANT_NA
    ).value_in_unit(openmm.unit.kilocalorie_per_mole)

    expected_reduced_potential = beta * (
        expected_potential * openmm.unit.kilocalorie_per_mole
        + pressure
        * (expected_volume * openmm.unit.angstrom**3)
        * openmm.unit.AVOGADRO_CONSTANT_NA
    )

    values = _compute_frame_observables(
        system,
        box_vectors,
        expected_potential,
        expected_kinetic,
        beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1),
        (pressure * openmm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**3
        ),
    )
    assert values == {
        "potential_energy": torch.tensor(expected_potential),
        "potential_energy^2": torch.tensor(expected_potential**2),
        "volume": pytest.approx(torch.tensor(expected_volume)),
        "volume^2": pytest.approx(torch.tensor(expected_volume**2)),
        "density": pytest.approx(torch.tensor(expected_density)),
        "enthalpy": pytest.approx(torch.tensor(expected_enthalpy)),
        "enthalpy^2": pytest.approx(torch.tensor(expected_enthalpy**2)),
        "reduced_potential": pytest.approx(torch.tensor(expected_reduced_potential)),
        "enthalpy_volume": pytest.approx(
            torch.tensor(expected_enthalpy * expected_volume)
        ),
    }


def test_compute_observables(tmp_path, mock_argon_tensors, mock_argon_params):
    eps, sig = mock_argon_params

    eps = eps.m_as("kilocalorie / mole")
    sig = sig.m_as("angstrom")

    tensor_ff, tensor_top = mock_argon_tensors
    tensor_system = smee.TensorSystem([tensor_top], [2], False)

    tensor_ff.potentials_by_type["vdW"].parameters.requires_grad = True

    theta = (
        tensor_ff.potentials_by_type["vdW"].parameters,
        tensor_ff.potentials_by_type["vdW"].attributes,
    )

    coords = numpy.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    box_vectors = numpy.stack([numpy.eye(3) * 20.0] * 3)

    frames_path = tmp_path / ("frames.msgpack")

    with frames_path.open("wb") as file:
        for coord, box_vector in zip(coords, box_vectors, strict=True):
            frame = (
                torch.tensor(coord).float(),
                torch.tensor(box_vector).float(),
                0.0,
                0.0,
            )
            file.write(msgpack.dumps(frame, default=_encoder))

    distances = numpy.linalg.norm(coords[:, 1] - coords[:, 0], axis=-1)

    beta = 2.0

    expected_du_d_eps = 4.0 * ((sig / distances) ** 12 - (sig / distances) ** 6)
    expected_du_d_sig = (eps * (sig**5) * (48.0 * (sig**6) - 24.0 * distances**6)) / (
        distances**12
    )

    expected_potential = eps * expected_du_d_eps

    with frames_path.open("rb") as file:
        values, columns, reduced_potential, du_d_theta = _compute_observables(
            tensor_system, tensor_ff, file, theta, beta, None
        )

    assert columns == ["potential_energy", "potential_energy^2"]

    assert values.shape == (len(expected_potential), 2)
    assert numpy.allclose(
        values.numpy(),
        numpy.stack([expected_potential, expected_potential**2], axis=-1),
    )

    assert reduced_potential.shape == (len(expected_potential),)
    assert numpy.allclose(reduced_potential.numpy(), beta * expected_potential)

    assert len(du_d_theta) == 2
    assert du_d_theta[1] is None

    assert numpy.allclose(
        du_d_theta[0][0, 0, :].numpy(), expected_du_d_eps, atol=1.0e-4
    )
    assert numpy.allclose(
        du_d_theta[0][0, 1, :].numpy(), expected_du_d_sig, atol=1.0e-4
    )


def test_compute_ensemble_averages(mocker, tmp_path, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors
    tensor_system = smee.TensorSystem([tensor_top], [2], False)

    output_path = tmp_path / "traj.msgpack"
    output_path.write_bytes(b"")

    mock_outputs = torch.stack(
        [
            torch.tensor([1.0, 1.0, 2.0, 4.0, 3.0]),
            torch.tensor([5.0, 25.0, 6.0, 36.0, 20.0]),
        ]
    )
    mock_columns = [
        "potential_energy",
        "potential_energy^2",
        "volume",
        "volume^2",
        "density",
    ]
    mock_du_d_theta = (torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]), None)

    mock_compute_observables = mocker.patch(
        "smee.mm._ops._compute_observables",
        autospec=True,
        return_value=(mock_outputs, mock_columns, None, mock_du_d_theta),
    )

    temperature = 86.0 * openmm.unit.kelvin

    tensor_ff.potentials_by_type["vdW"].parameters.requires_grad = True

    ensemble_avgs, ensemble_stds = compute_ensemble_averages(
        tensor_system, tensor_ff, output_path, temperature, None
    )

    assert mock_compute_observables.call_count == 1

    assert ensemble_stds["potential_energy"].grad_fn is None
    assert torch.isclose(
        ensemble_stds["potential_energy"], torch.std(torch.tensor([1.0, 5.0]))
    )

    ensemble_avgs["potential_energy"].backward(retain_graph=True)
    energy_grad = tensor_ff.potentials_by_type["vdW"].parameters.grad
    tensor_ff.potentials_by_type["vdW"].parameters.grad = None

    mock_compute_observables.assert_called_once()

    assert ensemble_stds["volume"].grad_fn is None
    assert torch.isclose(ensemble_stds["volume"], torch.std(torch.tensor([2.0, 6.0])))

    ensemble_avgs["volume"].backward(retain_graph=True)
    volume_grad = tensor_ff.potentials_by_type["vdW"].parameters.grad
    tensor_ff.potentials_by_type["vdW"].parameters.grad = None

    mock_compute_observables.assert_called_once()

    assert ensemble_stds["density"].grad_fn is None
    assert torch.isclose(ensemble_stds["density"], torch.std(torch.tensor([3.0, 20.0])))

    ensemble_avgs["density"].backward(retain_graph=False)
    density_grad = tensor_ff.potentials_by_type["vdW"].parameters.grad
    tensor_ff.potentials_by_type["vdW"].parameters.grad = None

    mock_compute_observables.assert_called_once()

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

    energy, volume, density = mock_outputs[:, 0], mock_outputs[:, 2], mock_outputs[:, 4]
    du_d_eps = mock_du_d_theta[0][0, 0, :]

    expected_d_avg_energy_d_eps = du_d_eps.mean(-1) - beta * (
        (energy * du_d_eps).mean(-1) - energy.mean() * du_d_eps.mean(-1)
    )
    expected_d_avg_volume_d_eps = -beta * (
        (volume * du_d_eps).mean(-1) - volume.mean() * du_d_eps.mean(-1)
    )
    expected_d_avg_density_d_eps = -beta * (
        (density * du_d_eps).mean(-1) - density.mean() * du_d_eps.mean(-1)
    )

    assert torch.isclose(expected_d_avg_energy_d_eps.double(), energy_grad[0, 0])
    assert torch.isclose(expected_d_avg_volume_d_eps.double(), volume_grad[0, 0])
    assert torch.isclose(expected_d_avg_density_d_eps.double(), density_grad[0, 0])


def test_reweight_ensemble_averages(mocker, tmp_path, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors
    tensor_system = smee.TensorSystem([tensor_top], [2], False)

    temperature = 86.0 * openmm.unit.kelvin
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)

    mock_outputs = torch.stack(
        [
            torch.tensor([1.0, 1.0, 2.0, 4.0, 3.0]),
            torch.tensor([5.0, 25.0, 6.0, 36.0, 20.0]),
        ]
    )
    mock_reduced = (
        beta.value_in_unit(openmm.unit.kilocalories_per_mole**-1) * mock_outputs[:, 0]
    )

    mock_columns = [
        "potential_energy",
        "potential_energy^2",
        "volume",
        "volume^2",
        "density",
    ]
    mock_du_d_theta = (torch.tensor([[[-9.0, 10.0], [11.0, -12.0]]]), None)

    mocker.patch(
        "smee.mm._ops._compute_observables",
        autospec=True,
        return_value=(mock_outputs, mock_columns, mock_reduced, mock_du_d_theta),
    )

    output_path = tmp_path / "traj.msgpack"
    output_path.write_bytes(b"")

    with output_path.open("wb") as file:
        for reduced_potential in mock_reduced:
            frame = (None, None, reduced_potential, 0.0)
            file.write(msgpack.dumps(frame, default=_encoder))

    vdw_parameters = tensor_ff.potentials_by_type["vdW"].parameters
    vdw_parameters.requires_grad = True

    ensemble_averages, _ = compute_ensemble_averages(
        tensor_system, tensor_ff, output_path, temperature, None
    )
    reweight_averages = reweight_ensemble_averages(
        tensor_system, tensor_ff, output_path, temperature, None, 0
    )

    for observable in ensemble_averages:
        (ensemble_grad,) = torch.autograd.grad(
            ensemble_averages[observable], vdw_parameters, retain_graph=True
        )
        (reweight_grad,) = torch.autograd.grad(
            reweight_averages[observable], vdw_parameters, retain_graph=True
        )

        assert reweight_grad.shape == ensemble_grad.shape
        assert torch.allclose(reweight_grad, ensemble_grad)


def test_compute_dg_solv(mocker, tmp_path, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors

    params = tensor_ff.potentials_by_type["vdW"].parameters
    params.requires_grad = True

    mocker.patch(
        "smee.mm._fe.compute_dg_and_grads",
        side_effect=[
            (torch.tensor(1.0).double(), (torch.tensor([[2.0, 3.0]]).double(),)),
            (torch.tensor(4.0).double(), (torch.tensor([[5.0, 6.0]]).double(),)),
        ],
    )

    dg = smee.mm.compute_dg_solv(tensor_top, None, tensor_top, tensor_ff, tmp_path)
    dg_dtheta = torch.autograd.grad(dg, params)[0]

    assert torch.isclose(dg, torch.tensor(-3.0).double())
    assert torch.allclose(dg_dtheta, torch.tensor([[3.0, 3.0]]).double())


def test_reweight_dg_solv(mocker, tmp_path, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors

    params = tensor_ff.potentials_by_type["vdW"].parameters
    params.requires_grad = True

    mocker.patch(
        "smee.mm._fe.reweight_dg_and_grads",
        side_effect=[
            (torch.tensor(1.0).double(), (torch.tensor([[2.0, 3.0]]).double(),), 4.0),
            (torch.tensor(5.0).double(), (torch.tensor([[6.0, 7.0]]).double(),), 8.0),
        ],
    )

    dg_0 = torch.tensor(-3.0).double()

    dg, n_eff = smee.mm.reweight_dg_solv(
        tensor_top, None, tensor_top, tensor_ff, tmp_path, dg_0, 3
    )
    dg_dtheta = torch.autograd.grad(dg, params)[0]

    assert torch.isclose(dg, torch.tensor(1.0).double())
    assert torch.allclose(dg_dtheta, torch.tensor([[4.0, 4.0]]).double())

    assert n_eff == 4.0


def test_reweight_dg_solv_error(mocker, tmp_path, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors

    params = tensor_ff.potentials_by_type["vdW"].parameters
    params.requires_grad = True

    mocker.patch(
        "smee.mm._fe.reweight_dg_and_grads",
        side_effect=[
            (torch.tensor(1.0).double(), (torch.tensor([[2.0, 3.0]]).double(),), 4.0),
            (torch.tensor(5.0).double(), (torch.tensor([[6.0, 7.0]]).double(),), 8.0),
        ],
    )

    dg_0 = torch.tensor(-3.0).double()

    with pytest.raises(smee.mm.NotEnoughSamplesError):
        smee.mm.reweight_dg_solv(
            tensor_top, None, tensor_top, tensor_ff, tmp_path, dg_0, 100
        )
