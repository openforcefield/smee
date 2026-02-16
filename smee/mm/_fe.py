"""Compute ddG from the output of ``femto`` / ``absolv``"""

import logging
import pathlib
import pickle
import typing

import numpy
import openff.toolkit
import openmm.unit
import torch
import yaml

import smee
import smee.converters
import smee.mm._utils
import smee.utils

if typing.TYPE_CHECKING:
    import absolv.config

_LOGGER = logging.getLogger(__name__)

_NM_TO_ANGSTROM = 10.0


class _Sample(typing.NamedTuple):
    system: smee.TensorSystem

    beta: float
    pressure: float | None

    u_kn: torch.Tensor
    n_k: torch.Tensor

    xyz: torch.Tensor
    box: torch.Tensor | None


def _extract_pure_solvent(
    solute: smee.TensorTopology,
    solvent: smee.TensorTopology | None,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = force_field.potentials[0].parameters.device
    dtype = force_field.potentials[0].parameters.dtype

    system, _, _, _, _, xyz, box = _load_samples(
        solute, solvent, output_dir, device, dtype, coord_state_idx=-1
    )

    if len(system.topologies) != 2 or system.n_copies[0] != 1:
        raise NotImplementedError("only single solute systems are supported.")

    xyz = xyz[:, solute.n_particles :, :]

    system = smee.TensorSystem([solvent], [system.n_copies[1]], is_periodic=True)
    energy = _compute_energy(system, force_field, xyz, box)

    return xyz, box, energy


def _to_solvent_dict(
    solvent: smee.TensorTopology, n_solvent: int
) -> dict[str, int] | None:
    if solvent is None:
        return None

    smiles = openff.toolkit.Molecule.from_rdkit(
        smee.mm._utils.topology_to_rdkit(solvent),
        allow_undefined_stereo=True,
    ).to_smiles(mapped=True)

    return {smiles: n_solvent}


def generate_dg_solv_data(
    solute: smee.TensorTopology,
    solvent_a: smee.TensorTopology | None,
    solvent_b: smee.TensorTopology | None,
    force_field: smee.TensorForceField,
    temperature: openmm.unit.Quantity = 298.15 * openmm.unit.kelvin,
    pressure: openmm.unit.Quantity = 1.0 * openmm.unit.atmosphere,
    solvent_a_protocol: typing.Optional["absolv.config.EquilibriumProtocol"] = None,
    solvent_b_protocol: typing.Optional["absolv.config.EquilibriumProtocol"] = None,
    n_solvent_a: int = 216,
    n_solvent_b: int = 216,
    output_dir: pathlib.Path | None = None,
):
    """Run a solvation free energy calculation using ``absolv``, and saves the output
    such that a differentiable free energy can be computed.

    The free energy will correspond to the free energy of transferring a solute from
    solvent A to solvent B.

    Args:
        solute: The solute topology.
        solvent_a: The topology of solvent A, or ``None`` if solvent A is vacuum.
        solvent_b: The topology of solvent B, or ``None`` if solvent B is vacuum.
        force_field: The force field to parameterize the system with.
        temperature: The temperature to simulate at.
        pressure: The pressure to simulate at.
        solvent_a_protocol: The protocol to use to decouple the solute in solvent A.
        solvent_b_protocol: The protocol to use to decouple the solute in solvent B.
        n_solvent_a: The number of solvent A molecules to use.
        n_solvent_b: The number of solvent B molecules to use.
        output_dir: The directory to write the output FEP data to.
    """
    import absolv.config
    import absolv.runner
    import femto.md.config
    import femto.md.system
    import parmed.openmm

    output_dir = pathlib.Path.cwd() if output_dir is None else output_dir

    vacuum_protocol = absolv.config.EquilibriumProtocol(
        production_protocol=absolv.config.HREMDProtocol(
            n_steps_per_cycle=500,
            n_cycles=2000,
            integrator=femto.md.config.LangevinIntegrator(
                timestep=1.0 * openmm.unit.femtosecond
            ),
            trajectory_interval=1,
        ),
        lambda_sterics=absolv.config.DEFAULT_LAMBDA_STERICS_VACUUM,
        lambda_electrostatics=absolv.config.DEFAULT_LAMBDA_ELECTROSTATICS_VACUUM,
    )
    solution_protocol = absolv.config.EquilibriumProtocol(
        production_protocol=absolv.config.HREMDProtocol(
            n_steps_per_cycle=500,
            n_cycles=1000,
            integrator=femto.md.config.LangevinIntegrator(
                timestep=4.0 * openmm.unit.femtosecond
            ),
            trajectory_interval=1,
            trajectory_enforce_pbc=True,
        ),
        lambda_sterics=absolv.config.DEFAULT_LAMBDA_STERICS_SOLVENT,
        lambda_electrostatics=absolv.config.DEFAULT_LAMBDA_ELECTROSTATICS_SOLVENT,
    )

    config = absolv.config.Config(
        temperature=temperature,
        pressure=pressure,
        alchemical_protocol_a=solvent_a_protocol
        if solvent_a_protocol is not None
        else (vacuum_protocol if solvent_a is None else solution_protocol),
        alchemical_protocol_b=solvent_b_protocol
        if solvent_b_protocol is not None
        else (vacuum_protocol if solvent_b is None else solution_protocol),
    )

    solute_mol = openff.toolkit.Molecule.from_rdkit(
        smee.mm._utils.topology_to_rdkit(solute),
        allow_undefined_stereo=True,
    )

    system_config = absolv.config.System(
        solutes={solute_mol.to_smiles(mapped=True): 1},
        solvent_a=_to_solvent_dict(solvent_a, n_solvent_a),
        solvent_b=_to_solvent_dict(solvent_b, n_solvent_b),
    )

    topologies = {
        "solvent-a": smee.TensorSystem([solute], [1], is_periodic=False)
        if solvent_a is None
        else smee.TensorSystem([solute, solvent_a], [1, n_solvent_a], is_periodic=True),
        "solvent-b": smee.TensorSystem([solute], [1], is_periodic=False)
        if solvent_b is None
        else smee.TensorSystem([solute, solvent_b], [1, n_solvent_b], is_periodic=True),
    }
    pressures = {
        "solvent-a": None
        if solvent_a is None
        else pressure.value_in_unit(openmm.unit.atmosphere),
        "solvent-b": None
        if solvent_b is None
        else pressure.value_in_unit(openmm.unit.atmosphere),
    }
    n_solvent = {
        "solvent-a": None if solvent_a is None else n_solvent_a,
        "solvent-b": None if solvent_b is None else n_solvent_b,
    }

    for phase in topologies:
        state = {
            "temperature": temperature.value_in_unit(openmm.unit.kelvin),
            "pressure": pressures[phase],
            "n_solvent": n_solvent[phase],
        }

        (output_dir / phase).mkdir(exist_ok=True, parents=True)
        (output_dir / phase / "state.yaml").write_text(yaml.safe_dump(state))

    def _parameterize(
        top, coords, phase: typing.Literal["solvent-a", "solvent-b"]
    ) -> openmm.System:
        return smee.converters.convert_to_openmm_system(force_field, topologies[phase])

    prepared_system_a, prepared_system_b = absolv.runner.setup(
        system_config, config, _parameterize
    )

    femto.md.system.apply_hmr(
        prepared_system_a.system,
        parmed.openmm.load_topology(prepared_system_a.topology),
    )
    femto.md.system.apply_hmr(
        prepared_system_b.system,
        parmed.openmm.load_topology(prepared_system_b.topology),
    )

    result = absolv.runner.run_eq(
        config, prepared_system_a, prepared_system_b, "CUDA", output_dir, parallel=True
    )

    if solvent_a is not None:
        solvent_a_output = _extract_pure_solvent(
            solute, solvent_a, force_field, output_dir / "solvent-a"
        )
        (output_dir / "solvent-a" / "pure.pkl").write_bytes(
            pickle.dumps(solvent_a_output)
        )
    if solvent_b is not None:
        solvent_b_output = _extract_pure_solvent(
            solute, solvent_b, force_field, output_dir / "solvent-b"
        )
        (output_dir / "solvent-b" / "pure.pkl").write_bytes(
            pickle.dumps(solvent_b_output)
        )

    return result


def _uncorrelated_frames(length: int, g: float) -> list[int]:
    """Return the indices of frames that are un-correlated.

    Args:
        length: The total number of correlated frames.
        g: The statistical inefficiency of the data.

    Returns:
        The indices of un-correlated frames.
    """
    indices = []
    n = 0

    while int(round(n * g)) < length:
        t = int(round(n * g))
        if n == 0 or t != indices[n - 1]:
            indices.append(t)
        n += 1

    return indices


def _load_trajectory(
    trajectory_dir: pathlib.Path,
    system: smee.TensorSystem,
    replica_to_state_idx: numpy.ndarray,
    state_idx: int = 0,
) -> tuple[numpy.ndarray, numpy.ndarray | None]:
    import mdtraj

    n_states = len(list(trajectory_dir.glob("r*.dcd")))

    topology_omm = smee.converters.convert_to_openmm_topology(system)
    topology_md = mdtraj.Topology.from_openmm(topology_omm)

    trajectories = [
        mdtraj.load(str(trajectory_dir / f"r{i}.dcd"), top=topology_md)
        for i in range(n_states)
    ]
    state_idxs = (replica_to_state_idx.reshape(-1, n_states).T == state_idx).argmax(
        axis=0
    )

    xyz = numpy.stack(
        [
            trajectories[traj_idx].xyz[frame_idx] * _NM_TO_ANGSTROM
            for frame_idx, traj_idx in enumerate(state_idxs)
        ]
    )
    assert xyz.shape[1] == system.n_particles, "unexpected number of particles."

    if trajectories[0].unitcell_vectors is None:
        return xyz, None

    box = numpy.stack(
        [
            trajectories[traj_idx].unitcell_vectors[frame_idx] * _NM_TO_ANGSTROM
            for frame_idx, traj_idx in enumerate(state_idxs)
        ]
    )

    return xyz, box


def _load_samples(
    solute: smee.TensorTopology,
    solvent: smee.TensorTopology | None,
    output_dir: pathlib.Path,
    device: str | torch.device,
    dtype: torch.dtype,
    coord_state_idx: int = 0,
) -> _Sample:
    import pyarrow
    import pymbar.timeseries

    state = yaml.safe_load((output_dir / "state.yaml").read_text())
    assert (state["n_solvent"] is None and solvent is None) or (
        state["n_solvent"] is not None and solvent is not None
    ), "solvent must be provided if and only if n_solvent is provided."

    system = smee.TensorSystem(
        [solute, solvent] if solvent is not None else [solute],
        [1, state["n_solvent"]] if solvent is not None else [1],
        is_periodic=solvent is not None,
    )

    temperature = state["temperature"] * openmm.unit.kelvin

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

    pressure = None

    if state["pressure"] is not None:
        pressure = state["pressure"] * openmm.unit.atmosphere
        pressure = (pressure * openmm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**3
        )

    with pyarrow.OSFile(str(output_dir / "samples.arrow"), "rb") as file:
        with pyarrow.RecordBatchStreamReader(file) as reader:
            output_table = reader.read_all()

    replica_to_state_idx = numpy.hstack(
        [numpy.array(x) for x in output_table["replica_to_state_idx"].to_pylist()]
    )

    # group the data along axis 1 so that data sampled in the same state is grouped.
    # this will let us more easily de-correlate the data.
    u_kn = numpy.hstack([numpy.array(x) for x in output_table["u_kn"].to_pylist()])
    u_kn_per_k = [u_kn[:, replica_to_state_idx == i] for i in range(len(u_kn))]

    n_uncorrelated = u_kn.shape[1] // u_kn.shape[0]

    g = pymbar.timeseries.statistical_inefficiency_multiple(
        [
            u_kn_per_k[i][i, i * n_uncorrelated : (i + 1) * n_uncorrelated]
            for i in range(len(u_kn))
        ]
    )
    uncorrelated_frames = _uncorrelated_frames(n_uncorrelated, g)

    for state_idx, state_u_kn in enumerate(u_kn_per_k):
        u_kn_per_k[state_idx] = state_u_kn[:, uncorrelated_frames]

    u_kn = numpy.hstack(u_kn_per_k)
    n_k = numpy.array([len(uncorrelated_frames)] * u_kn.shape[0])

    n_expected_frames = int(u_kn.shape[1] // len(u_kn))

    if coord_state_idx < 0:
        coord_state_idx = len(u_kn) + coord_state_idx

    xyz_i, box_i = _load_trajectory(
        output_dir / "trajectories", system, replica_to_state_idx, coord_state_idx
    )
    xyz_i = xyz_i[uncorrelated_frames]
    box_i = box_i[uncorrelated_frames] if box_i is not None else None
    assert len(xyz_i) == n_expected_frames
    assert box_i is None or len(box_i) == n_expected_frames

    return _Sample(
        system.to(device),
        beta,
        pressure,
        torch.tensor(u_kn, device=device, dtype=dtype),
        torch.tensor(n_k, device=device),
        torch.tensor(xyz_i, device=device, dtype=dtype),
        torch.tensor(box_i, device=device, dtype=dtype) if box_i is not None else None,
    )


def _compute_energy(
    system: smee.TensorSystem,
    ff: smee.TensorForceField,
    xyz_0: torch.Tensor,
    box_0: torch.Tensor,
) -> torch.Tensor:
    if system.is_periodic:
        energy_per_frame = [
            smee.compute_energy(system, ff, c, b)
            for c, b in zip(xyz_0, box_0, strict=True)
        ]
        energy = torch.concat(energy_per_frame)
    else:
        energy = smee.compute_energy(system, ff, xyz_0, box_0)

    return energy


def _compute_grads_solvent(
    solvent: smee.TensorTopology,
    force_field: smee.TensorForceField,
    theta: tuple[torch.Tensor, ...],
    output_dir: pathlib.Path,
) -> tuple[torch.Tensor, ...]:
    device = force_field.potentials[0].parameters.device

    xyz, box, *_ = pickle.loads((output_dir / "pure.pkl").read_bytes())

    n_solvent = int(xyz.shape[1] // solvent.n_particles)
    assert n_solvent * solvent.n_particles == xyz.shape[1]

    system = smee.TensorSystem([solvent], [n_solvent], is_periodic=True).to(device)

    with torch.enable_grad():
        energy = _compute_energy(system, force_field, xyz, box)
        grads = torch.autograd.grad(energy.mean(), theta)

    return grads


def compute_dg_and_grads(
    solute: smee.TensorTopology,
    solvent: smee.TensorTopology | None,
    force_field: smee.TensorForceField,
    theta: tuple[torch.Tensor, ...],
    output_dir: pathlib.Path,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    Notes:
        This function assumes that the energy computed using ``force_field`` is the
        same as the energy computed by the simulation. This function currently makes
        no attempt to validate this assumption.
    """
    import pymbar

    device = force_field.potentials[0].parameters.device
    dtype = force_field.potentials[0].parameters.dtype

    system, beta, _, u_kn, n_k, xyz_0, box_0 = _load_samples(
        solute, solvent, output_dir, device, dtype
    )
    assert (box_0 is not None) == system.is_periodic

    mbar = pymbar.MBAR(
        u_kn.detach().cpu().numpy(),
        n_k.detach().cpu().numpy(),
        solver_protocol="robust",
    )

    f_i = mbar.compute_free_energy_differences()["Delta_f"][0, :]

    dg = (f_i[-1] - f_i[0]) / beta
    dg = smee.utils.tensor_like(dg, force_field.potentials[0].parameters)

    if len(theta) == 0:
        return dg, ()

    with torch.enable_grad():
        energy = _compute_energy(system, force_field, xyz_0, box_0)
        grads = torch.autograd.grad(energy.mean(), theta)

        if (output_dir / "pure.pkl").exists():
            assert solvent is not None, "expected solvent to be provided."

            grads_solvent = _compute_grads_solvent(
                solvent, force_field, theta, output_dir
            )
            grads = tuple(g - g_s for g, g_s in zip(grads, grads_solvent, strict=True))

    return dg, grads


def _reweight_dg_and_grads(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    xyz: torch.Tensor,
    box: torch.Tensor | None,
    u_0: torch.Tensor,
    n_0: torch.Tensor,
    beta: float,
    pressure: float | None,
    theta: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], float]:
    import pymbar

    with torch.enable_grad():
        energy_new = _compute_energy(system, force_field, xyz, box)

        u_new = energy_new.detach().clone() * beta

        if pressure is not None:
            u_new += pressure * torch.det(box) * beta

        u_kn = numpy.stack([u_0.detach().cpu().numpy(), u_new.detach().cpu().numpy()])
        n_k = numpy.array([n_0.detach().cpu().numpy().item(), 0])

        mbar = pymbar.MBAR(u_kn, n_k, solver_protocol="robust")

        n_eff = mbar.compute_effective_sample_number().min().item()

        f_i = mbar.compute_free_energy_differences()["Delta_f"][0, :]

        dg = (f_i[-1] - f_i[0]) / beta
        dg = smee.utils.tensor_like(dg, force_field.potentials[0].parameters)

        weights = smee.utils.tensor_like(mbar.W_nk[:, 1], energy_new)
        grads = ()

        if len(theta) > 0:
            grads = torch.autograd.grad((energy_new * weights).sum(), theta)

    return dg, grads, n_eff


def reweight_dg_and_grads(
    solute: smee.TensorTopology,
    solvent: smee.TensorTopology | None,
    force_field: smee.TensorForceField,
    theta: tuple[torch.Tensor, ...],
    output_dir: pathlib.Path,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], float]:
    device = force_field.potentials[0].parameters.device
    dtype = force_field.potentials[0].parameters.dtype

    system, beta, pressure, u_kn, n_k, xyz_0, box_0 = _load_samples(
        solute, solvent, output_dir, device, dtype
    )
    assert (box_0 is not None) == system.is_periodic
    assert (box_0 is not None) == (pressure is not None)

    u_0 = u_kn[0, : n_k[0]]
    n_0 = n_k[0]

    dg, grads, n_eff = _reweight_dg_and_grads(
        system, force_field, xyz_0, box_0, u_0, n_0, beta, pressure, theta
    )

    if not (output_dir / "pure.pkl").exists():
        return dg, grads, n_eff

    xyz_solv, box_solv, energy_solv = pickle.loads(
        (output_dir / "pure.pkl").read_bytes()
    )

    n_solvent = int(xyz_solv.shape[1] // solvent.n_particles)
    assert n_solvent * solvent.n_particles == xyz_solv.shape[1]

    system_solv = smee.TensorSystem([solvent], [n_solvent], is_periodic=True).to(device)

    u_0_solv = energy_solv.detach().clone() * beta

    if pressure is not None:
        u_0_solv += pressure * torch.det(box_solv) * beta

    n_0_solv = smee.utils.tensor_like([len(u_0_solv)], u_0_solv)

    dg_solv, grads_solv, n_eff_solv = _reweight_dg_and_grads(
        system_solv,
        force_field,
        xyz_solv,
        box_solv,
        u_0_solv,
        n_0_solv,
        beta,
        pressure,
        theta,
    )

    dg -= dg_solv
    grads = tuple(g - g_s for g, g_s in zip(grads, grads_solv, strict=True))

    return dg, grads, min(n_eff, n_eff_solv)
