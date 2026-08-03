"""Convert espaloma Graph + GNN parameters to smee TensorForceField/TensorTopology.

The force field should be rebuilt each forward pass from fresh GNN predictions,
keeping all parameter tensors in the autograd graph.

Units
-----
Espaloma Graph stores nm, kJ/mol internally.
smee expects Å, kcal/mol.
"""

from __future__ import annotations

import openff.interchange.models
import torch
from openff.units import unit

import smee
import smee.utils


def _identity_sparse(
    n: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.sparse.Tensor:
    """Return a sparse (n, n) identity matrix."""
    idx = torch.arange(n, device=device)
    indices = torch.stack([idx, idx])
    values = torch.ones(n, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n))


def _synthetic_keys(
    prefix: str, n: int
) -> list[openff.interchange.models.PotentialKey]:
    """Make PotentialKeys to slot in"""
    return [
        openff.interchange.models.PotentialKey(id=f"{prefix}-{i}") for i in range(n)
    ]


def build_smee_topology(
    graph,
    max_torsion_terms: int = 4,
    include_vdw: bool = True,
    include_electrostatics: bool = True,
    dtype: torch.dtype = torch.float64,
) -> smee.TensorTopology:
    """Build a smee TensorTopology from an espaloma Graph.

    This constructs the molecule-level topology metadata (term indices,
    exclusions, assignment matrices) needed by smee's energy routines.
    The returned topology is independent of parameter values and can be
    cached per molecule.


    Note
    ----
    Each GNN-predicted term gets its own row in the parameter table via
    identity assignment matrices (``I @ params = params``), unlike
    SMIRKS-typed force fields where chemically equivalent terms share
    rows. This SHOULD be ok, I think, because equivalence comes from
    upstream: identical embeddings --> identical parameters, and gradients
    accumulate on the shared ``nn.Parameter`` weights regardless.


    Parameters
    ----------
    graph
        An espaloma Graph with ``atomic_numbers``, ``bond_idxs``,
        ``angle_idxs``, ``torsion_idxs``, ``improper_idxs``, and
        ``mol`` (an OpenFF ``Molecule``).
    max_torsion_terms
        Number of Fourier terms per torsion (must match GNN readout width).
    include_vdw
        Whether to include the vdW nonbonded parameter map.
    include_electrostatics
        Whether to include the electrostatics nonbonded parameter map.
    dtype
        Floating-point dtype for assignment matrices. Must match the
        parameter tensors that will be multiplied against them (e.g.
        ``torch.float32`` for a float32 GNN).

    Returns
    -------
    smee.TensorTopology
        Topology ready for use with ``build_smee_force_field`` output.
    """
    mol = graph.mol
    topology = mol.to_topology()

    n_atoms = graph.atomic_numbers.shape[0]
    n_bonds = graph.bond_idxs.shape[0]
    n_angles = graph.angle_idxs.shape[0]
    n_propers = graph.torsion_idxs.shape[0]

    improper_idxs = getattr(graph, "improper_idxs", None)
    n_impropers = (
        0
        if improper_idxs is None or improper_idxs.numel() == 0
        else improper_idxs.shape[0]
    )

    formal_charges = torch.tensor(
        [atom.formal_charge.m_as(unit.e) for atom in topology.atoms],
        dtype=torch.long,
    )
    bond_orders = torch.tensor(
        [bond.bond_order for bond in topology.bonds], dtype=torch.long
    )

    # --- Valence parameter maps (identity assignment) ---
    parameters: dict[str, smee.ParameterMap] = {}

    if n_bonds > 0:
        parameters["Bonds"] = smee.ValenceParameterMap(
            particle_idxs=graph.bond_idxs.long(),
            assignment_matrix=_identity_sparse(n_bonds, dtype),
        )

    if n_angles > 0:
        parameters["Angles"] = smee.ValenceParameterMap(
            particle_idxs=graph.angle_idxs.long(),
            assignment_matrix=_identity_sparse(n_angles, dtype),
        )

    # smee stores one row per Fourier term; espaloma packs n_terms k values
    # per torsion. Expand to match (same as Interchange's mult-indexed key_map).
    # note torch.repeat_interleave is equivalent to numpy.repeat!
    # torch.repeat is numpy.tile
    # I'm assuming we're only training ks
    if n_propers > 0:
        n_proper_rows = n_propers * max_torsion_terms
        expanded_proper_idxs = graph.torsion_idxs.long().repeat_interleave(
            max_torsion_terms, dim=0
        )
        parameters["ProperTorsions"] = smee.ValenceParameterMap(
            particle_idxs=expanded_proper_idxs,
            assignment_matrix=_identity_sparse(n_proper_rows, dtype),
        )

    if n_impropers > 0:
        n_improper_rows = n_impropers * max_torsion_terms
        expanded_improper_idxs = improper_idxs.long().repeat_interleave(
            max_torsion_terms, dim=0
        )
        parameters["ImproperTorsions"] = smee.ValenceParameterMap(
            particle_idxs=expanded_improper_idxs,
            assignment_matrix=_identity_sparse(n_improper_rows, dtype),
        )

    # --- Nonbonded parameter maps ---
    if include_vdw or include_electrostatics:
        exclusion_dict = smee.utils.find_exclusions(topology)
        attribute_cols = (
            "scale_12",
            "scale_13",
            "scale_14",
            "scale_15",
            "cutoff",
            "switch_width",
        )
        attribute_to_idx = {col: i for i, col in enumerate(attribute_cols)}

        if exclusion_dict:
            excl_pairs = list(exclusion_dict.keys())
            excl_scales = list(exclusion_dict.values())
            exclusions = torch.tensor(excl_pairs, dtype=torch.int64)
            exclusion_scale_idxs = torch.tensor(
                [[attribute_to_idx[s]] for s in excl_scales], dtype=torch.int64
            )
        else:
            exclusions = torch.zeros((0, 2), dtype=torch.int64)
            exclusion_scale_idxs = torch.zeros((0, 1), dtype=torch.int64)

        if include_vdw:
            parameters["vdW"] = smee.NonbondedParameterMap(
                assignment_matrix=_identity_sparse(n_atoms, dtype),
                exclusions=exclusions,
                exclusion_scale_idxs=exclusion_scale_idxs,
            )

        if include_electrostatics:
            parameters["Electrostatics"] = smee.NonbondedParameterMap(
                assignment_matrix=_identity_sparse(n_atoms, dtype),
                exclusions=exclusions.clone() if include_vdw else exclusions,
                exclusion_scale_idxs=(
                    exclusion_scale_idxs.clone()
                    if include_vdw
                    else exclusion_scale_idxs
                ),
            )

    return smee.TensorTopology(
        atomic_nums=graph.atomic_numbers.long(),
        formal_charges=formal_charges,
        bond_idxs=graph.bond_idxs.long(),
        bond_orders=bond_orders,
        parameters=parameters,
    )


def build_smee_force_field(
    gnn_params: dict[str, dict[str, torch.Tensor] | None],
    charges: torch.Tensor | None = None,
    cutoff_angstrom: float = 9.0,
    switch_width_angstrom: float = 1.0,
    vdw_scale_14: float = 0.5,
    coul_scale_14: float = 0.8333333333,
) -> smee.TensorForceField:
    """Build a smee TensorForceField from GNN-predicted parameters.

    All parameter tensors are passed through WITHOUT detaching, so gradients
    flow back to the GNN weights.

    Parameters
    ----------
    gnn_params
        Output of ``model(graph)`` — the Readout output dict with keys
        ``"atom"``, ``"bond"``, ``"angle"``, ``"torsion"``, ``"improper"``.
        If atom params include ``"sigma"`` [nm] and ``"epsilon"`` [kJ/mol],
        a vdW potential is created. Other atom-level keys (e.g.
        ``"electronegativity"``, ``"hardness"``) are ignored here.
        Bond params: ``"k"`` [kJ/mol/nm^2] and ``"length"`` [nm].
        Angle params: ``"k"`` [kJ/mol/rad^2] and ``"angle"`` [rad].
        Torsion params: ``"k"`` [kJ/mol], shape ``(n_torsions, n_terms)``.
    charges
        Partial charges [e], shape ``(n_atoms,)``. Can be fixed (AM1-BCC)
        or GNN-predicted (e.g. from charge equilibration). If ``None``,
        no electrostatics potential is created.
    cutoff_angstrom
        Nonbonded cutoff distance in Angstrom.
    switch_width_angstrom
        Switch function width in Angstrom.
    vdw_scale_14
        1-4 vdW scaling factor (0.5 for SMIRNOFF).
    coul_scale_14
        1-4 electrostatic scaling factor (5/6 for SMIRNOFF).

    Returns
    -------
    smee.TensorForceField
    """
    _kcal_mol = unit.kilocalorie_per_mole
    _angstrom = unit.angstrom
    _radians = unit.radian
    _unitless = unit.dimensionless

    _kj_per_nm2_to_kcal_per_a2 = (
        1.0 * unit.kilojoule_per_mole / unit.nanometer**2
    ).m_as(_kcal_mol / _angstrom**2)
    _kj_to_kcal = (1.0 * unit.kilojoule_per_mole).m_as(_kcal_mol)
    _nm_to_angstrom = (1.0 * unit.nanometer).m_as(_angstrom)

    potentials = []

    # --- Bonds ---
    bond_params = gnn_params.get("bond")
    if bond_params is not None and bond_params["k"].shape[0] > 0:
        k_bond = bond_params["k"] * _kj_per_nm2_to_kcal_per_a2
        length_bond = bond_params["length"] * _nm_to_angstrom
        if k_bond.dim() == 2 and k_bond.shape[-1] == 1:
            k_bond = k_bond.squeeze(-1)
        if length_bond.dim() == 2 and length_bond.shape[-1] == 1:
            length_bond = length_bond.squeeze(-1)

        bond_tensor = torch.stack([k_bond, length_bond], dim=-1)
        potentials.append(
            smee.TensorPotential(
                type=smee.PotentialType.BONDS,
                fn=smee.EnergyFn.BOND_HARMONIC,
                parameters=bond_tensor,
                parameter_keys=_synthetic_keys("bond", bond_tensor.shape[0]),
                parameter_cols=("k", "length"),
                parameter_units=(_kcal_mol / _angstrom**2, _angstrom),
            )
        )

    # --- Angles ---
    angle_params = gnn_params.get("angle")
    if angle_params is not None and angle_params["k"].shape[0] > 0:
        k_angle = angle_params["k"] * _kj_to_kcal
        eq_angle = angle_params["angle"]
        if k_angle.dim() == 2 and k_angle.shape[-1] == 1:
            k_angle = k_angle.squeeze(-1)
        if eq_angle.dim() == 2 and eq_angle.shape[-1] == 1:
            eq_angle = eq_angle.squeeze(-1)

        angle_tensor = torch.stack([k_angle, eq_angle], dim=-1)
        potentials.append(
            smee.TensorPotential(
                type=smee.PotentialType.ANGLES,
                fn=smee.EnergyFn.ANGLE_HARMONIC,
                parameters=angle_tensor,
                parameter_keys=_synthetic_keys("angle", angle_tensor.shape[0]),
                parameter_cols=("k", "angle"),
                parameter_units=(_kcal_mol / _radians**2, _radians),
            )
        )

    # --- Proper torsions ---
    torsion_params = gnn_params.get("torsion")
    if torsion_params is not None and torsion_params["k"].shape[0] > 0:
        k_torsion = torsion_params["k"] * _kj_to_kcal
        n_torsions, n_terms = k_torsion.shape

        k_flat = k_torsion.reshape(-1)
        periodicity = torch.arange(
            1, n_terms + 1, dtype=k_flat.dtype, device=k_flat.device
        ).repeat(n_torsions)
        phase = torch.zeros_like(k_flat)
        idivf = torch.ones_like(k_flat)

        proper_tensor = torch.stack([k_flat, periodicity, phase, idivf], dim=-1)
        potentials.append(
            smee.TensorPotential(
                type=smee.PotentialType.PROPER_TORSIONS,
                fn=smee.EnergyFn.TORSION_COSINE,
                parameters=proper_tensor,
                parameter_keys=_synthetic_keys("proper", proper_tensor.shape[0]),
                parameter_cols=("k", "periodicity", "phase", "idivf"),
                parameter_units=(_kcal_mol, _unitless, _radians, _unitless),
            )
        )

    # --- Improper torsions ---
    improper_params = gnn_params.get("improper")
    if (
        improper_params is not None
        and "k" in improper_params
        and improper_params["k"].shape[0] > 0
    ):
        k_improper = improper_params["k"] * _kj_to_kcal
        n_imp, n_terms_imp = k_improper.shape

        k_imp_flat = k_improper.reshape(-1)
        periodicity_imp = torch.arange(
            1, n_terms_imp + 1, dtype=k_imp_flat.dtype, device=k_imp_flat.device
        ).repeat(n_imp)
        phase_imp = torch.zeros_like(k_imp_flat)
        idivf_imp = torch.ones_like(k_imp_flat)

        improper_tensor = torch.stack(
            [k_imp_flat, periodicity_imp, phase_imp, idivf_imp], dim=-1
        )
        potentials.append(
            smee.TensorPotential(
                type=smee.PotentialType.IMPROPER_TORSIONS,
                fn=smee.EnergyFn.TORSION_COSINE,
                parameters=improper_tensor,
                parameter_keys=_synthetic_keys("improper", improper_tensor.shape[0]),
                parameter_cols=("k", "periodicity", "phase", "idivf"),
                parameter_units=(_kcal_mol, _unitless, _radians, _unitless),
            )
        )

    # --- vdW ---
    atom_params = gnn_params.get("atom")
    has_vdw = (
        atom_params is not None and "epsilon" in atom_params and "sigma" in atom_params
    )
    if has_vdw:
        epsilon = atom_params["epsilon"] * _kj_to_kcal
        sigma = atom_params["sigma"] * _nm_to_angstrom
        if epsilon.dim() == 2 and epsilon.shape[-1] == 1:
            epsilon = epsilon.squeeze(-1)
        if sigma.dim() == 2 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        vdw_tensor = torch.stack([epsilon, sigma], dim=-1)

        vdw_attributes = torch.tensor(
            [0.0, 0.0, vdw_scale_14, 1.0, cutoff_angstrom, switch_width_angstrom],
            dtype=vdw_tensor.dtype,
        )
        potentials.append(
            smee.TensorPotential(
                type=smee.PotentialType.VDW,
                fn=smee.EnergyFn.VDW_LJ,
                parameters=vdw_tensor,
                parameter_keys=_synthetic_keys("vdw", vdw_tensor.shape[0]),
                parameter_cols=("epsilon", "sigma"),
                parameter_units=(_kcal_mol, _angstrom),
                attributes=vdw_attributes,
                attribute_cols=(
                    "scale_12",
                    "scale_13",
                    "scale_14",
                    "scale_15",
                    "cutoff",
                    "switch_width",
                ),
                attribute_units=(
                    _unitless,
                    _unitless,
                    _unitless,
                    _unitless,
                    _angstrom,
                    _angstrom,
                ),
            )
        )

    # --- Electrostatics ---
    if charges is not None:
        if charges.dim() == 2 and charges.shape[-1] == 1:
            charges = charges.squeeze(-1)

        charge_tensor = charges.unsqueeze(-1)  # (n_atoms, 1)
        coul_attributes = torch.tensor(
            [0.0, 0.0, coul_scale_14, 1.0, cutoff_angstrom],
            dtype=charge_tensor.dtype,
        )
        potentials.append(
            smee.TensorPotential(
                type=smee.PotentialType.ELECTROSTATICS,
                fn=smee.EnergyFn.COULOMB,
                parameters=charge_tensor,
                parameter_keys=_synthetic_keys("charge", charge_tensor.shape[0]),
                parameter_cols=("charge",),
                parameter_units=(_unitless,),
                attributes=coul_attributes,
                attribute_cols=(
                    "scale_12",
                    "scale_13",
                    "scale_14",
                    "scale_15",
                    "cutoff",
                ),
                attribute_units=(_unitless, _unitless, _unitless, _unitless, _angstrom),
            )
        )

    return smee.TensorForceField(potentials=potentials)


def convert_espaloma(
    graph,
    gnn_params: dict[str, dict[str, torch.Tensor] | None],
    charges: torch.Tensor | None = None,
    *,
    max_torsion_terms: int = 4,
    topology_cache: dict[str, smee.TensorTopology] | None = None,
) -> tuple[smee.TensorForceField, smee.TensorTopology]:
    """Convert espaloma GNN output to smee objects for energy evaluation.

    Parameters
    ----------
    graph
        Espaloma molecular graph (must have ``mol`` attribute).
    gnn_params
        Output of ``Readout(graph)`` — parameter dicts keyed by topology
        order (``"atom"``, ``"bond"``, ``"angle"``, ``"torsion"``,
        ``"improper"``). All tensors must be in espaloma's internal units
        (nm, kJ/mol, rad). If atom params include ``"sigma"`` and
        ``"epsilon"``, a vdW potential is created.
    charges
        Partial charges [e], shape ``(n_atoms,)``. Can be fixed (AM1-BCC)
        or GNN-predicted. If ``None``, no electrostatics potential is
        created.
    max_torsion_terms
        Fourier terms per torsion (must match GNN readout width).
    topology_cache
        Optional dict keyed by SMILES. If provided and the graph's SMILES
        is found, the cached topology is reused. The cache key includes
        which nonbonded terms are present, so the same molecule can be
        cached with different configurations.

    Returns
    -------
    tuple of (TensorForceField, TensorTopology)
        Force field with GNN-predicted parameters (in autograd graph)
        and the molecule topology. Ready for ``smee.compute_energy``.
    """
    smiles = getattr(graph, "smiles", None)

    atom_params = gnn_params.get("atom")
    include_vdw = (
        atom_params is not None and "epsilon" in atom_params and "sigma" in atom_params
    )
    include_electrostatics = charges is not None

    cache_key = (
        f"{smiles}:vdw={include_vdw}:elec={include_electrostatics}"
        if smiles is not None
        else None
    )

    # Infer dtype from GNN output so assignment matrices match param tensors.
    _dtype = torch.float64
    for d in gnn_params.values():
        if d is not None:
            for v in d.values():
                if v.is_floating_point():
                    _dtype = v.dtype
                    break
            else:
                continue
            break

    if (
        topology_cache is not None
        and cache_key is not None
        and cache_key in topology_cache
    ):
        topology = topology_cache[cache_key]
    else:
        topology = build_smee_topology(
            graph,
            max_torsion_terms=max_torsion_terms,
            include_vdw=include_vdw,
            include_electrostatics=include_electrostatics,
            dtype=_dtype,
        )
        if topology_cache is not None and cache_key is not None:
            topology_cache[cache_key] = topology

    ff = build_smee_force_field(gnn_params, charges)
    return ff, topology
