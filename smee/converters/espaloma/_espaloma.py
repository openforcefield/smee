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

_KCAL_MOL = unit.kilocalorie_per_mole
_ANGSTROM = unit.angstrom
_RADIANS = unit.radian
_UNITLESS = unit.dimensionless

_KJ_PER_NM2_TO_KCAL_PER_A2 = (1.0 * unit.kilojoule_per_mole / unit.nanometer**2).m_as(
    _KCAL_MOL / _ANGSTROM**2
)
_KJ_TO_KCAL = (1.0 * unit.kilojoule_per_mole).m_as(_KCAL_MOL)
_NM_TO_ANGSTROM = (1.0 * unit.nanometer).m_as(_ANGSTROM)


def _offset_sparse(
    n_rows: int,
    n_cols: int,
    col_offset: int,
    dtype: torch.dtype = torch.float64,
) -> torch.sparse.Tensor:
    """Return a sparse (n_rows, n_cols) matrix with ones on the diagonal
    starting at column ``col_offset``."""
    row_idx = torch.arange(n_rows)
    col_idx = row_idx + col_offset
    values = torch.ones(n_rows, dtype=dtype)
    return torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]), values, (n_rows, n_cols)
    )


def _synthetic_keys(
    prefix: str, n: int
) -> list[openff.interchange.models.PotentialKey]:
    return [
        openff.interchange.models.PotentialKey(id=f"{prefix}-{i}") for i in range(n)
    ]


def _count_interactions(graph, max_torsion_terms: int) -> dict[str, int]:
    """Count the number of parameter rows each term type contributes."""
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

    return {
        "atoms": n_atoms,
        "bonds": n_bonds,
        "angles": n_angles,
        "propers": n_propers * max_torsion_terms,
        "impropers": n_impropers * max_torsion_terms,
    }


def _build_topology(
    graph,
    max_torsion_terms: int,
    include_vdw: bool,
    include_electrostatics: bool,
    dtype: torch.dtype,
    offsets: dict[str, int],
    totals: dict[str, int],
) -> smee.TensorTopology:
    """Build a single TensorTopology with assignment matrices that index
    into a shared (concatenated) parameter table."""
    mol = graph.mol
    topology = mol.to_topology()

    counts = _count_interactions(graph, max_torsion_terms)

    formal_charges = torch.tensor(
        [atom.formal_charge.m_as(unit.e) for atom in topology.atoms],
        dtype=torch.long,
    )
    bond_orders = torch.tensor(
        [bond.bond_order for bond in topology.bonds], dtype=torch.long
    )

    parameters: dict[str, smee.ParameterMap] = {}

    if counts["bonds"] > 0:
        parameters["Bonds"] = smee.ValenceParameterMap(
            particle_idxs=graph.bond_idxs.long(),
            assignment_matrix=_offset_sparse(
                counts["bonds"], totals["bonds"], offsets["bonds"], dtype
            ),
        )

    if counts["angles"] > 0:
        parameters["Angles"] = smee.ValenceParameterMap(
            particle_idxs=graph.angle_idxs.long(),
            assignment_matrix=_offset_sparse(
                counts["angles"], totals["angles"], offsets["angles"], dtype
            ),
        )

    if counts["propers"] > 0:
        expanded = graph.torsion_idxs.long().repeat_interleave(max_torsion_terms, dim=0)
        parameters["ProperTorsions"] = smee.ValenceParameterMap(
            particle_idxs=expanded,
            assignment_matrix=_offset_sparse(
                counts["propers"], totals["propers"], offsets["propers"], dtype
            ),
        )

    improper_idxs = getattr(graph, "improper_idxs", None)
    if counts["impropers"] > 0:
        expanded = improper_idxs.long().repeat_interleave(max_torsion_terms, dim=0)
        parameters["ImproperTorsions"] = smee.ValenceParameterMap(
            particle_idxs=expanded,
            assignment_matrix=_offset_sparse(
                counts["impropers"], totals["impropers"], offsets["impropers"], dtype
            ),
        )

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
                assignment_matrix=_offset_sparse(
                    counts["atoms"], totals["atoms"], offsets["atoms"], dtype
                ),
                exclusions=exclusions,
                exclusion_scale_idxs=exclusion_scale_idxs,
            )

        if include_electrostatics:
            parameters["Electrostatics"] = smee.NonbondedParameterMap(
                assignment_matrix=_offset_sparse(
                    counts["atoms"], totals["atoms"], offsets["atoms"], dtype
                ),
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


def _concat_valence_params(
    all_params: list[dict[str, dict[str, torch.Tensor] | None]],
    key: str,
    param_names: list[str],
) -> dict[str, torch.Tensor] | None:
    """Concatenate a valence parameter type across molecules, returning None
    if no molecule has that parameter type."""
    parts = [p.get(key) for p in all_params]
    if all(p is None for p in parts):
        return None

    result = {}
    for name in param_names:
        tensors = []
        for p in parts:
            if p is not None and name in p:
                tensors.append(p[name])
        if tensors:
            result[name] = torch.cat(tensors, dim=0)

    return result if result else None


def build_smee_force_field(
    gnn_params: dict[str, dict[str, torch.Tensor] | None],
    charges: torch.Tensor | None = None,
    cutoff_angstrom: float = 9.0,
    switch_width_angstrom: float = 1.0,
    vdw_scale_14: float = 0.5,
    coul_scale_14: float = 0.8333333333,
) -> smee.TensorForceField:
    """Build a smee TensorForceField from (possibly concatenated) GNN parameters.

    All parameter tensors are passed through WITHOUT detaching, so gradients
    flow back to the GNN weights.

    Parameters
    ----------
    gnn_params
        Parameter dicts keyed by topology order (``"atom"``, ``"bond"``,
        ``"angle"``, ``"torsion"``, ``"improper"``). For multi-molecule
        systems, tensors should already be concatenated across molecules.
    charges
        Partial charges [e], shape ``(n_atoms_total,)``.
    cutoff_angstrom
        Nonbonded cutoff in Angstrom.
    switch_width_angstrom
        Switch function width in Angstrom.
    vdw_scale_14
        1-4 vdW scaling factor.
    coul_scale_14
        1-4 electrostatic scaling factor.
    """
    potentials = []

    # --- Bonds ---
    bond_params = gnn_params.get("bond")
    if bond_params is not None and bond_params["k"].shape[0] > 0:
        k_bond = bond_params["k"] * _KJ_PER_NM2_TO_KCAL_PER_A2
        length_bond = bond_params["length"] * _NM_TO_ANGSTROM
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
                parameter_units=(_KCAL_MOL / _ANGSTROM**2, _ANGSTROM),
            )
        )

    # --- Angles ---
    angle_params = gnn_params.get("angle")
    if angle_params is not None and angle_params["k"].shape[0] > 0:
        k_angle = angle_params["k"] * _KJ_TO_KCAL
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
                parameter_units=(_KCAL_MOL / _RADIANS**2, _RADIANS),
            )
        )

    # --- Proper torsions ---
    torsion_params = gnn_params.get("torsion")
    if torsion_params is not None and torsion_params["k"].shape[0] > 0:
        k_torsion = torsion_params["k"] * _KJ_TO_KCAL
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
                parameter_units=(_KCAL_MOL, _UNITLESS, _RADIANS, _UNITLESS),
            )
        )

    # --- Improper torsions ---
    improper_params = gnn_params.get("improper")
    if (
        improper_params is not None
        and "k" in improper_params
        and improper_params["k"].shape[0] > 0
    ):
        k_improper = improper_params["k"] * _KJ_TO_KCAL
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
                parameter_units=(_KCAL_MOL, _UNITLESS, _RADIANS, _UNITLESS),
            )
        )

    # --- vdW ---
    atom_params = gnn_params.get("atom")
    has_vdw = (
        atom_params is not None and "epsilon" in atom_params and "sigma" in atom_params
    )
    if has_vdw:
        epsilon = atom_params["epsilon"] * _KJ_TO_KCAL
        sigma = atom_params["sigma"] * _NM_TO_ANGSTROM
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
                parameter_units=(_KCAL_MOL, _ANGSTROM),
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
                    _UNITLESS,
                    _UNITLESS,
                    _UNITLESS,
                    _UNITLESS,
                    _ANGSTROM,
                    _ANGSTROM,
                ),
            )
        )

    # --- Electrostatics ---
    if charges is not None:
        if charges.dim() == 2 and charges.shape[-1] == 1:
            charges = charges.squeeze(-1)

        charge_tensor = charges.unsqueeze(-1)
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
                parameter_units=(_UNITLESS,),
                attributes=coul_attributes,
                attribute_cols=(
                    "scale_12",
                    "scale_13",
                    "scale_14",
                    "scale_15",
                    "cutoff",
                ),
                attribute_units=(
                    _UNITLESS,
                    _UNITLESS,
                    _UNITLESS,
                    _UNITLESS,
                    _ANGSTROM,
                ),
            )
        )

    return smee.TensorForceField(potentials=potentials)


def convert_espaloma(
    graphs,
    gnn_params: (
        dict[str, dict[str, torch.Tensor] | None]
        | list[dict[str, dict[str, torch.Tensor] | None]]
    ),
    charges: torch.Tensor | list[torch.Tensor] | None = None,
    *,
    max_torsion_terms: int = 4,
    dtype: torch.dtype = torch.float64,
    topology_cache: dict[str, smee.TensorTopology] | None = None,
) -> tuple[smee.TensorForceField, list[smee.TensorTopology]]:
    """Convert espaloma GNN output to smee objects for energy evaluation.

    Accepts one or more molecules. When multiple molecules are provided,
    parameter tensors are concatenated into a single ``TensorForceField``
    and each molecule gets its own ``TensorTopology`` whose assignment
    matrices index into the shared parameter table — the same layout
    ``convert_interchange`` uses.

    Parameters
    ----------
    graphs
        One or more espaloma molecular graphs (each must have a ``mol``
        attribute).
    gnn_params
        Output of ``Readout(graph)`` for each graph — parameter dicts
        keyed by topology order (``"atom"``, ``"bond"``, ``"angle"``,
        ``"torsion"``, ``"improper"``). A single dict is accepted for
        one molecule.
    charges
        Partial charges [e] for each molecule. A single tensor or a list
        of tensors, one per graph. ``None`` omits electrostatics.
    max_torsion_terms
        Fourier terms per torsion (must match GNN readout width).
    dtype
        Floating-point dtype for assignment matrices. Must match the
        GNN parameter tensors.
    topology_cache
        Optional dict keyed by SMILES. If provided and the graph's
        SMILES is found, the cached topology is reused.

    Returns
    -------
    tuple of (TensorForceField, list[TensorTopology])
    """
    # --- normalise inputs to lists ---
    if not isinstance(graphs, list):
        graphs = [graphs]
    if isinstance(gnn_params, dict):
        gnn_params = [gnn_params]
    if charges is None:
        charges_list: list[torch.Tensor | None] = [None] * len(graphs)
    elif isinstance(charges, torch.Tensor):
        charges_list = [charges]
    else:
        charges_list = charges

    n_molecules = len(graphs)
    assert len(gnn_params) == n_molecules
    assert len(charges_list) == n_molecules

    # --- determine which nonbonded terms to include ---
    include_vdw = any(
        p.get("atom") is not None and "epsilon" in p["atom"] and "sigma" in p["atom"]
        for p in gnn_params
    )
    include_electrostatics = any(c is not None for c in charges_list)

    # --- compute per-molecule interaction counts and cumulative offsets ---
    all_counts = [_count_interactions(g, max_torsion_terms) for g in graphs]
    totals = {key: sum(c[key] for c in all_counts) for key in all_counts[0]}
    cumulative_offsets = []
    running = dict.fromkeys(all_counts[0], 0)
    for c in all_counts:
        cumulative_offsets.append(dict(running))
        for key in running:
            running[key] += c[key]

    # --- build topologies ---
    topologies = []
    for i, graph in enumerate(graphs):
        smiles = getattr(graph, "smiles", None)
        cache_key = (
            f"{smiles}:vdw={include_vdw}:elec={include_electrostatics}:n={n_molecules}"
            if smiles is not None
            else None
        )

        cached = (
            topology_cache is not None
            and cache_key is not None
            and cache_key in topology_cache
        )

        if cached:
            topologies.append(topology_cache[cache_key])
        else:
            topo = _build_topology(
                graph,
                max_torsion_terms,
                include_vdw,
                include_electrostatics,
                dtype,
                cumulative_offsets[i],
                totals,
            )
            topologies.append(topo)
            if topology_cache is not None and cache_key is not None:
                topology_cache[cache_key] = topo

    # --- concatenate parameters across molecules ---
    combined_params: dict[str, dict[str, torch.Tensor] | None] = {}

    bond_concat = _concat_valence_params(gnn_params, "bond", ["k", "length"])
    if bond_concat is not None:
        combined_params["bond"] = bond_concat

    angle_concat = _concat_valence_params(gnn_params, "angle", ["k", "angle"])
    if angle_concat is not None:
        combined_params["angle"] = angle_concat

    torsion_concat = _concat_valence_params(gnn_params, "torsion", ["k"])
    if torsion_concat is not None:
        combined_params["torsion"] = torsion_concat

    improper_concat = _concat_valence_params(gnn_params, "improper", ["k"])
    if improper_concat is not None:
        combined_params["improper"] = improper_concat

    atom_concat = _concat_valence_params(gnn_params, "atom", ["sigma", "epsilon"])
    if atom_concat is not None:
        combined_params["atom"] = atom_concat

    combined_charges = None
    if include_electrostatics:
        charge_tensors = [c for c in charges_list if c is not None]
        if charge_tensors:
            combined_charges = torch.cat(charge_tensors, dim=0)

    ff = build_smee_force_field(combined_params, combined_charges)
    return ff, topologies
