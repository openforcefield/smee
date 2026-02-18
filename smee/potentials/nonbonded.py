"""Non-bonded potential energy functions."""

import collections
import math
import typing

import openff.units
import torch

import smee.potentials
import smee.utils

_UNIT = openff.units.unit

_COULOMB_PRE_FACTOR_UNITS = _UNIT.kilocalorie / _UNIT.mole * _UNIT.angstrom / _UNIT.e**2
_COULOMB_PRE_FACTOR = (_UNIT.avogadro_constant / (4.0 * _UNIT.pi * _UNIT.eps0)).m_as(
    _COULOMB_PRE_FACTOR_UNITS
)

_PME_MIN_NODES = torch.tensor(6)  # taken to match OpenMM 8.0.0
_PME_ORDER = 5  # see OpenMM issue #2567


class PairwiseDistances(typing.NamedTuple):
    """A container for the pairwise distances between all particles, possibly within
    a given cutoff."""

    idxs: torch.Tensor
    """The particle indices of each pair with ``shape=(n_pairs, 2)``."""
    deltas: torch.Tensor
    """The vector between each pair with ``shape=(n_pairs, 3)``."""
    distances: torch.Tensor
    """The distance between each pair with ``shape=(n_pairs,)``."""

    cutoff: torch.Tensor | None = None
    """The cutoff used when computing the distances."""


def _broadcast_exclusions(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcasts the exclusions (indices and scale factors) of each topology to the
    full system.

    Args:
        system: The system.
        potential: The potential containing the scale factors to broadcast.

    Returns:
        The exception indices with shape ``(n_exceptions, 2)`` and the scale factors
        with shape ``(n_exceptions,)``.
    """

    idx_offset = 0

    per_topology_exclusion_idxs = []
    per_topology_exclusion_scales = []

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        exclusion_idxs = topology.parameters[potential.type].exclusions

        exclusion_offset = (
            idx_offset
            + smee.utils.arange_like(n_copies, exclusion_idxs) * topology.n_particles
        )
        idx_offset += n_copies * topology.n_particles

        if len(exclusion_idxs) == 0:
            continue

        exclusion_idxs = exclusion_offset[:, None, None] + exclusion_idxs[None, :, :]

        exclusion_scales = potential.attributes[
            topology.parameters[potential.type].exclusion_scale_idxs
        ]
        exclusion_scales = torch.broadcast_to(
            exclusion_scales, (n_copies, *exclusion_scales.shape)
        )

        per_topology_exclusion_idxs.append(exclusion_idxs.reshape(-1, 2))
        per_topology_exclusion_scales.append(exclusion_scales.reshape(-1))

    system_idxs = (
        torch.zeros((0, 2), dtype=torch.int32)
        if len(per_topology_exclusion_idxs) == 0
        else torch.vstack(per_topology_exclusion_idxs)
    )
    system_scales = (
        torch.zeros((0,), dtype=torch.float32)
        if len(per_topology_exclusion_scales) == 0
        else torch.cat(per_topology_exclusion_scales)
    )

    return system_idxs, system_scales


def compute_pairwise_scales(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Returns the scale factor for each pair of particles in the system by
    broadcasting and stacking the exclusions of each topology.

    Args:
        system: The system.
        potential: The potential containing the scale factors to broadcast.

    Returns:
        The scales for each pair of particles as a flattened upper triangular matrix
        with ``shape=(n_particles * (n_particles - 1) / 2,)``.
    """

    n_particles = system.n_particles
    n_pairs = (n_particles * (n_particles - 1)) // 2

    exclusion_idxs, exclusion_scales = _broadcast_exclusions(system, potential)

    pair_scales = smee.utils.ones_like(n_pairs, other=potential.parameters)

    if len(exclusion_idxs) > 0:
        exclusion_idxs, _ = exclusion_idxs.sort(dim=1)  # ensure upper triangle

        pair_idxs = smee.utils.to_upper_tri_idx(
            exclusion_idxs[:, 0], exclusion_idxs[:, 1], n_particles
        )
        pair_scales[pair_idxs] = exclusion_scales

    return pair_scales


def _compute_pairwise_periodic(
    conformer: torch.Tensor, box_vectors: torch.Tensor, cutoff: torch.Tensor
) -> PairwiseDistances:
    import NNPOps.neighbors

    assert box_vectors is not None, "box vectors must be specified for PBC."
    assert len(conformer.shape) == 2, "the conformer must not have a batch dimension."

    (
        pair_idxs,
        deltas,
        distances,
        _,
    ) = NNPOps.neighbors.getNeighborPairs(conformer, cutoff.item(), -1, box_vectors)

    are_interacting = ~torch.isnan(distances)

    distances = distances[are_interacting]
    deltas = deltas[are_interacting, :]
    pair_idxs = pair_idxs[:, are_interacting]
    # we sort the indices to get values correponding to upper triangles
    # but we need to track which have been reversed so we can reverse the deltas
    pair_idxs, indices = pair_idxs.sort(dim=0)
    reversed = -(indices[0] == 1).to(deltas.dtype)
    deltas = deltas * reversed[:, None]

    return PairwiseDistances(pair_idxs.T.contiguous(), deltas, distances, cutoff)


def _compute_pairwise_non_periodic(conformer: torch.Tensor) -> PairwiseDistances:
    n_particles = conformer.shape[-2]

    pair_idxs = torch.triu_indices(n_particles, n_particles, 1, dtype=torch.int32).T

    if conformer.ndim == 2:
        deltas = conformer[pair_idxs[:, 1], :] - conformer[pair_idxs[:, 0], :]
    else:
        deltas = conformer[:, pair_idxs[:, 1], :] - conformer[:, pair_idxs[:, 0], :]
    distances = deltas.norm(dim=-1)

    return PairwiseDistances(pair_idxs.contiguous(), deltas, distances)


def compute_pairwise(
    system: smee.TensorSystem,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None,
    cutoff: torch.Tensor,
) -> PairwiseDistances:
    """Computes all pairwise distances between particles in the system.

    Notes:
        If the system is not periodic, no cutoff and no PBC will be applied.

    Args:
        system: The system to compute the distances for.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        cutoff: The cutoff [Å] to apply for periodic systems.

    Returns:
        The pairwise distances between each pair of particles within the cutoff.
    """
    if system.is_periodic:
        return _compute_pairwise_periodic(conformer, box_vectors, cutoff)
    else:
        return _compute_pairwise_non_periodic(conformer)


def prepare_lrc_types(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finds the unique vdW interactions present in a system, ready to use
    for computing the long range dispersion correction.

    Args:
        system: The system to prepare the types for.
        potential: The potential to prepare the types for.

    Returns:
        Two tensors containing the i and j indices into ``potential.paramaters`` of
        each unique interaction parameter excluding ``i==j``, the number of ``ii``
        interactions with ``shape=(n_params,)``, and the numbers of ``ij`` interactions
        with ``shape=(len(idxs_i),)``.
    """
    n_by_type = collections.defaultdict(int)

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_counts = topology.parameters["vdW"].assignment_matrix.abs().sum(dim=0)

        for key, count in zip(potential.parameter_keys, parameter_counts, strict=True):
            n_by_type[key] += count.item() * n_copies

    counts = smee.utils.tensor_like(
        [n_by_type[key] for key in potential.parameter_keys], potential.parameters
    )

    n_ii_interactions = (counts * (counts + 1.0)) / 2.0

    idxs_i, idxs_j = torch.triu_indices(len(counts), len(counts), 1)
    n_ij_interactions = counts[idxs_i] * counts[idxs_j]

    idxs_ii = torch.arange(len(counts))

    idxs_i = torch.cat([idxs_i, idxs_ii])
    idxs_j = torch.cat([idxs_j, idxs_ii])

    n_ij_interactions = torch.cat([n_ij_interactions, n_ii_interactions])

    return idxs_i, idxs_j, n_ij_interactions


def lorentz_berthelot(
    epsilon_a: torch.Tensor,
    epsilon_b: torch.Tensor,
    sigma_a: torch.Tensor,
    sigma_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the Lorentz-Berthelot combination rules for the given parameters.

    Notes:
        A 'safe' geometric mean is used to avoid NaNs when the parameters are zero.
        This will yield non-analytic gradients in some cases.

    Args:
        epsilon_a: The epsilon [kcal / mol] values of the first particle in each pair
            with ``shape=(n_pairs, 1)``.
        epsilon_b: The epsilon [kcal / mol] values of the second particle in each pair
            with ``shape=(n_pairs, 1)``.
        sigma_a: The sigma [kcal / mol] values of the first particle in each pair
            with ``shape=(n_pairs, 1)``.
        sigma_b: The sigma [kcal / mol] values of the second particle in each pair
            with ``shape=(n_pairs, 1)``.

    Returns:
        The epsilon [kcal / mol] and sigma [Å] values of each pair, each with
        ``shape=(n_pairs, 1)``.
    """
    return smee.utils.geometric_mean(epsilon_a, epsilon_b), 0.5 * (sigma_a + sigma_b)


def _compute_switch_fn(
    potential: smee.TensorPotential,
    pairwise: PairwiseDistances,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if smee.SWITCH_ATTRIBUTE not in potential.attribute_cols:
        return torch.ones(1), None

    switch_width_idx = potential.attribute_cols.index(smee.SWITCH_ATTRIBUTE)
    switch_width = pairwise.cutoff - potential.attributes[switch_width_idx]

    x_switch = (pairwise.distances - switch_width) / (pairwise.cutoff - switch_width)

    switch_fn = 1.0 - 6.0 * x_switch**5 + 15.0 * x_switch**4 - 10.0 * x_switch**3

    switch_fn = torch.where(
        pairwise.distances < switch_width, torch.tensor(1.0), switch_fn
    )
    switch_fn = torch.where(
        pairwise.distances > pairwise.cutoff, torch.tensor(0.0), switch_fn
    )

    return switch_fn, switch_width


def _integrate_lj_switch(
    r: torch.Tensor,
    rs: torch.Tensor,
    rc: torch.Tensor,
    sig: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the intergral of the LJ potential multiplied by one minus the switching
    function.

    We define `b = 1 / (rc - rs)` and integrate

    (sig^12/r^10 - sig^6/r^4) (6 b^5 (r - rs)^5 - 15 b^4 (r - rs)^4 + 10 b^3 (r - rs)^3) dr

    to get:

    -(b**3 * c**6 *
        (
            - 28   * a**3        * c**6 *      (6 * a**2 * b**2 + 15 * a * b + 10) * r**-9
            + 945  * a**2        * c**6 *      (    a**2 * b**2 + 2  * a * b + 1 ) * r**-8
            - 1080 * a           * c**6 *      (2 * a**2 * b**2 + 3  * a * b + 1 ) * r**-7
            + 420                * c**6 *      (6 * a**2 * b**2 + 6  * a * b + 1 ) * r**-6
            - 756         * b    * c**6 *      (                  2  * a * b + 1 ) * r**-5
            + 378         * b**2 * c**6 *      (                               1 ) * r**-4
            + 84   * a**3               *      (6 * a**2 * b**2 + 15 * a * b + 10) * r**-3
            - 3780 * a**2               *      (    a**2 * b**2 + 2  * a * b + 1 ) * r**-2
            + 7560 * a                  *      (2 * a**2 * b**2 + 3  * a * b + 1 ) * r**-1
            + 2520                      *      (6 * a**2 * b**2 + 6  * a * b + 1 ) * log(r)
            - 3780        * b           *      (                  2  * a * b + 1 ) * r**1
            + 756         * b**2                                                   * r**2
        )
    ) / 252

    Args:
        r: The distance to evaluate the integral at.
        rs: The switching distance.
        rc: The cutoff distance.
        sig: The sigma parameter of the LJ potential with ``shape=(n_params,)``.

    Returns:
        The value of the integral.
    """
    b = 1.0 / (rc - rs)

    coeff_0 = smee.utils.tensor_like([rs**3, rs**2, rs, 1, b, b**2], rs) * (
        smee.utils.tensor_like([rs**2 * b**2, rs * b, 1], rs)
        * smee.utils.tensor_like(
            [[6, 15, 10], [1, 2, 1], [2, 3, 1], [6, 6, 1], [0, 2, 1], [0, 0, 1]], rs
        )
    ).sum(dim=-1)

    coeff_01 = (
        sig[:, None] ** 6
        * smee.utils.tensor_like([-28, 945, -1080, 420, -756, 378], rs)
        * coeff_0
    )
    coeff_11 = smee.utils.tensor_like([84, -3780, 7560, 2520, -3780, 756], rs) * coeff_0

    r_pow = torch.pow(
        r, smee.utils.tensor_like([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2], rs)
    )
    r_pow[-3] = torch.log(r)

    integral = (
        -(b**3)
        * sig**6
        * (coeff_01 * r_pow[:6] + coeff_11 * r_pow[6:]).sum(dim=-1)
        / 252
    )

    return integral


def _compute_lj_lrc(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    rs: torch.Tensor | None,
    rc: torch.Tensor | None,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Computes the long range dispersion correction due to the double exponential
    potential, possibly with a switching function."""

    idxs_i, idxs_j, n_ij_interactions = smee.potentials.nonbonded.prepare_lrc_types(
        system, potential
    )

    eps_col = potential.parameter_cols.index("epsilon")
    sig_col = potential.parameter_cols.index("sigma")

    eps_ii, sig_ii = (
        potential.parameters[:, eps_col],
        potential.parameters[:, sig_col],
    )
    eps_ij, sig_ij = smee.potentials.nonbonded.lorentz_berthelot(
        eps_ii[idxs_i], eps_ii[idxs_j], sig_ii[idxs_i], sig_ii[idxs_j]
    )

    if potential.exceptions is not None:
        exceptions = {
            **potential.exceptions,
            **{(j, i): idx for (i, j), idx in potential.exceptions.items()},
        }

        eps_ij, sig_ij = eps_ij.clone(), sig_ij.clone()  # prevent in-place modification

        for i, (idx_i, idx_j) in enumerate(zip(idxs_i, idxs_j, strict=True)):
            idx_i, idx_j = int(idx_i), int(idx_j)

            if (idx_i, idx_j) not in exceptions:
                continue

            param_idx = exceptions[(idx_i, idx_j)]

            eps_ij[i] = potential.parameters[param_idx, eps_col]
            sig_ij[i] = potential.parameters[param_idx, sig_col]

    integral = sig_ij**12 / (9 * rc**9) - sig_ij**6 / (3 * rc**3)

    if rs is not None:
        integral_rc = _integrate_lj_switch(rc, rs, rc, sig_ij)
        integral_rs = _integrate_lj_switch(rs, rs, rc, sig_ij)

        integral += integral_rc - integral_rs

    integral = (n_ij_interactions * 4.0 * eps_ij * integral).sum(dim=-1)
    integral /= n_ij_interactions.sum()

    return 2.0 * system.n_particles**2 * torch.pi / volume * integral


@smee.potentials.potential_energy_fn(smee.PotentialType.VDW, smee.EnergyFn.VDW_LJ)
def compute_lj_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise: PairwiseDistances | None = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] of the vdW interactions using the
    standard Lennard-Jones potential.

    Notes:
        * No cutoff / switching function will be applied if the system is not
          periodic.
        * A switching function will only be applied if the potential has a
          ``switch_width`` attribute.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs, 3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        pairwise: The pre-computed pairwise distances between each pair of particles
            in the system. If none, these will be computed within the function.

    Returns:
        The computed potential energy [kcal / mol] with ``shape=(n_confs,)`` if the
        input conformer has a batch dimension, or ``shape=()`` otherwise.
    """

    box_vectors = None if not system.is_periodic else box_vectors

    cutoff = potential.attributes[potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)]

    pairwise = (
        pairwise
        if pairwise is not None
        else compute_pairwise(system, conformer, box_vectors, cutoff)
    )

    if system.is_periodic and not torch.isclose(pairwise.cutoff, cutoff):
        raise ValueError("the pairwise cutoff does not match the potential.")

    parameters = smee.potentials.broadcast_parameters(system, potential)
    pair_scales = compute_pairwise_scales(system, potential)

    pairs_1d = smee.utils.to_upper_tri_idx(
        pairwise.idxs[:, 0], pairwise.idxs[:, 1], len(parameters)
    )
    pair_scales = pair_scales[pairs_1d]

    eps_column = potential.parameter_cols.index("epsilon")
    sig_column = potential.parameter_cols.index("sigma")

    eps, sig = lorentz_berthelot(
        parameters[pairwise.idxs[:, 0], eps_column],
        parameters[pairwise.idxs[:, 1], eps_column],
        parameters[pairwise.idxs[:, 0], sig_column],
        parameters[pairwise.idxs[:, 1], sig_column],
    )

    if potential.exceptions is not None:
        exception_idxs, exceptions = smee.potentials.broadcast_exceptions(
            system, potential, pairwise.idxs[:, 0], pairwise.idxs[:, 1]
        )

        eps, sig = eps.clone(), sig.clone()  # prevent in-place modification

        eps[exception_idxs] = exceptions[:, eps_column]
        sig[exception_idxs] = exceptions[:, sig_column]

    x = (sig / pairwise.distances) ** 6
    energies = pair_scales * 4.0 * eps * (x * (x - 1.0))

    if not system.is_periodic:
        return energies.sum(-1)

    switch_fn, switch_width = _compute_switch_fn(potential, pairwise)
    energies *= switch_fn

    energy = energies.sum(-1)
    energy += _compute_lj_lrc(
        system,
        potential.to(precision="double"),
        switch_width.double(),
        pairwise.cutoff.double(),
        torch.det(box_vectors),
    )

    return energy


def _integrate_dexp_switch(
    r: torch.Tensor,
    rs: torch.Tensor,
    rc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the intergral of one term (repulsive or attractive) of the double
    exponential potential multiplied by one minus the switching function.

    We define `rc_rs = 1 / (rc - rs)` and expand r^2 * (6 * rc_rs**5 * (r - rs)**5 + 15 * rc_rs**4 * (r - rs)**4 - 10 * rc_rs**3 * (r - rs)**3)
    into:

      (+ 6 * rs**5 * rc_rs**5 + 15 * rs**4 * rc_rs**4 + 10 * rs**3 * rc_rs**3) * r**2
    + (-30 * rs**4 * rc_rs**5 - 60 * rs**3 * rc_rs**4 - 30 * rs**2 * rc_rs**3) * r**3
    + ( 60 * rs**3 * rc_rs**5 + 90 * rs**2 * rc_rs**4 + 30 * rs    * rc_rs**3) * r**4
    + (-60 * rs**2 * rc_rs**5 - 60 * rs    * rc_rs**4 - 10         * rc_rs**3) * r**5
    + ( 30 * rs    * rc_rs**5 + 15         * rc_rs**4)                         * r**6
    + (- 6         * rc_rs**5)                                                 * r**7

    We then define c_n as the coefficient of r^n, then integral of c_n * r**n * a * exp(-b * x):

    integral r^2 = c_0 * a * exp(-b * r) * (-2    / b**3 - 2    * r / b**2 -        r**2 / b)
    integral r^3 = c_1 * a * exp(-b * r) * (-6    / b**4 - 6    * r / b**3 - 3    * r**2 / b**2 -       r**3 / b)
    integral r^4 = c_2 * a * exp(-b * r) * (-24   / b**5 - 24   * r / b**4 - 12   * r**2 / b**3 - 4   * r**3 / b**2 -       r**4 / b)
    integral r^5 = c_3 * a * exp(-b * r) * (-120  / b**6 - 120  * r / b**5 - 60   * r**2 / b**4 - 20  * r**3 / b**3 - 5   * r**4 / b**2 -      r**5 / b)
    integral r^6 = c_4 * a * exp(-b * r) * (-720  / b**7 - 720  * r / b**6 - 360  * r**2 / b**5 - 120 * r**3 / b**4 - 30  * r**4 / b**3 - 6  * r**5 / b**2 -     r**6 / b)
    integral r^7 = c_5 * a * exp(-b * r) * (-5040 / b**8 - 5040 * r / b**7 - 2520 * r**2 / b**6 - 840 * r**3 / b**5 - 210 * r**4 / b**4 - 42 * r**5 / b**3 - 7 * r**6 / b**2 - r**7 / b)

    so the integral of the entire function is the sum of the integrals of each power of
    r

    Args:
        r: The distance to evaluate the integral at.
        rs: The switching distance.
        rc: The cutoff distance.
        a: The prefactor of the exponential term.
        b: The exponent of the exponential term.
    """
    rs_pow = smee.utils.tensor_like(
        [rs**5, rs**4, rs**3, rs**2, rs, 1, 0, 0], rc
    ).unsqueeze(1)

    # fmt: off
    c_n = smee.utils.tensor_like(
        [
            [  6,  15,  10],  # noqa: E201,E241
            [-30, -60, -30],  # noqa: E201,E241
            [ 60,  90,  30],  # noqa: E201,E241
            [-60, -60, -10],  # noqa: E201,E241
            [ 30,  15,   0],  # noqa: E201,E241
            [-6,    0,   0],  # noqa: E201,E241
        ],
        rs
    )
    # fmt: on
    c_n *= torch.hstack(
        [
            rs_pow[0:6] * (rc - rs) ** -5,
            rs_pow[1:7] * (rc - rs) ** -4,
            rs_pow[2:8] * (rc - rs) ** -3,
        ]
    )
    c_n = c_n.sum(dim=1)

    b = b.unsqueeze(1)
    b_pow = torch.hstack(
        [smee.utils.zeros_like((len(b), 1), b)] * 4
        + [torch.ones_like(b), b, b**2, b**3, b**4, b**5, b**6, b**7, b**8],
    )

    mat = torch.stack(
        [
            1 / b_pow[:, 7:],
            r / b_pow[:, 6:-1],
            r**2 / b_pow[:, 5:-2],
            r**3 / b_pow[:, 4:-3],
            r**4 / b_pow[:, 3:-4],
            r**5 / b_pow[:, 2:-5],
            r**6 / b_pow[:, 1:-6],
            r**7 / b_pow[:, :-7],
        ],
        dim=-1,
    )
    mat = torch.where(torch.isinf(mat), torch.zeros_like(mat), mat)

    # fmt: off
    mat_coeff = smee.utils.tensor_like(
        [
            [-2,    -2,    -1,     0,    0,    0,   0,  0],  # noqa: E241
            [-6,    -6,    -3,    -1,    0,    0,   0,  0],  # noqa: E241
            [-24,   -24,   -12,   -4,   -1,    0,   0,  0],  # noqa: E241
            [-120,  -120,  -60,   -20,  -5,   -1,   0,  0],  # noqa: E241
            [-720,  -720,  -360,  -120, -30,  -6,  -1,  0],  # noqa: E241
            [-5040, -5040, -2520, -840, -210, -42, -7, -1],  # noqa: E241
        ],
        rc
    )
    # fmt: on

    int_n = a * torch.exp(-b * r) * c_n * (mat * mat_coeff).sum(dim=-1)
    return -int_n.sum(dim=-1)


def _compute_dexp_lrc(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    rs: torch.Tensor | None,
    rc: torch.Tensor | None,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Computes the long range dispersion correction due to the double exponential
    potential, possibly with a switching function."""

    idxs_i, idxs_j, n_ij_interactions = prepare_lrc_types(system, potential)

    rs = None if rs is None else rs.double()
    rc = rc.double()

    alpha = potential.attributes[potential.attribute_cols.index("alpha")]
    beta = potential.attributes[potential.attribute_cols.index("beta")]

    eps_col = potential.parameter_cols.index("epsilon")
    r_min_col = potential.parameter_cols.index("r_min")

    eps_ii, r_min_ii = (
        potential.parameters[:, eps_col],
        potential.parameters[:, r_min_col],
    )
    eps_ij, r_min_ij = smee.potentials.nonbonded.lorentz_berthelot(
        eps_ii[idxs_i], eps_ii[idxs_j], r_min_ii[idxs_i], r_min_ii[idxs_j]
    )

    if potential.exceptions is not None:
        exceptions = {
            **potential.exceptions,
            **{(j, i): idx for (i, j), idx in potential.exceptions.items()},
        }

        # prevent in-place modification
        eps_ij, r_min_ij = eps_ij.clone(), r_min_ij.clone()

        for i, (idx_i, idx_j) in enumerate(zip(idxs_i, idxs_j, strict=True)):
            idx_i, idx_j = int(idx_i), int(idx_j)

            if (idx_i, idx_j) not in exceptions:
                continue

            param_idx = exceptions[(idx_i, idx_j)]

            eps_ij[i] = potential.parameters[param_idx, eps_col]
            r_min_ij[i] = potential.parameters[param_idx, r_min_col]

    a_rep = beta * torch.exp(alpha) / (alpha - beta)
    b_rep = alpha / r_min_ij

    a_att = alpha * torch.exp(beta) / (alpha - beta)
    b_att = beta / r_min_ij

    integral = -(
        a_rep
        * torch.exp(-b_rep * rc)
        * (-2 / b_rep**3 - 2 * rc / b_rep**2 - rc**2 / b_rep)
        + a_att
        * torch.exp(-b_att * rc)
        * (2 / b_att**3 + 2 * rc / b_att**2 + rc**2 / b_att)
    )

    if rs is not None:
        integral_rep_rs = _integrate_dexp_switch(rs, rs, rc, a_rep, b_rep)
        integral_rep_rc = _integrate_dexp_switch(rc, rs, rc, a_rep, b_rep)
        integral_rep_rs_to_rc = integral_rep_rc - integral_rep_rs

        integral_att_rs = _integrate_dexp_switch(rs, rs, rc, a_att, b_att)
        integral_att_rc = _integrate_dexp_switch(rc, rs, rc, a_att, b_att)
        integral_att_rs_to_rc = integral_att_rc - integral_att_rs

        integral += integral_rep_rs_to_rc - integral_att_rs_to_rc

    integral = (n_ij_interactions * eps_ij * integral).sum(dim=-1)
    integral /= n_ij_interactions.sum()

    return 2.0 * system.n_particles**2 * torch.pi / volume * integral


@smee.potentials.potential_energy_fn(smee.PotentialType.VDW, smee.EnergyFn.VDW_DEXP)
def compute_dexp_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise: PairwiseDistances | None = None,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of the vdW interactions using the
    double-exponential potential.

    Notes:
        * No cutoff function will be applied if the system is not periodic.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs, 3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        pairwise: Pre-computed distances between each pair of particles
            in the system.

    Returns:
        The evaluated potential energy [kcal / mol].
    """
    box_vectors = None if not system.is_periodic else box_vectors

    cutoff = potential.attributes[potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)]

    pairwise = (
        pairwise
        if pairwise is not None
        else compute_pairwise(system, conformer, box_vectors, cutoff)
    )

    if system.is_periodic and not torch.isclose(pairwise.cutoff, cutoff):
        raise ValueError("the pairwise cutoff does not match the potential.")

    parameters = smee.potentials.broadcast_parameters(system, potential)
    pair_scales = compute_pairwise_scales(system, potential)

    pairs_1d = smee.utils.to_upper_tri_idx(
        pairwise.idxs[:, 0], pairwise.idxs[:, 1], len(parameters)
    )
    pair_scales = pair_scales[pairs_1d]

    eps_column = potential.parameter_cols.index("epsilon")
    r_min_column = potential.parameter_cols.index("r_min")

    eps, r_min = smee.potentials.nonbonded.lorentz_berthelot(
        parameters[pairwise.idxs[:, 0], eps_column],
        parameters[pairwise.idxs[:, 1], eps_column],
        parameters[pairwise.idxs[:, 0], r_min_column],
        parameters[pairwise.idxs[:, 1], r_min_column],
    )

    if potential.exceptions is not None:
        exception_idxs, exceptions = smee.potentials.broadcast_exceptions(
            system, potential, pairwise.idxs[:, 0], pairwise.idxs[:, 1]
        )

        eps, r_min = eps.clone(), r_min.clone()  # prevent in-place modification

        eps[exception_idxs] = exceptions[:, eps_column]
        r_min[exception_idxs] = exceptions[:, r_min_column]

    alpha = potential.attributes[potential.attribute_cols.index("alpha")]
    beta = potential.attributes[potential.attribute_cols.index("beta")]

    x = pairwise.distances / r_min

    energies_repulsion = beta / (alpha - beta) * torch.exp(alpha * (1.0 - x))
    energies_attraction = alpha / (alpha - beta) * torch.exp(beta * (1.0 - x))

    energies = pair_scales * eps * (energies_repulsion - energies_attraction)

    if not system.is_periodic:
        return energies.sum(-1)

    switch_fn, switch_width = _compute_switch_fn(potential, pairwise)
    energies *= switch_fn

    energy = energies.sum(-1)

    energy += _compute_dexp_lrc(
        system,
        potential.to(precision="double"),
        switch_width.double(),
        pairwise.cutoff.double(),
        torch.det(box_vectors),
    )

    return energy


def _compute_pme_exclusions(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Builds the exclusion tensor required by NNPOps pme functions

    Args:
        system: The system to compute the exclusions for.
        potential: The electrostatics potential.

    Returns:
        The exclusion tensor with ``shape=(n_particles, max_exclusions)`` where
        ``max_exclusions`` is the maximum number of exclusions of any atom. A value
        of -1 is used for padding.
    """
    exclusion_templates = [
        [[] for _ in range(topology.n_particles)] for topology in system.topologies
    ]
    max_exclusions = 0

    for exclusions, topology in zip(
        exclusion_templates, system.topologies, strict=True
    ):
        for i, j in topology.parameters[potential.type].exclusions:
            exclusions[i].append(int(j))
            exclusions[j].append(int(i))

            max_exclusions = max(len(exclusions[i]), max_exclusions)
            max_exclusions = max(len(exclusions[j]), max_exclusions)

    idx_offset = 0

    exclusions_per_type = []

    for exclusions, topology, n_copies in zip(
        exclusion_templates, system.topologies, system.n_copies, strict=True
    ):
        for atom_exclusions in exclusions:
            n_padding = max_exclusions - len(atom_exclusions)
            atom_exclusions.extend([-1] * n_padding)

        exclusions = torch.tensor(exclusions, dtype=torch.int32)

        exclusion_offset = idx_offset + torch.arange(n_copies) * topology.n_particles
        idx_offset += n_copies * topology.n_particles

        if exclusions.shape[-1] == 0:
            continue

        exclusions = torch.broadcast_to(
            exclusions, (n_copies, len(exclusions), max_exclusions)
        )
        exclusions = torch.where(
            exclusions >= 0, exclusions + exclusion_offset[:, None, None], exclusions
        )

        exclusions_per_type.append(exclusions.reshape(-1, max_exclusions))

    return (
        torch.zeros((0, 0), dtype=torch.int32)
        if len(exclusions_per_type) == 0
        else torch.vstack(exclusions_per_type)
    )


def _compute_pme_grid(
    box_vectors: torch.Tensor, cutoff: torch.Tensor, error_tolerance: torch.Tensor
) -> tuple[int, int, int, float]:
    alpha = torch.sqrt(-torch.log(2.0 * error_tolerance)) / cutoff

    factor = 2.0 * alpha / (3 * error_tolerance ** (1.0 / 5.0))

    grid_x = torch.maximum(torch.ceil(factor * box_vectors[0, 0]), _PME_MIN_NODES)
    grid_y = torch.maximum(torch.ceil(factor * box_vectors[1, 1]), _PME_MIN_NODES)
    grid_z = torch.maximum(torch.ceil(factor * box_vectors[2, 2]), _PME_MIN_NODES)

    return int(grid_x), int(grid_y), int(grid_z), float(alpha)


def _compute_coulomb_energy_non_periodic(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    pairwise: PairwiseDistances,
):
    parameters = smee.potentials.broadcast_parameters(system, potential)
    pair_scales = compute_pairwise_scales(system, potential)

    energy = (
        _COULOMB_PRE_FACTOR
        * pair_scales
        * parameters[pairwise.idxs[:, 0], 0]
        * parameters[pairwise.idxs[:, 1], 0]
        / pairwise.distances
    ).sum(-1)

    return energy


def _compute_coulomb_energy_periodic(
    system: smee.TensorSystem,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor,
    potential: smee.TensorPotential,
    pairwise: PairwiseDistances,
) -> torch.Tensor:
    import NNPOps.pme

    assert system.is_periodic, "the system must be periodic."

    charges = smee.potentials.broadcast_parameters(system, potential).squeeze(-1)

    cutoff = potential.attributes[potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)]
    error_tol = torch.tensor(0.0001)

    exceptions = _compute_pme_exclusions(system, potential).to(charges.device)

    grid_x, grid_y, grid_z, alpha = _compute_pme_grid(box_vectors, cutoff, error_tol)

    pme = NNPOps.pme.PME(
        grid_x, grid_y, grid_z, _PME_ORDER, alpha, _COULOMB_PRE_FACTOR, exceptions
    )

    energy_direct = torch.ops.pme.pme_direct(
        conformer.float(),
        charges.float(),
        pairwise.idxs.T,
        pairwise.deltas,
        pairwise.distances,
        pme.exclusions,
        pme.alpha,
        pme.coulomb,
    )
    energy_self = -torch.sum(charges**2) * pme.coulomb * pme.alpha / math.sqrt(torch.pi)
    energy_recip = energy_self + torch.ops.pme.pme_reciprocal(
        conformer.float(),
        charges.float(),
        box_vectors.float(),
        pme.gridx,
        pme.gridy,
        pme.gridz,
        pme.order,
        pme.alpha,
        pme.coulomb,
        pme.moduli[0].to(charges.device),
        pme.moduli[1].to(charges.device),
        pme.moduli[2].to(charges.device),
    )

    exclusion_idxs, exclusion_scales = _broadcast_exclusions(system, potential)

    exclusion_distances = (
        conformer[exclusion_idxs[:, 0], :] - conformer[exclusion_idxs[:, 1], :]
    ).norm(dim=-1)

    energy_exclusion = (
        _COULOMB_PRE_FACTOR
        * exclusion_scales
        * charges[exclusion_idxs[:, 0]]
        * charges[exclusion_idxs[:, 1]]
        / exclusion_distances
    ).sum(-1)

    return energy_direct + energy_recip + energy_exclusion


@smee.potentials.potential_energy_fn(
    smee.PotentialType.ELECTROSTATICS, smee.EnergyFn.COULOMB
)
def compute_coulomb_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise: PairwiseDistances | None = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] of the electrostatic interactions
    using the Coulomb potential.

    Notes:
        * No cutoff will be applied if the system is not periodic.
        * PME will be used to compute the energy if the system is periodic.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs, 3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        pairwise: The pre-computed pairwise distances between each pair of particles
            in the system. If none, these will be computed within the function.

    Returns:
        The computed potential energy [kcal / mol] with ``shape=(n_confs,)`` if the
        input conformer has a batch dimension, or ``shape=()`` otherwise.
    """

    box_vectors = None if not system.is_periodic else box_vectors

    cutoff = potential.attributes[potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)]

    pairwise = (
        pairwise
        if pairwise is not None
        else compute_pairwise(system, conformer, box_vectors, cutoff)
    )

    if system.is_periodic and not torch.isclose(pairwise.cutoff, cutoff):
        raise ValueError("the distance cutoff does not match the potential.")

    if potential.exceptions is not None:
        raise NotImplementedError("exceptions are not supported for charges.")

    if system.is_periodic:
        return _compute_coulomb_energy_periodic(
            system, conformer, box_vectors, potential, pairwise
        )
    else:
        return _compute_coulomb_energy_non_periodic(system, potential, pairwise)
