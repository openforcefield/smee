"""Convert SMEE parameters to OpenMM ForceField XML files."""

import collections
import math
import typing
import uuid
import xml.etree.ElementTree as ElementTree

import openmm
import openmm.app
import torch

import smee

_CONVERTER_FUNCTIONS: dict[
    tuple[str, str],
    typing.Callable[
        [smee.TensorPotential, smee.TensorTopology, list[str]], ElementTree.Element
    ],
] = {}

_ANGSTROM_TO_NM = 1.0 / 10.0


def ffxml_converter(potential_type: str, energy_expression: str):
    """A decorator used to flag a function as being able to convert a tensor potential
    of a given type and energy function to an OpenMM force field XML representation.

    The decorated function should take a `smee.TensorPotential`, and
    the associated `smee.ParameterMap` and list of atom types, and return a
    ``xml.etree.ElementTree`` representing the potential.
    """

    def _openmm_converter_inner(func):
        if (potential_type, energy_expression) in _CONVERTER_FUNCTIONS:
            raise KeyError(
                f"An OpenMM converter function is already defined for "
                f"handler={potential_type} fn={energy_expression}."
            )

        _CONVERTER_FUNCTIONS[(str(potential_type), str(energy_expression))] = func
        return func

    return _openmm_converter_inner


def _has_bond(idx_1: int, idx_2: int, bonds: set[tuple[int, int]]) -> bool:
    """Check if a bond exists between two atoms."""
    return (idx_1, idx_2) in bonds or (idx_2, idx_1) in bonds


def _get_param_col(pot: smee.TensorPotential, param_name: str) -> tuple[int, float]:
    """Get the index of a parameter column."""

    param_col = pot.parameter_cols.index(param_name)
    param_conv = (
        (1.0 * pot.parameter_units[param_col])
        .to_openmm()
        .value_in_unit_system(openmm.unit.md_unit_system)
    )

    return param_col, param_conv


def _to_atom_types(
    omm_top: openmm.app.Topology, atom_name_to_type: dict[str, str]
) -> ElementTree.Element:
    atom_types_xml = ElementTree.Element("AtomTypes")

    for atom in omm_top.atoms():
        is_v_site = atom.name.startswith("X")

        attribs = {
            "name": atom_name_to_type[atom.name],
            "class": atom_name_to_type[atom.name],
        }

        if is_v_site:
            attribs["mass"] = "0.0"
        else:
            mass = atom.element.mass.value_in_unit_system(openmm.unit.md_unit_system)

            attribs["element"] = atom.element.symbol
            attribs["mass"] = str(mass)

        ElementTree.SubElement(atom_types_xml, "Type", **attribs)

    return atom_types_xml


def _to_residue(
    top: smee.TensorTopology,
    ff: smee.TensorForceField,
    omm_top: openmm.app.Topology,
    atom_name_to_type: dict[str, str],
    name: str,
) -> ElementTree.Element:
    """Add the residue matcher to the XML tree."""
    residues_xml = ElementTree.Element("Residues")
    residue_xml = ElementTree.SubElement(residues_xml, "Residue", name=name)

    for atom in omm_top.atoms():
        ElementTree.SubElement(
            residue_xml, "Atom", name=atom.name, type=atom_name_to_type[atom.name]
        )

    for bond in omm_top.bonds():
        ElementTree.SubElement(
            residue_xml,
            "Bond",
            atomName1=bond.atom1.name,
            atomName2=bond.atom2.name,
        )

    if top.v_sites is None:
        return residues_xml

    atom_names = [atom.name for atom in omm_top.atoms()]

    for v_site_idx, param_idx in enumerate(top.v_sites.parameter_idxs.tolist()):
        local_frame_coords = (
            smee.geometry.polar_to_cartesian_coords(
                ff.v_sites.parameters[[param_idx], :].detach()
            )
            * _ANGSTROM_TO_NM
        )
        weight_origin, weight_x, weight_y = ff.v_sites.weights[param_idx]

        parent_idxs = top.v_sites.keys[v_site_idx].orientation_atom_indices
        parent_names = {
            f"atomName{i + 1}": atom_names[parent_idx]
            for i, parent_idx in enumerate(parent_idxs)
        }

        ElementTree.SubElement(
            residue_xml,
            "VirtualSite",
            type="localCoords",
            siteName=f"X{v_site_idx + 1}",
            **parent_names,
            **{f"wo{i + 1}": str(weight_origin[i].item()) for i in range(3)},
            **{f"wx{i + 1}": str(weight_x[i].item()) for i in range(3)},
            **{f"wy{i + 1}": str(weight_y[i].item()) for i in range(3)},
            **{f"p{i + 1}": str(local_frame_coords[0, i].item()) for i in range(3)},
        )

    return residues_xml


def default_valence_converter(
    pot: smee.TensorPotential,
    param_map: smee.ValenceParameterMap,
    atom_types: list[str],
    param_cols: tuple[str, ...],
    force_type: str,
    force_tag: str,
    extra_attrs: dict[str, str] | None = None,
) -> ElementTree.Element:
    assert pot.exceptions is None, "valence potentials do not support exceptions."

    force_xml = ElementTree.Element(force_type)

    param_cols = {
        param_name: _get_param_col(pot, param_name) for param_name in param_cols
    }
    params = (param_map.assignment_matrix @ pot.parameters).detach().cpu().tolist()

    for param_idx, (idxs) in enumerate(param_map.particle_idxs):
        ElementTree.SubElement(
            force_xml,
            force_tag,
            **{f"class{i + 1}": atom_types[idx] for i, idx in enumerate(idxs)},
            **{
                param_name: str(
                    params[param_idx][param_cols[param_name][0]]
                    * param_cols[param_name][1]
                )
                for param_name in param_cols
            },
            **(extra_attrs if extra_attrs is not None else {}),
        )

    return force_xml


@ffxml_converter(smee.PotentialType.VDW, smee.EnergyFn.VDW_LJ)
def convert_lj_force(
    pot: smee.TensorPotential, param_map: smee.NonbondedParameterMap, types: list[str]
) -> ElementTree.Element:
    if pot.exceptions is not None:
        raise NotImplementedError("custom exclusions are not supported.")

    scale_12 = pot.attributes[pot.attribute_cols.index("scale_12")]
    assert torch.isclose(scale_12, torch.zeros_like(scale_12))
    scale_13 = pot.attributes[pot.attribute_cols.index("scale_13")]
    assert torch.isclose(scale_13, torch.zeros_like(scale_13))
    scale_15 = pot.attributes[pot.attribute_cols.index("scale_15")]
    assert torch.isclose(scale_15, torch.ones_like(scale_15))

    scale_14 = pot.attributes[pot.attribute_cols.index("scale_14")].detach().item()

    force_xml = ElementTree.Element(
        "NonbondedForce", attrib={"lj14scale": str(scale_14)}
    )

    params = (param_map.assignment_matrix @ pot.parameters).detach().cpu().tolist()

    eps_col, eps_scale = _get_param_col(pot, "epsilon")
    sig_col, sig_scale = _get_param_col(pot, "sigma")

    for atom_idx in range(len(params)):
        eps = float(params[atom_idx][eps_col] * eps_scale)
        sig = float(params[atom_idx][sig_col] * sig_scale)

        ElementTree.SubElement(
            force_xml,
            "Atom",
            sigma=str(sig),
            epsilon=str(eps),
            **{"class": types[atom_idx]},
        )

    return force_xml


@ffxml_converter(smee.PotentialType.ELECTROSTATICS, smee.EnergyFn.COULOMB)
def convert_electrostatics_force(
    pot: smee.TensorPotential, param_map: smee.NonbondedParameterMap, types: list[str]
) -> ElementTree.Element:
    if pot.exceptions is not None:
        raise NotImplementedError("custom exclusions are not supported.")

    scale_12 = pot.attributes[pot.attribute_cols.index("scale_12")]
    assert torch.isclose(scale_12, torch.zeros_like(scale_12))
    scale_13 = pot.attributes[pot.attribute_cols.index("scale_13")]
    assert torch.isclose(scale_13, torch.zeros_like(scale_13))
    scale_15 = pot.attributes[pot.attribute_cols.index("scale_15")]
    assert torch.isclose(scale_15, torch.ones_like(scale_15))

    scale_14 = pot.attributes[pot.attribute_cols.index("scale_14")].detach().item()

    force_xml = ElementTree.Element(
        "NonbondedForce", attrib={"coulomb14scale": str(scale_14)}
    )

    params = (param_map.assignment_matrix @ pot.parameters).detach().cpu().tolist()

    charge_col, charge_scale = _get_param_col(pot, "charge")

    for atom_idx in range(len(params)):
        charge = float(params[atom_idx][charge_col] * charge_scale)

        ElementTree.SubElement(
            force_xml, "Atom", charge=str(charge), **{"class": types[atom_idx]}
        )

    return force_xml


@ffxml_converter(smee.PotentialType.BONDS, smee.EnergyFn.BOND_HARMONIC)
def convert_bond_potential(
    pot: smee.TensorPotential, param_map: smee.ValenceParameterMap, types: list[str]
):
    return default_valence_converter(
        pot, param_map, types, ("k", "length"), "HarmonicBondForce", "Bond"
    )


@ffxml_converter(smee.PotentialType.ANGLES, smee.EnergyFn.ANGLE_HARMONIC)
def convert_angle_potential(
    pot: smee.TensorPotential, param_map: smee.ValenceParameterMap, types: list[str]
):
    return default_valence_converter(
        pot, param_map, types, ("k", "angle"), "HarmonicAngleForce", "Angle"
    )


@ffxml_converter(smee.PotentialType.PROPER_TORSIONS, smee.EnergyFn.TORSION_COSINE)
@ffxml_converter(smee.PotentialType.IMPROPER_TORSIONS, smee.EnergyFn.TORSION_COSINE)
def convert_torsion_potential(
    pot: smee.TensorPotential, param_map: smee.ValenceParameterMap, types: list[str]
):
    is_proper = pot.type == smee.PotentialType.PROPER_TORSIONS

    params = (param_map.assignment_matrix @ pot.parameters).detach().cpu().tolist()

    param_names = ("k", "periodicity", "phase", "idivf")
    param_scales = {
        param_name: _get_param_col(pot, param_name) for param_name in param_names
    }
    param_idx_by_particle_idxs = collections.defaultdict(list)

    for param_idx, idxs in enumerate(param_map.particle_idxs.tolist()):
        param_idx_by_particle_idxs[tuple(idxs)].append(param_idx)

    force_xml = ElementTree.Element("PeriodicTorsionForce", ordering="smirnoff")

    for (idx_1, idx_2, idx_3, idx_4), param_idxs in param_idx_by_particle_idxs.items():
        tag = "Proper" if is_proper else "Improper"

        attrib = {
            "class1": types[idx_1],
            "class2": types[idx_2],
            "class3": types[idx_3],
            "class4": types[idx_4],
            "ordering": "smirnoff",
        }

        for i, param_idx in enumerate(param_idxs):
            param_vals = {
                param_name: params[param_idx][param_scales[param_name][0]]
                * param_scales[param_name][1]
                for param_name in param_names
            }
            param_vals["k"] /= param_vals.pop("idivf")
            param_vals["periodicity"] = int(param_vals["periodicity"])

            attrib.update(
                {
                    f"{param_name}{i + 1}": str(param_val)
                    for param_name, param_val in param_vals.items()
                }
            )

        ElementTree.SubElement(force_xml, tag, attrib=attrib)

    return force_xml


def _add_constraints(
    top: smee.TensorTopology,
    types: list[str],
    bond_xml: ElementTree.Element,
    angle_xml: ElementTree.Element,
):
    bonds = set((idx_a, idx_b) for idx_a, idx_b in top.bond_idxs.tolist())  # noqa: C401

    def has_bond(i, j):
        return (i, j) in bonds or (j, i) in bonds

    neighbours = collections.defaultdict(set)

    for bond in bonds:
        neighbours[bond[0]].add(bond[1])
        neighbours[bond[1]].add(bond[0])

    angles = {
        (idx_a, idx_c): (idx_a, idx_b, idx_c)
        for idx_a in range(top.n_atoms)
        for idx_b in neighbours[idx_a]
        for idx_c in neighbours[idx_b]
        if idx_a != idx_c and not has_bond(idx_a, idx_c)
    }

    constraints = {
        (int(idx_a), int(idx_b)): dist.item() * _ANGSTROM_TO_NM
        for (idx_a, idx_b), dist in zip(
            top.constraints.idxs, top.constraints.distances, strict=True
        )
    }

    def get_constraint(i, j):
        return constraints.get((i, j), constraints.get((j, i)))

    for (idx_a, idx_b), dist in constraints.items():
        if has_bond(idx_a, idx_b):
            ElementTree.SubElement(
                bond_xml,
                "Bond",
                class1=types[idx_a],
                class2=types[idx_b],
                length=str(dist),
                k="0.0",
            )
        elif (idx_a, idx_b) in angles or (idx_b, idx_a) in angles:
            angle_idxs = angles.get((idx_a, idx_b), angles.get((idx_b, idx_a)))

            dist_ab = get_constraint(angle_idxs[0], angle_idxs[1])
            dist_bc = get_constraint(angle_idxs[1], angle_idxs[2])
            dist_ac = get_constraint(angle_idxs[0], angle_idxs[2])

            angle = math.acos(
                -(dist_ac**2 - dist_ab**2 - dist_bc**2) / (2 * dist_ab * dist_bc)
            )

            ElementTree.SubElement(
                angle_xml,
                "Angle",
                class1=types[angle_idxs[0]],
                class2=types[angle_idxs[1]],
                class3=types[angle_idxs[2]],
                angle=str(angle),
                k="0.0",
            )
        else:
            raise NotImplementedError(
                f"cannot convert constraint between atoms {idx_a} and {idx_b}."
            )


def _convert_to_openmm_ffxml(
    ff: smee.TensorForceField, top: smee.TensorTopology, element_counts: dict[str:int]
) -> str:
    import smee.converters

    omm_top = smee.converters.convert_to_openmm_topology(top)

    atom_name_to_type: dict[str, str] = {}
    short_uuid = str(uuid.uuid4())[:8]

    for atom in omm_top.atoms():
        atom_symbol = None if atom.element is None else atom.element.symbol

        if atom.name.startswith("X"):
            atom_type = f"X{element_counts[atom_symbol] + 1}"
        else:
            atom_type = f"{atom_symbol}{element_counts[atom_symbol] + 1}"

        atom_name_to_type[atom.name] = f"smee-{short_uuid}-{atom_type}"
        element_counts[atom_symbol] += 1

    atom_types = [*atom_name_to_type.values()]

    root_xml = ElementTree.Element("ForceField")
    root_xml.append(_to_atom_types(omm_top, atom_name_to_type))
    root_xml.append(_to_residue(top, ff, omm_top, atom_name_to_type, "UNK"))

    xml_by_tag = collections.defaultdict(list)

    for potential_type, parameter_map in top.parameters.items():
        potential = ff.potentials_by_type[potential_type]

        converter_key = (potential_type, potential.fn)

        if converter_key not in _CONVERTER_FUNCTIONS:
            raise NotImplementedError(
                f"cannot convert type={potential_type} fn={potential.fn} to an "
                f"OpenMM FFXML file."
            )

        converter = _CONVERTER_FUNCTIONS[converter_key]

        force_xml = converter(potential, parameter_map, atom_types)
        xml_by_tag[force_xml.tag].append(force_xml)

    assert len(xml_by_tag["NonbondedForce"]) <= 2

    if len(xml_by_tag["NonbondedForce"]) == 2:
        if "lj14scale" in xml_by_tag["NonbondedForce"][0].attrib:
            force_xml, coul_force_xml = xml_by_tag["NonbondedForce"]
        else:
            coul_force_xml, force_xml = xml_by_tag["NonbondedForce"]

        force_xml.attrib.update(coul_force_xml.attrib)

        for lj_atom, coul_atom in zip(force_xml, coul_force_xml, strict=True):
            assert lj_atom.attrib["class"] == coul_atom.attrib["class"]
            lj_atom.attrib.update(coul_atom.attrib)

        xml_by_tag["NonbondedForce"] = [force_xml]

    if top.constraints is not None:
        if not xml_by_tag["HarmonicBondForce"]:
            xml_by_tag["HarmonicBondForce"] = [ElementTree.Element("HarmonicBondForce")]
        if not xml_by_tag["HarmonicAngleForce"]:
            xml_by_tag["HarmonicAngleForce"] = [
                ElementTree.Element("HarmonicAngleForce")
            ]

        _add_constraints(
            top,
            atom_types,
            xml_by_tag["HarmonicBondForce"][0],
            xml_by_tag["HarmonicAngleForce"][0],
        )

    for xmls in xml_by_tag.values():
        root_xml.extend(xmls)

    return ElementTree.tostring(root_xml, encoding="unicode")


def convert_to_openmm_ffxml(
    force_field: smee.TensorForceField, system: smee.TensorSystem | smee.TensorTopology
) -> list[str]:
    """Convert a SMEE force field and system to OpenMM force field XML
    representations.

    Args:
        force_field: The force field to convert.
        system: The system to convert.

    Returns:
        One OpenMM force field XML representation per topology in the system.
    """
    if isinstance(system, smee.TensorTopology):
        system = smee.TensorSystem([system], [1], False)

    element_counts: dict[str, int] = collections.defaultdict(int)

    ffxml_contents = []

    for top in system.topologies:
        top_ffxml = _convert_to_openmm_ffxml(force_field, top, element_counts)
        ffxml_contents.append(top_ffxml)

    return ffxml_contents
