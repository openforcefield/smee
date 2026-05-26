import pytest

pytest.importorskip("openmm")

import openff.interchange
import openff.toolkit

from smee.converters.openff.valence import (
    convert_angles,
    convert_bonds,
    convert_impropers,
    convert_propers,
)


def test_convert_bonds(ethanol, ethanol_interchange):
    bond_collection = ethanol_interchange.collections["Bonds"]

    potential, parameter_maps = convert_bonds([bond_collection], [set()])

    assert potential.type == "Bonds"
    assert potential.fn == "k/2*(r-length)**2"

    assert potential.attributes is None
    assert potential.attribute_cols is None

    assert potential.parameter_cols == ("k", "length")

    parameter_keys = [key.id for key in potential.parameter_keys]
    expected_parameter_keys = [
        "[#6:1]-[#8:2]",
        "[#6X4:1]-[#1:2]",
        "[#6X4:1]-[#6X4:2]",
        "[#8:1]-[#1:2]",
    ]
    assert sorted(parameter_keys) == sorted(expected_parameter_keys)

    assert potential.parameters.shape == (4, 2)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert len(parameter_map.assignment_matrix) == len(parameter_map.particle_idxs)
    assignment_matrix = parameter_map.assignment_matrix.to_dense()

    actual_parameters = {
        tuple(particle_idxs.tolist()): parameter_keys[parameter_idxs.nonzero()]
        for parameter_idxs, particle_idxs in zip(
            assignment_matrix, parameter_map.particle_idxs, strict=True
        )
    }
    expected_parameters = {
        (0, 2): "[#6:1]-[#8:2]",
        (0, 3): "[#8:1]-[#1:2]",
        (1, 2): "[#6X4:1]-[#6X4:2]",
        (1, 4): "[#6X4:1]-[#1:2]",
        (1, 5): "[#6X4:1]-[#1:2]",
        (1, 6): "[#6X4:1]-[#1:2]",
        (2, 7): "[#6X4:1]-[#1:2]",
        (2, 8): "[#6X4:1]-[#1:2]",
    }

    assert actual_parameters == expected_parameters


def test_convert_bonds_with_constraints(ethanol):
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-1.3.0.offxml"), ethanol.to_topology()
    )

    bond_collection = interchange.collections["Bonds"]

    constraints = {
        (bond.atom1_index, bond.atom2_index)
        for bond in ethanol.bonds
        if bond.atom1.atomic_number == 1 or bond.atom2.atomic_number == 1
    }

    potential, [parameter_map] = convert_bonds([bond_collection], [constraints])
    parameter_keys = [key.id for key in potential.parameter_keys]

    assert len(parameter_map.assignment_matrix) == len(parameter_map.particle_idxs)
    assignment_matrix = parameter_map.assignment_matrix.to_dense()

    actual_parameters = {
        tuple(particle_idxs.tolist()): parameter_keys[parameter_idxs.nonzero()]
        for parameter_idxs, particle_idxs in zip(
            assignment_matrix, parameter_map.particle_idxs, strict=True
        )
    }
    expected_parameters = {(0, 2): "[#6:1]-[#8:2]", (1, 2): "[#6X4:1]-[#6X4:2]"}

    assert actual_parameters == expected_parameters


@pytest.mark.parametrize("with_constraints", [True, False])
def test_convert_angles_etoh(ethanol, ethanol_interchange, with_constraints):
    angle_collection = ethanol_interchange.collections["Angles"]

    h_bond_idxs = {
        (bond.atom1_index, bond.atom2_index)
        for bond in ethanol.bonds
        if bond.atom1.atomic_number == 1 or bond.atom2.atomic_number == 1
    }
    constraints = set() if not with_constraints else h_bond_idxs

    potential, parameter_maps = convert_angles([angle_collection], [constraints])

    assert potential.type == "Angles"
    assert potential.fn == "k/2*(theta-angle)**2"

    assert potential.attributes is None
    assert potential.attribute_cols is None

    assert potential.parameter_cols == ("k", "angle")

    parameter_keys = [key.id for key in potential.parameter_keys]
    expected_parameter_keys = [
        "[#1:1]-[#6X4:2]-[#1:3]",
        "[*:1]-[#8:2]-[*:3]",
        "[*:1]~[#6X4:2]-[*:3]",
    ]
    assert sorted(parameter_keys) == sorted(expected_parameter_keys)

    assert potential.parameters.shape == (3, 2)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert len(parameter_map.assignment_matrix) == len(parameter_map.particle_idxs)
    assignment_matrix = parameter_map.assignment_matrix.to_dense()

    actual_parameters = {
        tuple(particle_idxs.tolist()): parameter_keys[parameter_idxs.nonzero()]
        for parameter_idxs, particle_idxs in zip(
            assignment_matrix, parameter_map.particle_idxs, strict=True
        )
    }
    expected_parameters = {
        (0, 2, 1): "[*:1]~[#6X4:2]-[*:3]",
        (0, 2, 7): "[*:1]~[#6X4:2]-[*:3]",
        (0, 2, 8): "[*:1]~[#6X4:2]-[*:3]",
        (1, 2, 7): "[*:1]~[#6X4:2]-[*:3]",
        (1, 2, 8): "[*:1]~[#6X4:2]-[*:3]",
        (2, 0, 3): "[*:1]-[#8:2]-[*:3]",
        (2, 1, 4): "[*:1]~[#6X4:2]-[*:3]",
        (2, 1, 5): "[*:1]~[#6X4:2]-[*:3]",
        (2, 1, 6): "[*:1]~[#6X4:2]-[*:3]",
        (4, 1, 5): "[#1:1]-[#6X4:2]-[#1:3]",
        (4, 1, 6): "[#1:1]-[#6X4:2]-[#1:3]",
        (5, 1, 6): "[#1:1]-[#6X4:2]-[#1:3]",
        (7, 2, 8): "[#1:1]-[#6X4:2]-[#1:3]",
    }

    assert actual_parameters == expected_parameters


@pytest.mark.parametrize("with_constraints", [True, False])
def test_convert_angle_water(with_constraints):
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-1.3.0.offxml"),
        openff.toolkit.Molecule.from_mapped_smiles("[O:1]([H:2])[H:3]").to_topology(),
    )

    angle_collection = interchange.collections["Angles"]

    constraints = set() if not with_constraints else {(0, 1), (0, 2), (1, 2)}

    potential, [parameter_map] = convert_angles([angle_collection], [constraints])
    parameter_keys = [key.id for key in potential.parameter_keys]

    assert len(parameter_map.assignment_matrix) == len(parameter_map.particle_idxs)
    assignment_matrix = parameter_map.assignment_matrix.to_dense()

    actual_parameters = {
        tuple(particle_idxs.tolist()): parameter_keys[parameter_idxs.nonzero()]
        for parameter_idxs, particle_idxs in zip(
            assignment_matrix, parameter_map.particle_idxs, strict=True
        )
    }
    expected_parameters = {} if with_constraints else {(1, 0, 2): "[*:1]-[#8:2]-[*:3]"}

    assert actual_parameters == expected_parameters


def test_convert_propers(ethanol, ethanol_interchange):
    proper_collection = ethanol_interchange.collections["ProperTorsions"]

    potential, parameter_maps = convert_propers([proper_collection])

    assert potential.type == "ProperTorsions"
    assert potential.fn == "k*(1+cos(periodicity*theta-phase))"

    hcco_smirks = "[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]"
    ccoh_smirks = "[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]"
    xcoh_smirks = "[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]"
    hcch_smirks = "[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]"

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert len(parameter_map.assignment_matrix) == len(parameter_map.particle_idxs)
    assignment_matrix = parameter_map.assignment_matrix.to_dense()

    actual_parameters = {
        (
            tuple(particle_idxs.tolist()),
            potential.parameter_keys[parameter_idxs.nonzero()].id,
            potential.parameter_keys[parameter_idxs.nonzero()].mult,
        )
        for parameter_idxs, particle_idxs in zip(
            assignment_matrix, parameter_map.particle_idxs, strict=True
        )
    }
    expected_parameters = {
        ((0, 2, 1, 4), hcco_smirks, 1),
        ((0, 2, 1, 4), hcco_smirks, 0),
        ((0, 2, 1, 5), hcco_smirks, 1),
        ((0, 2, 1, 5), hcco_smirks, 0),
        ((0, 2, 1, 6), hcco_smirks, 1),
        ((0, 2, 1, 6), hcco_smirks, 0),
        ((1, 2, 0, 3), ccoh_smirks, 1),
        ((1, 2, 0, 3), ccoh_smirks, 0),
        ((3, 0, 2, 7), xcoh_smirks, 0),
        ((3, 0, 2, 8), xcoh_smirks, 0),
        ((4, 1, 2, 7), hcch_smirks, 0),
        ((4, 1, 2, 8), hcch_smirks, 0),
        ((5, 1, 2, 7), hcch_smirks, 0),
        ((5, 1, 2, 8), hcch_smirks, 0),
        ((6, 1, 2, 7), hcch_smirks, 0),
        ((6, 1, 2, 8), hcch_smirks, 0),
    }
    assert actual_parameters == expected_parameters


def test_convert_impropers(formaldehyde, formaldehyde_interchange):
    improper_collection = formaldehyde_interchange.collections["ImproperTorsions"]

    potential, parameter_maps = convert_impropers([improper_collection])

    assert potential.type == "ImproperTorsions"
    assert potential.fn == "k*(1+cos(periodicity*theta-phase))"
