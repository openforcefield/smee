import openff.interchange
import openff.interchange.models
import openff.toolkit
import openff.units
import torch

import smee
import smee.converters
from smee.converters.openff.nonbonded import (
    convert_dampedexp6810,
    convert_dexp,
    convert_electrostatics,
    convert_vdw,
)


def test_convert_electrostatics_am1bcc(ethanol, ethanol_interchange):
    charge_collection = ethanol_interchange.collections["Electrostatics"]

    potential, parameter_maps = convert_electrostatics(
        [charge_collection], [ethanol.to_topology()], [None]
    )

    assert potential.type == "Electrostatics"
    assert potential.fn == "coul"

    expected_attributes = torch.tensor(
        [0.0, 0.0, 5.0 / 6.0, 1.0, 9.0], dtype=torch.float64
    )
    assert torch.allclose(potential.attributes, expected_attributes)
    assert potential.attribute_cols == (
        "scale_12",
        "scale_13",
        "scale_14",
        "scale_15",
        smee.CUTOFF_ATTRIBUTE,
    )

    assert potential.parameter_cols == ("charge",)

    assert all(
        parameter_key.id == "[O:1]([C:3]([C:2]([H:5])([H:6])[H:7])([H:8])[H:9])[H:4]"
        for parameter_key in potential.parameter_keys
    )
    assert potential.parameters.shape == (9, 1)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert parameter_map.assignment_matrix.shape == (ethanol.n_atoms, ethanol.n_atoms)
    assert torch.allclose(
        parameter_map.assignment_matrix.to_dense(),
        torch.eye(ethanol.n_atoms, dtype=torch.float64),
    )

    n_expected_exclusions = 36
    assert parameter_map.exclusions.shape == (n_expected_exclusions, 2)
    assert parameter_map.exclusion_scale_idxs.shape == (n_expected_exclusions, 1)


def test_convert_electrostatics_v_site():
    force_field = openff.toolkit.ForceField()
    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler("vdW")

    charge_handler = force_field.get_parameter_handler("LibraryCharges")
    charge_handler.add_parameter(
        {
            "smirks": "[Cl:1]-[H:2]",
            "charge1": -0.75 * openff.units.unit.e,
            "charge2": 0.25 * openff.units.unit.e,
        }
    )

    v_site_handler = force_field.get_parameter_handler("VirtualSites")
    v_site_handler.add_parameter(
        {
            "type": "BondCharge",
            "smirks": "[Cl:1]-[H:2]",
            "distance": 2.0 * openff.units.unit.angstrom,
            "match": "all_permutations",
            "charge_increment1": -0.25 * openff.units.unit.e,
            "charge_increment2": 0.5 * openff.units.unit.e,
        }
    )

    molecule = openff.toolkit.Molecule.from_mapped_smiles("[Cl:2]-[H:1]")

    interchange = openff.interchange.Interchange.from_smirnoff(
        force_field, molecule.to_topology(), allow_nonintegral_charges=True
    )
    charge_collection = interchange.collections["Electrostatics"]

    potential, parameter_maps = convert_electrostatics(
        [charge_collection],
        [molecule.to_topology()],
        [
            smee.VSiteMap(
                keys=[*interchange.collections["VirtualSites"].key_map],
                key_to_idx=interchange.collections[
                    "VirtualSites"
                ].virtual_site_key_topology_index_map,
                parameter_idxs=torch.tensor([[0]]),
            )
        ],
    )

    assert potential.parameter_cols == ("charge",)
    expected_keys = [
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2]", mult=0, associated_handler="LibraryCharges"
        ),
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2]", mult=1, associated_handler="LibraryCharges"
        ),
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2] EP all_permutations",
            mult=0,
            associated_handler="Electrostatics",
        ),
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2] EP all_permutations",
            mult=1,
            associated_handler="Electrostatics",
        ),
    ]
    assert potential.parameter_keys == expected_keys
    assert potential.parameters.shape == (4, 1)

    expected_parameters = torch.tensor(
        [[-0.75], [0.25], [-0.25], [0.5]], dtype=torch.float64
    )
    assert torch.allclose(potential.parameters, expected_parameters)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    n_particles = 3

    assert parameter_map.assignment_matrix.shape == (n_particles, len(expected_keys))
    expected_assignment_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(
        parameter_map.assignment_matrix.to_dense(), expected_assignment_matrix
    )

    n_expected_exclusions = 3
    assert parameter_map.exclusions.shape == (n_expected_exclusions, 2)
    assert parameter_map.exclusion_scale_idxs.shape == (n_expected_exclusions, 1)

    expected_exclusions = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long)
    assert torch.allclose(parameter_map.exclusions, expected_exclusions)

    expected_scales = torch.zeros((n_expected_exclusions, 1), dtype=torch.long)
    assert torch.allclose(parameter_map.exclusion_scale_idxs, expected_scales)


def test_convert_electrostatics_tip4p():
    """Explicitly test the case of TIP4P (FB) water to make sure v-site charges are
    correct.
    """

    force_field = openff.toolkit.ForceField("tip4p_fb.offxml")
    molecule = openff.toolkit.Molecule.from_mapped_smiles("[H:2][O:1][H:3]")

    interchange = openff.interchange.Interchange.from_smirnoff(
        force_field, molecule.to_topology(), allow_nonintegral_charges=True
    )

    tensor_top: smee.TensorTopology
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    q = 0.5258681106763
    expected_charges = torch.tensor([[0.0], [q], [q], [-2.0 * q]], dtype=torch.float64)

    charges = (
        tensor_top.parameters["Electrostatics"].assignment_matrix
        @ tensor_ff.potentials_by_type["Electrostatics"].parameters
    )
    assert charges.shape == expected_charges.shape
    assert torch.allclose(charges, expected_charges)


def test_convert_bci_and_vsite():
    ff_off = openff.toolkit.ForceField()
    ff_off.get_parameter_handler("Electrostatics")
    ff_off.get_parameter_handler("vdW")

    charge_handler = ff_off.get_parameter_handler("ChargeIncrementModel")
    charge_handler.partial_charge_method = "am1-mulliken"
    charge_handler.add_parameter(
        {"smirks": "[O:1]-[H:2]", "charge_increment1": -0.1 * openff.units.unit.e}
    )
    v_site_handler = ff_off.get_parameter_handler("VirtualSites")
    v_site_handler.add_parameter(
        {
            "type": "DivalentLonePair",
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "distance": -0.1 * openff.units.unit.angstrom,
            "outOfPlaneAngle": 0.0 * openff.units.unit.degree,
            "match": "once",
            "charge_increment1": 0.0 * openff.units.unit.e,
            "charge_increment2": 0.53 * openff.units.unit.e,
            "charge_increment3": 0.53 * openff.units.unit.e,
        }
    )

    mol = openff.toolkit.Molecule.from_mapped_smiles("[O:1]([H:2])[H:3]")
    mol.assign_partial_charges(charge_handler.partial_charge_method)

    interchange = openff.interchange.Interchange.from_smirnoff(
        ff_off, mol.to_topology()
    )

    expected_charges = [
        q.m_as("e") for q in interchange.collections["Electrostatics"].charges.values()
    ]

    ff, [top] = smee.converters.convert_interchange(interchange)

    charge_pot = ff.potentials_by_type["Electrostatics"]

    assert charge_pot.attribute_cols == (
        "scale_12",
        "scale_13",
        "scale_14",
        "scale_15",
        "cutoff",
    )
    assert torch.allclose(
        charge_pot.attributes,
        torch.tensor([0.0000, 0.0000, 1.0 / 1.2, 1.0000, 9.0000], dtype=torch.float64),
    )

    assert charge_pot.parameter_cols == ("charge",)

    found_keys = [
        (key.associated_handler, key.id, key.mult) for key in charge_pot.parameter_keys
    ]
    expected_keys = [
        ("ChargeModel", "[O:1]([H:2])[H:3]", 0),
        ("ChargeModel", "[O:1]([H:2])[H:3]", 1),
        ("ChargeModel", "[O:1]([H:2])[H:3]", 2),
        ("ChargeModel", "[#1:2]-[#8X2H2+0:1]-[#1:3] EP once", 0),
        ("ChargeModel", "[#1:2]-[#8X2H2+0:1]-[#1:3] EP once", 1),
        ("ChargeModel", "[#1:2]-[#8X2H2+0:1]-[#1:3] EP once", 2),
        ("ChargeIncrementModel", "[O:1]-[H:2]", 0),
    ]
    assert found_keys == expected_keys

    expected_charge_params = torch.tensor(
        [*mol.partial_charges.m_as("e"), 0.0, 0.53, 0.53, -0.1]
    ).reshape(-1, 1)
    assert torch.allclose(charge_pot.parameters, expected_charge_params)

    param_map = top.parameters["Electrostatics"]

    found_exclusions = sorted((i, j) for i, j in param_map.exclusions.tolist())
    expected_exclusions = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert found_exclusions == expected_exclusions

    expected_assignment = torch.tensor(
        [
            [1, 0, 0, +1, +0, +0, +2],
            [0, 1, 0, +0, +1, +0, -1],
            [0, 0, 1, +0, +0, +1, -1],
            [0, 0, 0, -1, -1, -1, +0],
        ],
        dtype=torch.float64,
    )
    found_assignment = param_map.assignment_matrix.to_dense()

    assert found_assignment.shape == expected_assignment.shape
    assert torch.allclose(found_assignment, expected_assignment)

    found_charges = param_map.assignment_matrix @ charge_pot.parameters
    assert torch.allclose(found_charges.flatten(), torch.tensor(expected_charges))


def test_convert_vdw(ethanol, ethanol_interchange):
    vdw_collection = ethanol_interchange.collections["vdW"]

    potential, parameter_maps = convert_vdw(
        [vdw_collection], [ethanol.to_topology()], [None]
    )

    assert potential.type == "vdW"
    assert potential.fn == smee.EnergyFn.VDW_LJ


def test_convert_dexp(ethanol, test_data_dir):
    ff = openff.toolkit.ForceField(
        str(test_data_dir / "de-ff.offxml"), load_plugins=True
    )

    interchange = openff.interchange.Interchange.from_smirnoff(
        ff, ethanol.to_topology()
    )
    vdw_collection = interchange.collections["DoubleExponential"]

    potential, parameter_maps = convert_dexp(
        [vdw_collection], [ethanol.to_topology()], [None]
    )

    assert potential.attribute_cols[-2:] == ("alpha", "beta")
    assert potential.parameter_cols == ("epsilon", "r_min")

    assert potential.type == "vdW"
    assert potential.fn == smee.EnergyFn.VDW_DEXP


def test_convert_dampedexp6810(ethanol, test_data_dir):
    ff = openff.toolkit.ForceField(
        str(test_data_dir / "PHAST-H2CNO-nonpolar-2.0.0.offxml"), load_plugins=True
    )

    interchange = openff.interchange.Interchange.from_smirnoff(
        ff, ethanol.to_topology()
    )
    vdw_collection = interchange.collections["DampedExp6810"]

    potential, parameter_maps = convert_dampedexp6810(
        [vdw_collection], [ethanol.to_topology()], [None]
    )

    assert potential.attribute_cols[-1] == "force_at_zero"
    assert potential.parameter_cols == ("rho", "beta", "c6", "c8", "c10")

    assert potential.type == "vdW"
    assert potential.fn == smee.EnergyFn.VDW_DAMPEDEXP6810
