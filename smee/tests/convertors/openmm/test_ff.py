import numpy
import openff.interchange
import openff.toolkit
import openmm.app
import openmm.unit
import pytest

import smee
import smee.converters
import smee.converters.openmm


def compute_energy(system: openmm.System, coords: numpy.ndarray) -> float:
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.01))
    ctx.setPositions(coords * openmm.unit.angstrom)

    state = ctx.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)


def compute_v_site_coords(
    system: openmm.System, coords: numpy.ndarray
) -> numpy.ndarray:
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.01))
    ctx.setPositions(coords * openmm.unit.angstrom)
    ctx.computeVirtualSites()

    state = ctx.getState(getPositions=True)
    return state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)


@pytest.mark.parametrize("with_constraints", [True, False])
@pytest.mark.parametrize("smiles", ["CO", "C=O", "Oc1ccccc1"])
def test_convert_to_openmm_ffxml(tmp_cwd, with_constraints, smiles):
    off_ff = openff.toolkit.ForceField(
        "openff-2.0.0.offxml"
        if with_constraints
        else "openff_unconstrained-2.0.0.offxml"
    )

    off_mol = openff.toolkit.Molecule.from_smiles(smiles)
    off_top = off_mol.to_topology()

    interchange = openff.interchange.Interchange.from_smirnoff(off_ff, off_top)
    ff, [top] = smee.converters.convert_interchange(interchange)

    [ffxml] = smee.converters.openmm._ff.convert_to_openmm_ffxml(ff, top)

    ffxml_path = tmp_cwd / "ff.xml"
    ffxml_path.write_text(ffxml)

    omm_ff = openmm.app.ForceField(str(ffxml_path))

    system_from_xml = omm_ff.createSystem(
        smee.converters.convert_to_openmm_topology(top),
        nonbondedCutoff=9.0 * openmm.unit.angstrom,
        switchDistance=8.0 * openmm.unit.angstrom,
        constraints=openmm.app.HBonds if with_constraints else None,
        rigidWater=True,
        removeCMMotion=True,
    )
    system_from_off = off_ff.create_openmm_system(off_top)

    assert system_from_xml.getNumParticles() == system_from_off.getNumParticles()
    assert system_from_xml.getNumForces() == system_from_off.getNumForces()

    assert system_from_xml.getNumConstraints() == system_from_off.getNumConstraints()

    constraints_from_off = {}

    for i in range(system_from_off.getNumConstraints()):
        idx_a, idx_b, dist = system_from_off.getConstraintParameters(i)
        constraints_from_off[idx_a, idx_b] = dist.value_in_unit(openmm.unit.nanometer)

    constraints_from_xml = {}

    for i in range(system_from_xml.getNumConstraints()):
        idx_a, idx_b, dist = system_from_xml.getConstraintParameters(i)
        constraints_from_xml[idx_a, idx_b] = dist.value_in_unit(openmm.unit.nanometer)

    assert constraints_from_xml == pytest.approx(constraints_from_off)

    off_mol.generate_conformers(n_conformers=1)
    coords = off_mol.conformers[0].m_as("angstrom")

    for _ in range(5):
        coords_rand = coords + numpy.random.randn(*coords.shape) * 0.1

        energy_off = compute_energy(system_from_off, coords_rand)
        energy_xml = compute_energy(system_from_xml, coords_rand)

        assert energy_off == pytest.approx(energy_xml, abs=1.0e-3)


def test_convert_to_openmm_ffxml_v_sites(tmp_cwd):
    off_ff = openff.toolkit.ForceField("tip5p.offxml")

    off_mol = openff.toolkit.Molecule.from_smiles("O")
    off_top = off_mol.to_topology()

    interchange = openff.interchange.Interchange.from_smirnoff(off_ff, off_top)
    ff, [top] = smee.converters.convert_interchange(interchange)

    [ffxml] = smee.converters.openmm._ff.convert_to_openmm_ffxml(ff, top)

    ffxml_path = tmp_cwd / "ff.xml"
    ffxml_path.write_text(ffxml)

    omm_ff = openmm.app.ForceField(str(ffxml_path))

    system_from_xml = omm_ff.createSystem(
        smee.converters.convert_to_openmm_topology(top),
        nonbondedCutoff=9.0 * openmm.unit.angstrom,
        switchDistance=8.0 * openmm.unit.angstrom,
        constraints=openmm.app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    system_from_off = off_ff.create_openmm_system(off_top)

    assert system_from_xml.getNumParticles() == system_from_off.getNumParticles()

    off_mol.generate_conformers(n_conformers=1)

    coords = off_mol.conformers[0].m_as("angstrom")
    coords = numpy.vstack([coords, numpy.zeros((2, 3))])

    coords_from_xml = compute_v_site_coords(system_from_xml, coords)
    coords_from_off = compute_v_site_coords(system_from_off, coords)

    assert coords_from_xml.shape == coords_from_off.shape
    assert numpy.allclose(coords_from_xml, coords_from_off, atol=1.0e-3)

    params_from_xml = {}
    [nb_force_from_xml] = [
        force
        for force in system_from_xml.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]

    for i in range(nb_force_from_xml.getNumParticles()):
        charge, sigma, epsilon = nb_force_from_xml.getParticleParameters(i)
        params_from_xml[i] = (
            charge.value_in_unit(openmm.unit.elementary_charge),
            sigma.value_in_unit(openmm.unit.angstrom),
            epsilon.value_in_unit(openmm.unit.kilojoule_per_mole),
        )

    params_from_off = {}
    [nb_force_from_off] = [
        force
        for force in system_from_off.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]

    for i in range(nb_force_from_off.getNumParticles()):
        charge, sigma, epsilon = nb_force_from_off.getParticleParameters(i)
        params_from_off[i] = (
            charge.value_in_unit(openmm.unit.elementary_charge),
            sigma.value_in_unit(openmm.unit.angstrom),
            epsilon.value_in_unit(openmm.unit.kilojoule_per_mole),
        )

    assert len(params_from_xml) == len(params_from_off)

    for i in range(nb_force_from_off.getNumParticles()):
        assert params_from_xml[i][0] == pytest.approx(params_from_off[i][0])
        assert params_from_xml[i][1] == pytest.approx(params_from_off[i][1])
        assert params_from_xml[i][2] == pytest.approx(params_from_off[i][2])
