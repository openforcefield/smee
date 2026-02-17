"""Configuration from MM simulations."""

import openmm.unit
import pydantic
from pydantic_units import OpenMMQuantity, quantity_serializer

_KCAL_PER_MOL = openmm.unit.kilocalories_per_mole
_ANGSTROM = openmm.unit.angstrom
_GRAMS_PER_ML = openmm.unit.grams / openmm.unit.milliliters


if pydantic.__version__.startswith("1."):

    class BaseModel(pydantic.BaseModel):
        class Config:
            json_encoders = {openmm.unit.Quantity: quantity_serializer}

else:
    BaseModel = pydantic.BaseModel


class GenerateCoordsConfig(BaseModel):
    """Configure how coordinates should be generated for a system using PACKMOL."""

    target_density: OpenMMQuantity[_GRAMS_PER_ML] = pydantic.Field(
        0.95 * _GRAMS_PER_ML,
        description="Target mass density for final system with units compatible with "
        "g / mL.",
    )

    scale_factor: float = pydantic.Field(
        1.1,
        description="The amount to scale the approximate box size by to help alleviate "
        "issues with packing larger molecules.",
    )
    padding: OpenMMQuantity[openmm.unit.angstrom] = pydantic.Field(
        2.0 * openmm.unit.angstrom,
        description="The amount of padding to add to the final box size to help "
        "alleviate PBC issues.",
    )

    tolerance: OpenMMQuantity[openmm.unit.angstrom] = pydantic.Field(
        2.0 * openmm.unit.angstrom,
        description="The minimum spacing between molecules during packing.",
    )

    seed: int | None = pydantic.Field(
        None, description="The random seed to use when generating the coordinates."
    )


class MinimizationConfig(BaseModel):
    """Configure how a system should be energy minimized."""

    tolerance: OpenMMQuantity[_KCAL_PER_MOL / _ANGSTROM] = pydantic.Field(
        10.0 * _KCAL_PER_MOL / _ANGSTROM,
        description="Minimization will be halted once the root-mean-square value of "
        "all force components reaches this tolerance.",
    )
    max_iterations: int = pydantic.Field(
        0,
        description="The maximum number of iterations to perform. If 0, minimization "
        "will continue until the tolerance is met.",
    )


class SimulationConfig(BaseModel):
    temperature: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        ...,
        description="The temperature to simulate at.",
    )
    pressure: OpenMMQuantity[openmm.unit.atmospheres] | None = pydantic.Field(
        ...,
        description="The pressure to simulate at, or none to run in NVT.",
    )

    n_steps: int = pydantic.Field(
        ..., description="The number of steps to simulate for."
    )

    timestep: OpenMMQuantity[openmm.unit.femtoseconds] = pydantic.Field(
        2.0 * openmm.unit.femtoseconds,
        description="The timestep to use during the simulation.",
    )
    friction_coeff: OpenMMQuantity[1.0 / openmm.unit.picoseconds] = pydantic.Field(
        1.0 / openmm.unit.picoseconds,
        description="The integrator friction coefficient.",
    )
