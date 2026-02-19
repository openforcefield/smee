"""Constants used throughout the package."""

import enum

if hasattr(enum, "StrEnum"):
    _StrEnum = enum.StrEnum
else:
    import typing

    _S = typing.TypeVar("_S", bound="_StrEnum")

    class _StrEnum(str, enum.Enum):
        """TODO: remove when python 3.10 support is dropped."""

        def __new__(cls: typing.Type[_S], *values: str) -> _S:
            value = str(*values)

            member = str.__new__(cls, value)
            member._value_ = value

            return member

        __str__ = str.__str__


class PotentialType(_StrEnum):
    """An enumeration of the potential types supported by ``smee`` out of the box."""

    BONDS = "Bonds"
    ANGLES = "Angles"

    PROPER_TORSIONS = "ProperTorsions"
    IMPROPER_TORSIONS = "ImproperTorsions"

    LINEAR_BONDS = "LinearBonds"
    LINEAR_ANGLES = "LinearAngles"

    VDW = "vdW"
    ELECTROSTATICS = "Electrostatics"


class EnergyFn(_StrEnum):
    """An enumeration of the energy functions supported by ``smee`` out of the box."""

    COULOMB = "coul"
    POLARIZATION = "coul+pol"

    VDW_LJ = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    VDW_DEXP = (
        "epsilon*("
        "beta/(alpha-beta)*exp(alpha*(1-r/r_min))-"
        "alpha/(alpha-beta)*exp(beta*(1-r/r_min)))"
    )
    # VDW_BUCKINGHAM = "a*exp(-b*r)-c*r^-6"
    VDW_DAMPEDEXP6810 = "force_at_zero*beta**-1*exp(-beta*(r-rho))-f_6(beta*r)*c6**6-f_8(beta*r)*c8**8-f_10(beta*r)*c10**10"

    BOND_HARMONIC = "k/2*(r-length)**2"

    ANGLE_HARMONIC = "k/2*(theta-angle)**2"

    TORSION_COSINE = "k*(1+cos(periodicity*theta-phase))"

    BOND_LINEAR = "(k1+k2)/2*(r-(k1*length1+k2*length2)/(k1+k2))**2"

    ANGLE_LINEAR = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"


CUTOFF_ATTRIBUTE = "cutoff"
"""The attribute that should be used to store the cutoff distance of a potential."""
SWITCH_ATTRIBUTE = "switch_width"
"""The attribute that should be used to store the switch width of a potential, if the
potential should use the standard OpenMM switch function.

This attribute should be omitted if the potential should not use a switch function.
"""
