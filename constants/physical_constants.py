from enum import Enum

from numpy import pi


class PhysicalConstants(Enum):
    """
    All given physical constants are in atomic units.
    """

    VACUUM_PERMITTIVITY = 1.439964547  # Unit for the vacuum permittivity: (elementary charge ^ 2 * Hartree) / Angstrom
    BOLTZMANN = 3.167 * (10 ** -6)  # Unit for the Boltzmann constant: Hartree / K
    PLANCK = 2 * pi

    AU_TO_EV = 27.2113957  # Conversion constant
