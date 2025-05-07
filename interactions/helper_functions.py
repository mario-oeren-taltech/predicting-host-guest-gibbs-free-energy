from numpy import linalg, pi

from constants.physical_constants import PhysicalConstants
from interactions.dipole_moment import DipoleMoment


def get_polarizability(dipole_moment: DipoleMoment) -> float:
    """
    The function calculates the polarizability (denoted as alpha) of a bond between two atoms. The equation is based
    on equation 5.5 from Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

    :param dipole_moment: Dipole moment (a DipoleMoment object).
    :return: The polarizability in (e ** 2) * Hartree * (Ångström ** 2).
    """

    return +4.0 * pi * PhysicalConstants.VACUUM_PERMITTIVITY.value * (linalg.norm(dipole_moment.get_vector()) ** 3)


def get_electronic_absorption_frequency(homo_energy: float) -> float:
    """
    The function calculates the electronic absorption frequency.

    :param homo_energy: The energy of the highest occupied molecular orbital (HOMO) in Hartrees.
    :return: The electronic absorption energy in atomic units of frequency.
    """

    # The elementary charge in the atomic units is 1.0; thus, it can be left out from the equation.
    return homo_energy / (2 * PhysicalConstants.PLANCK.value)
