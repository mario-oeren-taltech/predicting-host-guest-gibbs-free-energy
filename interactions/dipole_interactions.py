from numpy import cos, linalg, pi, sin

from constants.physical_constants import PhysicalConstants
from interactions.dipole_moment import DipoleMoment
from interactions.helper_functions import get_polarizability, get_electronic_absorption_frequency
from molecular_structure.atom import Atom
from molecular_structure.spatial_analysis import get_angle, get_dihedral_angle


def get_charge_dipole_interaction(charge_atom: Atom, dipole_moment: DipoleMoment,
                                  relative_permittivity: float) -> float:
    """
    The function calculates the interaction energy between a charge from an individual atom and a dipole formed by
    two additional atoms in the same system. The equation is based on equation 4.5 from Intermolecular and Surface
    Forces by Jacob N. Israelachvili (Third Edition).

   The relative permittivity is included in the equation to account for the influence of solvents.

    :param charge_atom: The atom carrying the charge in the charge-dipole interaction (an Atom object).
    :param dipole_moment: The dipole moment in the charge-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :return: The interaction energy (in Hartrees) between the charge from an individual atom and the dipole.
    """

    # Calculate the vector from the charge to the center of the dipole.
    charge_to_dipole_vector = dipole_moment.center - charge_atom.coord

    # Calculate the angle between the dipole vector and the charge to dipole vector.
    charge_dipole_angle = get_angle(dipole_moment.get_vector(), charge_to_dipole_vector)

    # Return the interaction energy.
    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value

    return ((-1.0 * charge_atom.charge * dipole_moment.magnitude * cos(charge_dipole_angle)) /
            (4.0 * pi * permittivity_of_free_space * relative_permittivity * linalg.norm(charge_to_dipole_vector) ** 2))


def get_charge_non_polar_dipole_interaction(charge_atom: Atom, dipole_moment: DipoleMoment,
                                            relative_permittivity: float) -> float:
    """
    The function calculates the interaction between a charge and a non-polar dipole. The equation is based
    on equation in Table 2.2 from Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

   The relative permittivity is included in the equation to account for the influence of solvents.

    :param charge_atom: The atom carrying the charge in the charge-dipole interaction (an Atom object).
    :param dipole_moment: The dipole moment in the charge-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :return: The interaction energy (in Hartrees) between a charge and a non-polar dipole.
    """

    # Calculate the polarizability.
    polarizability = get_polarizability(dipole_moment)

    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value

    return ((-1.0 * (charge_atom.charge ** 2) * polarizability) /
            (2.0 * (4.0 * pi * permittivity_of_free_space * relative_permittivity) ** 2 *
             linalg.norm(dipole_moment.get_center() - charge_atom.coord) ** 4))


def get_charge_freely_rotating_dipole_interaction(charge_atom: Atom, dipole_moment: DipoleMoment,
                                                  relative_permittivity: float, temperature: float = 298.0) -> float:
    """
    The function calculates the interaction energy between a charge from an individual atom and a freely rotating
    dipole formed by two additional atoms in the same system. The equation is based on equation 4.16 from
    Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

    The relative permittivity is included in the equation to account for the influence of solvents.

    :param charge_atom: The atom carrying the charge in the charge-dipole interaction (an Atom object).
    :param dipole_moment: The dipole moment in the charge-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :param temperature: The temperature of the experiment.
    :return: The interaction energy (in Hartrees) between the charge from an individual atom and the dipole.
    """

    # Calculate the vector from the charge to the center of the dipole.
    charge_to_dipole_vector = dipole_moment.center - charge_atom.coord

    # Return the interaction energy.
    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value
    boltzmann_constant = PhysicalConstants.BOLTZMANN.value

    return ((-1.0 * (charge_atom.charge ** 2) * (dipole_moment.magnitude ** 2)) /
            (+6.0 * ((+4.0 * pi * permittivity_of_free_space * relative_permittivity) ** 2) *
            boltzmann_constant * temperature * (linalg.norm(charge_to_dipole_vector) ** 4)))


def get_dipole_dipole_interaction(dipole_moment_a: DipoleMoment, dipole_moment_b: DipoleMoment,
                                  relative_permittivity: float) -> float:
    """
    The function calculates the interaction energy between two dipoles. The equation is based on equation 4.9 from
    Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

   The relative permittivity is included in the equation to account for the influence of solvents.

    :param dipole_moment_a: The first dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param dipole_moment_b: The second dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :return: The interaction energy (in Hartrees) between the two dipoles.
    """

    # Calculate the angle between the first dipole moment, and the center of the second dipole moment.
    dipole_to_dipole_vector = dipole_moment_a.center - dipole_moment_b.center
    dipole_dipole_angle_a = get_angle(dipole_moment_a.get_vector(), dipole_to_dipole_vector)

    # Calculate the angle between the second dipole moment, and the center of the first dipole moment.
    dipole_to_dipole_vector = dipole_moment_b.center - dipole_moment_a.center
    dipole_dipole_angle_b = get_angle(dipole_moment_b.get_vector(), dipole_to_dipole_vector)

    # Calculate the dihedral angle between the dipole-dipole angles.
    vector_from_dipole_to_dipole = dipole_moment_b.atom_a.coord - dipole_moment_a.atom_a.coord
    dihedral_angle = get_dihedral_angle(dipole_moment_a.vector * -1.0, vector_from_dipole_to_dipole,
                                        dipole_moment_b.vector)

    # Return the interaction energy.
    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value
    dipole_to_dipole_vector_length = linalg.norm(dipole_to_dipole_vector)

    return (((-1.0 * dipole_moment_a.magnitude * dipole_moment_b.magnitude) /
             (4.0 * pi * permittivity_of_free_space * relative_permittivity * dipole_to_dipole_vector_length ** 3)) *
            ((2.0 * cos(dipole_dipole_angle_a) * cos(dipole_dipole_angle_b)) -
             (sin(dipole_dipole_angle_a) * sin(dipole_dipole_angle_b) * cos(dihedral_angle))))


def get_dipole_non_polar_molecule_interaction(dipole_moment_a: DipoleMoment, dipole_moment_b: DipoleMoment,
                                              relative_permittivity: float) -> float:
    """
    The function calculates the non-polar interaction energy between two freely rotating dipoles. The equation is based
    on equation 5.22 from Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

   The relative permittivity is included in the equation to account for the influence of solvents.

    :param dipole_moment_a: The first dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param dipole_moment_b: The second dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :return: The non-polar interaction energy (in Hartrees) between the dipole and the neutral molecule.
    """

    # Calculate the polarizability.
    polarizability_b = get_polarizability(dipole_moment_b)

    # Calculate the angle between the second dipole moment, and the center of the first dipole moment.
    dipole_to_dipole_vector = dipole_moment_a.center - dipole_moment_b.center
    dipole_dipole_angle_a = get_angle(dipole_moment_a.get_vector(), dipole_to_dipole_vector)

    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value

    return ((-1.0 * (dipole_moment_a.get_magnitude() ** 2) * polarizability_b *
             (+1.0 + +3.0 * (cos(dipole_dipole_angle_a) ** 2))) /
            (+2.0 * (+4.0 * pi * permittivity_of_free_space * relative_permittivity) ** 2 *
             (linalg.norm(dipole_moment_a.center - dipole_moment_b.center) ** 6)))


def get_freely_rotating_dipole_dipole_interaction(dipole_moment_a: DipoleMoment, dipole_moment_b: DipoleMoment,
                                                  relative_permittivity: float, temperature: float = 298.0) -> float:
    """
    The function calculates the interaction energy between two freely rotating dipoles. The equation is based on
    equation 4.17 from Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

   The relative permittivity is included in the equation to account for the influence of solvents.

    :param dipole_moment_a: The first dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param dipole_moment_b: The second dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :param temperature: The temperature of the experiment.
    :return: The interaction energy (in Hartrees) between the two freely rotating dipoles.
    """

    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value
    boltzmann_constant = PhysicalConstants.BOLTZMANN.value

    return (-1.0 * ((dipole_moment_a.get_magnitude() ** 2) * (dipole_moment_b.get_magnitude() ** 2)) /
            (+3.0 * ((4.0 * pi * permittivity_of_free_space * relative_permittivity) ** 2) * boltzmann_constant *
             temperature * (linalg.norm(dipole_moment_a.center - dipole_moment_b.center) ** 6)))


def get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a: DipoleMoment,
                                                            dipole_moment_b: DipoleMoment,
                                                            relative_permittivity: float) -> float:
    """
    The function calculates the non-polar interaction energy between two freely rotating dipoles. The equation is based
    on equation 5.23 from Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

   The relative permittivity is included in the equation to account for the influence of solvents.

    :param dipole_moment_a: The first dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param dipole_moment_b: The second dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :return: The non-polar interaction energy (in Hartrees) between the two freely rotating dipoles.
    """

    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value
    polarizability_b = get_polarizability(dipole_moment_b)

    return ((-1.0 * ((dipole_moment_a.get_magnitude() ** 2) * polarizability_b)) /
            (((+4.0 * pi * permittivity_of_free_space * relative_permittivity) ** 2) *
            (linalg.norm(dipole_moment_a.center - dipole_moment_b.center) ** 6)))


def get_london_dispersion_force(dipole_moment_a: DipoleMoment, dipole_moment_b: DipoleMoment, homo_energy,
                                relative_permittivity: float) -> float:
    """
    The function calculates the London dispersion force between two non-polar molecules. The equation is based
    on equation in Table 2.2 from Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

    The relative permittivity is included in the equation to account for the influence of solvents.

    :param dipole_moment_a: The first dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param dipole_moment_b: The second dipole moment in the dipole-dipole interaction (a DipoleMoment object).
    :param homo_energy: The energy of the highest occupied molecular orbital (HOMO) in Hartrees.
    :param relative_permittivity: The relative permittivity (dielectric constant) of the medium.
    :return: The London dispersion force (in Hartrees) between the two dipoles.
    """

    planck_constant = PhysicalConstants.PLANCK.value
    permittivity_of_free_space = PhysicalConstants.VACUUM_PERMITTIVITY.value

    # Calculate the electronic absorption (ionization) frequency.
    electronic_absorption_frequency = get_electronic_absorption_frequency(homo_energy)

    # Calculate the polarizability values.
    polarizability_a = get_polarizability(dipole_moment_a)
    polarizability_b = get_polarizability(dipole_moment_b)

    return ((-0.75 * (planck_constant * electronic_absorption_frequency * polarizability_a * polarizability_b)) /
            (((+4.0 * pi * permittivity_of_free_space * relative_permittivity) ** 2) *
             (linalg.norm(dipole_moment_a.center - dipole_moment_b.center) ** 6)))
