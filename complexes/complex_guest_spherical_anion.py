from complexes.complex_guest_anion import ComplexGuestAnion
from complexes.helper_functions import get_host_atoms, get_dipole_moments
from interactions.dipole_interactions import (get_charge_dipole_interaction, get_charge_non_polar_dipole_interaction,
                                              get_charge_freely_rotating_dipole_interaction)


class ComplexGuestSphericalAnion(ComplexGuestAnion):
    """
    Represents a host-guest complex with a spherical anion as the guest.
    """

    def get_dipole_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the anion and all fixed dipole moments within the host structure.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        for dipole_moment in get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius)):
            interaction_energy += get_charge_dipole_interaction(self.atoms[self.guest.central_atom], dipole_moment,
                                                                self.solvent)

        return interaction_energy

    def get_non_polar_dipole_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the anion and all non-polar dipole moments within the host structure.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        return 0.0

    def get_freely_rotating_dipole_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the anion and all freely rotating dipole moments within the host structure.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        for dipole_moment in get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius)):
            interaction_energy += get_charge_freely_rotating_dipole_interaction(self.atoms[self.guest.central_atom],
                                                                                dipole_moment, self.solvent)

        return interaction_energy

    def get_freely_rotating_dipoles_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        Spherical anion is not a freely rotating dipole!

        :return: The value returned is zero.
        """

        return 0.0

    def get_london_dispersion_force(self, interaction_radius: float = 50.0) -> float:
        """
        Spherical anion is not a dipole!

        :return: The value returned is zero.
        """

        return 0.0
