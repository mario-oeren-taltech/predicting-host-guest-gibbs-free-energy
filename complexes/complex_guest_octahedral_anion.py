from complexes.complex_guest_anion import ComplexGuestAnion
from complexes.helper_functions import get_host_atoms, get_dipole_moments
from interactions.dipole_interactions import (
    get_dipole_dipole_interaction, get_freely_rotating_dipole_dipole_interaction, get_london_dispersion_force,
    get_dipole_non_polar_molecule_interaction, get_non_polar_freely_rotating_dipole_dipole_interaction
)


class ComplexGuestOctahedralAnion(ComplexGuestAnion):
    """
    Represents a host-guest complex with an octahedral anion as the guest.
    """

    def get_dipole_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the dipole moments of the guest and the dipole moments of the host.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        host_dipole_moments = get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius))
        guest_dipole_moments = get_dipole_moments([self.atoms[index] for index in self.guest.atoms])

        for host_dipole_moment in host_dipole_moments:
            for guest_dipole_moment in guest_dipole_moments:
                interaction_energy += get_dipole_dipole_interaction(host_dipole_moment, guest_dipole_moment,
                                                                    self.solvent)

        return interaction_energy

    def get_non_polar_dipole_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the non-polar dipole moments of the guest and the dipole moments of the host.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        host_dipole_moments = get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius))
        guest_dipole_moments = get_dipole_moments([self.atoms[index] for index in self.guest.atoms])

        for host_dipole_moment in host_dipole_moments:
            for guest_dipole_moment in guest_dipole_moments:
                interaction_energy += get_dipole_non_polar_molecule_interaction(host_dipole_moment, guest_dipole_moment,
                                                                                self.solvent)

        return interaction_energy

    def get_freely_rotating_dipole_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the freely rotating dipole moments of the guest and the dipole moments of the host.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        host_dipole_moments = get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius))
        guest_dipole_moments = get_dipole_moments([self.atoms[index] for index in self.guest.atoms])

        for host_dipole_moment in host_dipole_moments:
            for guest_dipole_moment in guest_dipole_moments:
                interaction_energy += get_freely_rotating_dipole_dipole_interaction(host_dipole_moment,
                                                                                    guest_dipole_moment, self.solvent)

        return interaction_energy

    def get_freely_rotating_dipoles_interactions(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between the non-polar freely rotating dipole moments of the guest and the dipole moments of the host.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        host_dipole_moments = get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius))
        guest_dipole_moments = get_dipole_moments([self.atoms[index] for index in self.guest.atoms])

        for host_dipole_moment in host_dipole_moments:
            for guest_dipole_moment in guest_dipole_moments:
                interaction_energy += get_non_polar_freely_rotating_dipole_dipole_interaction(host_dipole_moment,
                                                                                              guest_dipole_moment,
                                                                                              self.solvent)

        return interaction_energy

    def get_london_dispersion_force(self, interaction_radius: float = 50.0) -> float:
        """
        The function calculates the interaction energy of the host-guest complex by summing the interaction energy
        values between non-polar dipole moments of the guest and the dipole moments of the host.

        :param interaction_radius: The cut-off radius of the interaction in Angstroms.
        :return: The interaction energy in Hartrees.
        """

        interaction_energy = 0.0

        host_dipole_moments = get_dipole_moments(get_host_atoms(self.atoms, self.guest, interaction_radius))
        guest_dipole_moments = get_dipole_moments([self.atoms[index] for index in self.guest.atoms])

        for host_dipole_moment in host_dipole_moments:
            for guest_dipole_moment in guest_dipole_moments:
                interaction_energy += get_london_dispersion_force(host_dipole_moment, guest_dipole_moment,
                                                                  self.homo_energy, self.solvent)

        return interaction_energy
