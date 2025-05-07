from abc import ABC, abstractmethod

from complexes.guest import Guest
from constants.relative_permittivity import RelativePermittivity
from molecular_structure.atom import Atom


class ComplexGuestAnion(ABC):
    """
    An abstract base class that represents host-guest complex with an anion as a guest.
    """

    def __init__(self, atoms: list[Atom], guest: Guest, solvent: str, homo_energy: float | None = None):
        """
        :param atoms: A list of Atom objects that represent the structure of the complex.
        :param guest: A class that represents the guest ion or molecule.
        :param solvent: The name of the solvent used (e.g., water or chloroform).
        """

        self.atoms = atoms
        self.guest = guest
        self.solvent = RelativePermittivity[solvent.upper()].value
        self.homo_energy = homo_energy

    @abstractmethod
    def get_dipole_interactions(self):
        pass

    @abstractmethod
    def get_non_polar_dipole_interactions(self):
        pass

    @abstractmethod
    def get_freely_rotating_dipole_interactions(self):
        pass

    @abstractmethod
    def get_freely_rotating_dipoles_interactions(self):
        pass

    @abstractmethod
    def get_london_dispersion_force(self):
        pass
