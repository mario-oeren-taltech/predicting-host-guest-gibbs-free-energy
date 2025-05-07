from numpy import equal, linalg, ndarray

from molecular_structure.atom import Atom


class DipoleMoment:
    """
    The DipoleMoment class represents the dipole moment between two atoms in a molecular system.

    The DipoleMoment class stores both the dipole vector (which indicates the direction and magnitude of the dipole)
    and the magnitude of the dipole moment (a scalar value). In addition, it stores the center point of the dipole.
    """

    def __init__(self, atom_a: Atom, atom_b: Atom):
        """
        The equations for calculating the dipole moment vector and magnitude are based on equation 4.1 from
        Intermolecular and Surface Forces by Jacob N. Israelachvili (Third Edition).

        The dipole vector must point from the smaller (more negative) charge to the larger (more positive) charge. The
        equation 4.1 is expanded to take into account charges of different magnitude.

        The atom charges are in elementary units (e), the coordinates are in Ångströms (Å), and the dipole magnitude is
        in e * Å.

        :param atom_a: The first atom contributing to the dipole moment (an Atom object).
        :param atom_b: The second atom contributing to the dipole moment (an Atom object).
        :raises RuntimeError: The given atoms are superimposed.
        """

        # Raise an error if atom positions are superimposed or the same atom is used twice.
        if equal(atom_a.coord, atom_b.coord).all():
            raise RuntimeError('the given atoms are superimposed.')

        # Calculate the dipole vector (points from the negative charge to the positive charge).
        if atom_a.charge >= atom_b.charge:

            # Store the atoms.
            self.atom_a = atom_b
            self.atom_b = atom_a

            self.vector = atom_a.coord - atom_b.coord
        else:

            # Store the atoms.
            self.atom_a = atom_a
            self.atom_b = atom_b

            self.vector = atom_b.coord - atom_a.coord
        
        # Calculate the magnitude of the dipole moment.
        self.magnitude = linalg.norm(self.vector) * abs(atom_a.charge - atom_b.charge)

        # Calculate the center of the dipole.
        self.center = (atom_a.coord + atom_b.coord) / 2

    def get_vector(self) -> ndarray:
        """
        :return: The dipole vector (coordinates are in Å).
        """

        return self.vector

    def get_magnitude(self) -> float:
        """
        :return: The dipole magnitude (in e * Å)
        """

        return self.magnitude

    def get_center(self) -> ndarray:
        """
        :return: The center of the dipole (coordinates are in Å).
        """

        return self.center
