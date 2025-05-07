from typing import List, Union
from constants.covalent_radii import CovalentRadii

from numpy import array, ndarray


class Atom:
    """
    Class representing an atom with coordinates, charge, and index.
    """

    def __init__(self, element: str, coord: Union[List[float], ndarray], partial_charge: float, index: int):
        """
        :param element: The name of the element.
        :param coord: The Cartesian coordinates of the atom.
        :param partial_charge: The partial charge of the atom.
        :param index: The index of the atom in the Cartesian coordinates file.
        """

        self.element = element
        self.coord = array(coord, dtype=float)
        self.charge = partial_charge
        self.index = index  # Atom's number in the Cartesian coordinate (.xyz) file

        self.covalent_radius = CovalentRadii[self.element.capitalize()].value[0]

    def __repr__(self) -> str:
        """
        :return: Returns the string representation of the Atom class.
        """

        return f"{self.coord.tolist()}, {self.charge}, {self.index}"
