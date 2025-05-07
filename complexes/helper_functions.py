from itertools import combinations

from complexes.guest import Guest
from interactions.dipole_moment import DipoleMoment
from molecular_structure.atom import Atom
from molecular_structure.spatial_analysis import get_distance


def get_host_atoms(atoms: list[Atom], guest: Guest, interaction_radius: float) -> list[Atom]:
    """
    The function returns the atom indices of the host that are within the given interaction radius cut-off.

    :param atoms: The list of Atom objects that form the host-guest system.
    :param guest: The guest as a Guest object.
    :param interaction_radius: The cut-off radius.
    :return: A list of atom objects of the host.
    """

    host_atoms = []

    for atom in atoms:
        distance = get_distance(atom.coord, atoms[guest.central_atom].coord)

        if atom.index not in guest.atoms and distance <= interaction_radius:
            host_atoms.append(atom)

    return host_atoms


def get_dipole_moments(atoms: list[Atom]) -> list[DipoleMoment]:
    """
    The function iterates over the given atoms and forms all potential dipole moments between atoms.

    :param atoms: Atoms of the given system represented as Atom objects.
    :return: A list of DipoleMoment objects.
    """

    dipole_moments = []

    for atom_a, atom_b in combinations(atoms, 2):
        if get_distance(atom_a.coord, atom_b.coord) <= (atom_a.covalent_radius + atom_b.covalent_radius) * 1.3:
            dipole_moments.append(DipoleMoment(atom_a, atom_b))

    return dipole_moments
