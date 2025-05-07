from unittest import TestCase

from numpy import array, round

from interactions.dipole_moment import DipoleMoment
from interactions.helper_functions import get_polarizability, get_electronic_absorption_frequency
from molecular_structure.atom import Atom
from molecular_structure.molecular_structure import make_list_of_atoms
from tests.helper_functions import build_path


class TestHelperFunctions(TestCase):

    def setUp(self):

        # Set up the list of Atom objects.
        self.atoms = make_list_of_atoms(
            build_path('anion_spherical_geometry.xyz'), build_path('anion_spherical_charges')
        )

    def test_get_polarizability(self):
        """
        Test the calculation of polarizability.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -1.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        self.assertEqual(round(get_polarizability(dipole_moment_a), 9), 51.180751339)
        self.assertEqual(round(get_polarizability(dipole_moment_b), 9), 94.025044075)

        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -3.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        self.assertEqual(round(get_polarizability(dipole_moment_a), 9), +51.180751339)
        self.assertEqual(round(get_polarizability(dipole_moment_b), 9), +94.025044075)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)
        atom_d = Atom('H', array([8.9, 0.1, 2.3]), -2.345, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        self.assertEqual(round(get_polarizability(dipole_moment_a), 9), +19882.167456322)
        self.assertEqual(round(get_polarizability(dipole_moment_b), 9), +13505.264309828)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[194]
        atom_c = self.atoms[20]
        atom_d = self.atoms[84]

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        self.assertEqual(round(get_polarizability(dipole_moment_a), 9), +402.093643455)
        self.assertEqual(round(get_polarizability(dipole_moment_b), 9), +55.1673314650)

    def test_get_electronic_absorption_frequency(self):
        """
        Test the calculation of electronic absorption frequency.
        """

        homo_energy = 0.0001

        self.assertEqual(round(get_electronic_absorption_frequency(homo_energy), 9), +0.000007958)
