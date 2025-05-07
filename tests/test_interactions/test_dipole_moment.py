from unittest import TestCase

from numpy import round, array, array_equal

from interactions.dipole_moment import DipoleMoment
from molecular_structure.atom import Atom
from molecular_structure.molecular_structure import make_list_of_atoms
from tests.helper_functions import build_path


class TestDipoleMoment(TestCase):

    def setUp(self):

        # Set up the list of Atom objects.
        self.atoms = make_list_of_atoms(
            build_path('anion_spherical_geometry.xyz'), build_path('anion_spherical_charges')
        )

    def test_dipole_moment(self):
        """
        Test the calculation of dipole moments.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), -1.0, 0)
        atom_b = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 1)

        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_vector(), array([-1.0, +0.0, 1.0])))
        self.assertTrue(array_equal(DipoleMoment(atom_b, atom_a).get_vector(), array([-1.0, +0.0, 1.0])))
        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_vector(),
                                    DipoleMoment(atom_b, atom_a).get_vector()))

        self.assertEqual(DipoleMoment(atom_a, atom_b).get_magnitude(), 2.8284271247461903)
        self.assertEqual(DipoleMoment(atom_b, atom_a).get_magnitude(), 2.8284271247461903)
        self.assertEqual(DipoleMoment(atom_a, atom_b).get_magnitude(), DipoleMoment(atom_b, atom_a).get_magnitude())

        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_center(), array([+0.5, +0.0, +0.5])))
        self.assertTrue(array_equal(DipoleMoment(atom_b, atom_a).get_center(), array([+0.5, +0.0, +0.5])))
        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_center(),
                                    DipoleMoment(atom_b, atom_a).get_center()))

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), -0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), +4.567, 1)

        result = DipoleMoment(atom_a, atom_b).get_vector()
        self.assertTrue(array_equal(round(result, 6), array([+6.6, +6.6, -4.4])))

        result = DipoleMoment(atom_b, atom_a).get_vector()
        self.assertTrue(array_equal(round(result, 6), array([+6.6, +6.6, -4.4])))

        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_vector(),
                                    DipoleMoment(atom_b, atom_a).get_vector()))

        self.assertEqual(round(DipoleMoment(atom_a, atom_b).get_magnitude(), 6), 48.39571)
        self.assertEqual(round(DipoleMoment(atom_b, atom_a).get_magnitude(), 6), 48.39571)
        self.assertEqual(DipoleMoment(atom_a, atom_b).get_magnitude(), DipoleMoment(atom_b, atom_a).get_magnitude())

        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_center(), array([+3.4, +5.6, +2.3])))
        self.assertTrue(array_equal(DipoleMoment(atom_b, atom_a).get_center(), array([+3.4, +5.6, +2.3])))
        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_center(),
                                    DipoleMoment(atom_b, atom_a).get_center()))

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[194]

        result = DipoleMoment(atom_a, atom_b).get_vector()
        self.assertTrue(array_equal(round(result, 6), array([+2.475093, +1.141934, -0.688363])))

        result = DipoleMoment(atom_b, atom_a).get_vector()
        self.assertTrue(array_equal(round(result, 6), array([+2.475093, +1.141934, -0.688363])))

        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_vector(),
                                    DipoleMoment(atom_b, atom_a).get_vector()))

        self.assertEqual(round(DipoleMoment(atom_a, atom_b).get_magnitude(), 6), 2.69569)
        self.assertEqual(round(DipoleMoment(atom_b, atom_a).get_magnitude(), 6), 2.69569)
        self.assertEqual(DipoleMoment(atom_a, atom_b).get_magnitude(), DipoleMoment(atom_b, atom_a).get_magnitude())

        result = DipoleMoment(atom_a, atom_b).get_center()
        self.assertTrue(array_equal(round(result, 6), array([+1.535561, +1.518747, +0.190265])))

        result = DipoleMoment(atom_b, atom_a).get_center()
        self.assertTrue(array_equal(round(result, 6), array([+1.535561, +1.518747, +0.190265])))

        self.assertTrue(array_equal(DipoleMoment(atom_a, atom_b).get_center(),
                                    DipoleMoment(atom_b, atom_a).get_center()))

        # Test exceptions.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([1.0, 0.0, 0.0]), -1.0, 1)

        with self.assertRaises(RuntimeError):
            DipoleMoment(atom_a, atom_b)

        with self.assertRaises(RuntimeError):
            DipoleMoment(atom_a, atom_a)
