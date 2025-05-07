from unittest import TestCase

from numpy import array, array_equal

from molecular_structure.atom import Atom


class TestAtom(TestCase):

    def test_atom(self):
        """
        Tests the Atom class.
        """

        atom = Atom('H', [0.0, 0.0, 0.0], +1.0, 0)
        self.assertTrue(array_equal(atom.coord, array([0.0, 0.0, 0.0])))
        self.assertEqual(atom.charge, +1.0)
        self.assertEqual(atom.index, 0)

        atom = Atom('H', array([0.0, 0.0, 0.0]), -0.7, 3)
        self.assertTrue(array_equal(atom.coord, array([0.0, 0.0, 0.0])))
        self.assertEqual(atom.charge, -0.7)
        self.assertEqual(atom.index, 3)
