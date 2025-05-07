from unittest import TestCase

from numpy import round, array

from constants.relative_permittivity import RelativePermittivity
from interactions.dipole_interactions import (
    get_dipole_dipole_interaction, get_charge_dipole_interaction, get_charge_freely_rotating_dipole_interaction,
    get_freely_rotating_dipole_dipole_interaction, get_non_polar_freely_rotating_dipole_dipole_interaction,
    get_dipole_non_polar_molecule_interaction, get_charge_non_polar_dipole_interaction, get_london_dispersion_force
)
from interactions.dipole_moment import DipoleMoment
from molecular_structure.atom import Atom
from molecular_structure.molecular_structure import make_list_of_atoms
from tests.helper_functions import build_path


class TestDipoleInteractions(TestCase):

    def setUp(self):

        # Set up the list of Atom objects.
        self.atoms = make_list_of_atoms(
            build_path('anion_spherical_geometry.xyz'), build_path('anion_spherical_charges')
        )

    def test_get_charge_dipole_interaction(self):
        """
        Test the calculation of charge-dipole interactions.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +0.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        self.assertEqual(get_charge_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value), 0.0)

        atom_a = Atom('H', array([0.0, 0.0, 0.0]), +0.0, 0)
        atom_b = Atom('H', array([0.0, 2.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        self.assertEqual(get_charge_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value), 0.0)

        atom_a = Atom('H', array([0.0, 0.0, 0.0]), -1.0, 0)
        atom_b = Atom('H', array([0.0, 2.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.002271002)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), +0.000212911)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[84]
        atom_c = self.atoms[194]

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000001413)

        result = get_charge_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000046622)

    def test_get_dipole_dipole_interactions(self):
        """
        Test the calculation of dipole-dipole interactions.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -1.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.036855669)

        result = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.036855669)

        result_a = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        result_b = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -3.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.110567008)

        result = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.110567008)

        result_a = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        result_b = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)
        atom_d = Atom('H', array([8.9, 0.1, 2.3]), -2.345, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.04048258)

        result = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.04048258)

        result_a = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        result_b = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[194]
        atom_c = self.atoms[20]
        atom_d = self.atoms[84]

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000036649)

        result = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000036649)

        result_a = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.WATER.value)
        result_b = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        result = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.001209564)

        result = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.001209564)

        result_a = get_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b, RelativePermittivity.TOLUENE.value)
        result_b = get_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a, RelativePermittivity.TOLUENE.value)
        self.assertEqual(result_a, result_b)

    def test_get_charge_freely_rotating_dipole_interaction(self):
        """
        Test the calculation of charge-dipole interactions for a freely rotating dipole.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +0.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_freely_rotating_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(result, 0.0)

        atom_a = Atom('H', array([0.0, 0.0, 0.0]), +0.0, 0)
        atom_b = Atom('H', array([0.0, 2.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_freely_rotating_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(result, 0.0)

        atom_a = Atom('H', array([0.0, 0.0, 0.0]), -1.0, 0)
        atom_b = Atom('H', array([0.0, 2.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_freely_rotating_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.002529978)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_freely_rotating_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.00001244)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[84]
        atom_c = self.atoms[194]

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_freely_rotating_dipole_interaction(atom_a, dipole_moment,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000000)

        result = get_charge_freely_rotating_dipole_interaction(atom_a, dipole_moment,
                                                               RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000000506)

    def test_get_freely_rotating_dipole_dipole_interaction(self):
        """
        Test the calculation of freely rotating dipole-dipole interactions.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -1.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -1.079457485)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -1.079457485)

        result_a = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                 RelativePermittivity.WATER.value)
        result_b = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                 RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -3.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -9.715117367)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -9.715117367)

        result_a = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                 RelativePermittivity.WATER.value)
        result_b = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                 RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)
        atom_d = Atom('H', array([8.9, 0.1, 2.3]), -2.345, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.491989268)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.491989268)

        result_a = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                 RelativePermittivity.WATER.value)
        result_b = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                 RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[194]
        atom_c = self.atoms[20]
        atom_d = self.atoms[84]

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000229)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                               RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000229)

        result_a = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                 RelativePermittivity.WATER.value)
        result_b = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                 RelativePermittivity.WATER.value)
        self.assertEqual(result_a, result_b)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                               RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000249324)

        result = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                               RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000249324)

        result_a = get_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                 RelativePermittivity.TOLUENE.value)
        result_b = get_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                 RelativePermittivity.TOLUENE.value)
        self.assertEqual(result_a, result_b)

    def test_get_non_polar_freely_rotating_dipole_dipole_interaction(self):
        """
        Test the calculation of freely rotating dipole-dipole non-polar interactions.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -1.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.023947127)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.019552748)

        # result_a = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
        #                                                                    RelativePermittivity.WATER.value)
        # result_b = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
        #                                                                    RelativePermittivity.WATER.value)
        # self.assertEqual(result_a, result_b)

        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), -3.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +1.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.053881037)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.07821099)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)
        atom_d = Atom('H', array([8.9, 0.1, 2.3]), -2.345, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.001807814)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.011824729)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[194]
        atom_c = self.atoms[20]
        atom_d = self.atoms[84]

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000408)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                         RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000036)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_a, dipole_moment_b,
                                                                         RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000444306)

        result = get_non_polar_freely_rotating_dipole_dipole_interaction(dipole_moment_b, dipole_moment_a,
                                                                         RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000039060)

    def test_get_dipole_non_polar_molecule_interaction(self):
        """
        Test the calculation of interactions between dipoles and non-polar molecules.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +0.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +0.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_non_polar_molecule_interaction(dipole_moment_a, dipole_moment_b,
                                                           RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.011973564)

        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +0.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +0.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_non_polar_molecule_interaction(dipole_moment_a, dipole_moment_b,
                                                           RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.026940518)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +0.000, 2)
        atom_d = Atom('H', array([8.9, 0.1, 2.3]), +0.000, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_dipole_non_polar_molecule_interaction(dipole_moment_a, dipole_moment_b,
                                                           RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.001259186)

    def test_get_charge_non_polar_dipole_interaction(self):
        """
        Test the calculation of charge-dipole interactions for a freely rotating dipole.
        """

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +0.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_non_polar_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(result, -0.0)

        atom_a = Atom('H', array([0.0, 0.0, 0.0]), +0.0, 0)
        atom_b = Atom('H', array([0.0, 2.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_non_polar_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(result, -0.0)

        atom_a = Atom('H', array([0.0, 0.0, 0.0]), -1.0, 0)
        atom_b = Atom('H', array([0.0, 2.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +1.0, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_non_polar_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000032204)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +8.901, 2)

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_non_polar_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000032)

        # Test real-life cases.
        atom_a = self.atoms[0]
        atom_b = self.atoms[84]
        atom_c = self.atoms[194]

        dipole_moment = DipoleMoment(atom_b, atom_c)

        result = get_charge_non_polar_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000041)

        result = get_charge_non_polar_dipole_interaction(atom_a, dipole_moment, RelativePermittivity.TOLUENE.value)
        self.assertEqual(round(result, 9), -0.000044290)

    def test_get_london_dispersion_force(self):
        """
        Test the calculation of London dispersion force.
        """

        # Register the energy of the highest occupied molecular orbital.
        homo_energy = 0.0001

        # Test simple cases.
        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -1.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +0.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +0.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_london_dispersion_force(dipole_moment_a, dipole_moment_b, homo_energy,
                                             RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000005745)

        atom_a = Atom('H', array([1.0, 0.0, 0.0]), +1.0, 0)
        atom_b = Atom('H', array([0.0, 1.0, 0.0]), -2.0, 1)
        atom_c = Atom('H', array([0.0, 0.0, 1.0]), +0.0, 2)
        atom_d = Atom('H', array([1.0, 1.0, 0.0]), +0.0, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_london_dispersion_force(dipole_moment_a, dipole_moment_b, homo_energy,
                                             RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000005745)

        # Test complex (absurd) cases.
        atom_a = Atom('H', array([0.1, 2.3, 4.5]), +0.123, 0)
        atom_b = Atom('H', array([6.7, 8.9, 0.1]), -4.567, 1)
        atom_c = Atom('H', array([2.3, 4.5, 6.7]), +0.000, 2)
        atom_d = Atom('H', array([8.9, 0.1, 2.3]), +0.000, 3)

        dipole_moment_a = DipoleMoment(atom_a, atom_b)
        dipole_moment_b = DipoleMoment(atom_c, atom_d)

        result = get_london_dispersion_force(dipole_moment_a, dipole_moment_b, homo_energy,
                                             RelativePermittivity.WATER.value)
        self.assertEqual(round(result, 9), -0.000000575)
