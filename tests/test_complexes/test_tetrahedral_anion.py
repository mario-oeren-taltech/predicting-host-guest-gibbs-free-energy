import unittest

from numpy import round

from complexes.complex_guest_tetrahedral_anion import ComplexGuestTetrahedralAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms
from tests.helper_functions import build_path


class TestBindingEnergies(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment before each test.
        """

        # Create a list of Atom objects from the given XYZ and charge files
        self.atoms = make_list_of_atoms(
            build_path('anion_tetrahedral_geometry.xyz'), build_path('anion_tetrahedral_charges')
        )

        # Create a Guest object representing the central atom and its vertex atoms
        self.guest = Guest(central_atom=260, vertex_atoms=[258, 259, 261, 262])

        # Instantiate the ComplexGuestTetrahedralAnion object with the atoms, guest, and solvent
        self.complex_guest = ComplexGuestTetrahedralAnion(self.atoms, self.guest, 'methanol', 10.0)

    def test_binding_energy_fixed_dipole(self):
        """
        Test binding energy with fixed dipole interactions.
        """

        binding_energy = self.complex_guest.get_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00003068)

    def test_binding_energy_non_polar(self):
        """
        Test binding energy with non-polar interactions.
        """

        binding_energy = self.complex_guest.get_non_polar_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00000132)

    def test_binding_energy_freely_rotating_dipole(self):
        """
        Test binding energy with freely rotating dipole interactions.
        """

        binding_energy = self.complex_guest.get_freely_rotating_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00003179)

    def test_binding_energy_non_polar_freely_rotating_dipole(self):
        """
        Test binding energy with non-polar interactions and freely rotating dipole.
        """

        binding_energy = self.complex_guest.get_freely_rotating_dipoles_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00000163)

    def test_binding_energy_two_non_polar(self):
        """
        Test binding energy with two non-polar interactions.
        """

        binding_energy = self.complex_guest.get_london_dispersion_force()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00194021)
