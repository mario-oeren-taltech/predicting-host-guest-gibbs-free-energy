import unittest

from numpy import round

from complexes.complex_guest_octahedral_anion import ComplexGuestOctahedralAnion
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
        self.guest = Guest(central_atom=197, vertex_atoms=[194, 195, 196, 198, 199, 200])

        # Instantiate the ComplexGuestTetrahedralAnion object with the atoms, guest, and solvent
        self.complex_guest = ComplexGuestOctahedralAnion(self.atoms, self.guest, 'methanol', 10.0)

    def test_binding_energy_fixed_dipole(self):
        """
        Test binding energy with fixed dipole interactions.
        """

        binding_energy = self.complex_guest.get_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00000508)

    def test_binding_energy_freely_rotating_dipole(self):
        """
        Test binding energy with freely rotating dipole interactions.
        """

        binding_energy = self.complex_guest.get_freely_rotating_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00000003)
