import unittest

from numpy import round

from complexes.complex_guest_spherical_anion import ComplexGuestSphericalAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms
from tests.helper_functions import build_path


class TestComplexGuestSphericalAnion(unittest.TestCase):

    def setUp(self):

        # Set up the list of Atom objects and a Guest object.
        self.atoms = make_list_of_atoms(
            build_path('anion_spherical_geometry.xyz'), build_path('anion_spherical_charges')
        )

        self.guest = Guest(central_atom=0, vertex_atoms=[])

        # Instantiate the ComplexGuestSphericalAnion object
        self.complex_guest = ComplexGuestSphericalAnion(self.atoms, self.guest, "methanol")

    def test_binding_energy(self):
        """
        Test binding energy calculation.
        """

        binding_energy = self.complex_guest.get_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00013829)

    def test_binding_energy_charge_non_polar(self):
        """
        Test binding energy with charge non-polar interactions.
        """

        binding_energy = self.complex_guest.get_non_polar_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), 0.0)

    def test_binding_energy_with_freely_rotating_dipole(self):
        """
        Test binding energy with freely rotating dipole interactions.
        """

        binding_energy = self.complex_guest.get_freely_rotating_dipole_interactions()

        self.assertIsInstance(binding_energy, float)
        self.assertEqual(round(binding_energy, 8), -0.00000847)
