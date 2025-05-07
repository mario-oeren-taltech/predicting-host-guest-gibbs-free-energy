# affinity-order
Calculating the affinity for host-guest complexes.

````commandline
import os

from complexes.complex_guest_spherical_anion import ComplexGuestSphericalAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms


cartesian_coordinates = os.path.join(os.getcwd(), 'example_geometry.xyz')
charges = os.path.join(os.getcwd(), 'example_charges')

host_guest_complex = ComplexGuestSphericalAnion(make_list_of_atoms(cartesian_coordinates, charges), Guest(0, []), 'Water')

print(host_guest_complex.get_binding_energy())

````
