import json
import os

from ase.io import read
from dscribe.descriptors import ACSF

from complexes.complex_guest_octahedral_anion import ComplexGuestOctahedralAnion
from complexes.complex_guest_spherical_anion import ComplexGuestSphericalAnion
from complexes.complex_guest_tetrahedral_anion import ComplexGuestTetrahedralAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms


# Set up the descriptor parameters.
element_symbols = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Sb', 'I', 'Re', 'Li', 'Si']
r_cut = 15.0

# Set up the ACSF descriptor.
g2 = [[1, 1], [1, 2], [1, 3]]
g4 = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

acsf = ACSF(
    species=element_symbols, r_cut=r_cut, g2_params=g2, g4_params=g4
)





confromer = "conf1"

# Empty the previous dataset (if it exists).
with open('organic-molecules-data.csv', 'w') as anion_data:
    pass


# Go over every potential data point.

index = 0
for index in range(800):

    try:
        print(index)
        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'organic-molecules-{confromer}-{index:03}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'organic-molecules-{confromer}-{index:03}.xyz'),
                                   os.path.join(os.getcwd(), f'organic-molecules-{confromer}-{index:03}-charges'))
        guest = Guest(central_atom, vertex_atoms)


        host_guest_complex = ComplexGuestTetrahedralAnion(atoms, guest, solvent, homo_lumo_gap)



        dipole_interactions = host_guest_complex.get_dipole_interactions(6.0)
        non_polar_dipole_interactions = host_guest_complex.get_non_polar_dipole_interactions(6.0)
        freely_rotating_dipole_interactions = host_guest_complex.get_freely_rotating_dipole_interactions(6.0)
        freely_rotating_dipoles_interactions = host_guest_complex.get_freely_rotating_dipoles_interactions(6.0)
        london_dispersion_force = host_guest_complex.get_london_dispersion_force(6.0)

        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'organic-molecules-{confromer}-{index:03}.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        with open('organic-molecules-data.csv', 'a') as anion_data:

            anion_data.write(
                f'{index:03},'
                f'{dG},'
                f'{dipole_interactions},'
                f'{non_polar_dipole_interactions},'
                f'{freely_rotating_dipole_interactions},'
                f'{freely_rotating_dipoles_interactions},'
                f'{london_dispersion_force},'
                f'{atoms[central_atom].covalent_radius},'
                f'{acsf_descriptor}'
                '\n'
            )

    except FileNotFoundError:
        pass

