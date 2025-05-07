import json
import os

from ase.io import read
from dscribe.descriptors import ACSF

from complexes.complex_guest_octahedral_anion import ComplexGuestOctahedralAnion
from complexes.complex_guest_spherical_anion import ComplexGuestSphericalAnion
from complexes.complex_guest_tetrahedral_anion import ComplexGuestTetrahedralAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms


# Set up the ACSF descriptor.
element_symbols = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Sb', 'I', 'Re']
r_cut = 15.0
g2 = [[1, 1], [1, 2], [1, 3]]
g4 = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

acsf = ACSF(species=element_symbols, r_cut=r_cut, g2_params=g2, g4_params=g4)

# Empty the previous dataset (if it exists).
with open('anion-external-data.csv', 'w') as anion_data:
    pass

# Go over every potential data point.
for index in range(10):

    try:

        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'anions-external-{index:03}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'anions-external-{index:03}-geometry.xyz'),
                                   os.path.join(os.getcwd(), f'anions-external-{index:03}-charges'))
        guest = Guest(central_atom, vertex_atoms)

        if len(vertex_atoms) == 0:
            host_guest_complex = ComplexGuestSphericalAnion(atoms, guest, solvent, homo_lumo_gap)

        elif len(vertex_atoms) == 4:
            host_guest_complex = ComplexGuestTetrahedralAnion(atoms, guest, solvent, homo_lumo_gap)

        elif len(vertex_atoms) == 6:
            host_guest_complex = ComplexGuestOctahedralAnion(atoms, guest, solvent, homo_lumo_gap)

        else:
            raise RuntimeError('not dealing with spherical, tetrahedral, or octahedral anions!')

        dipole_interactions = host_guest_complex.get_dipole_interactions(6.0)
        non_polar_dipole_interactions = host_guest_complex.get_non_polar_dipole_interactions(6.0)
        freely_rotating_dipole_interactions = host_guest_complex.get_freely_rotating_dipole_interactions(6.0)
        freely_rotating_dipoles_interactions = host_guest_complex.get_freely_rotating_dipoles_interactions(6.0)

        # Get the ACSF descriptor.
        structure = read(os.path.join(os.getcwd(), f'anions-external-{index:03}-geometry.xyz'))
        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        # Write data to data set file.
        if index == 0:

            # Open the anion data file.
            with open('anion-external-data.csv', 'w') as anion_data:
                anion_data.write(
                    f'Anion Index,'
                    f'Experimental dG,'
                    f'Dipole Interactions,'
                    f'Non-polar Dipole Interactions,'
                    f'Freely rotating Dipole Interactions,'
                    f'Freely rotating Dipoles Interactions,'
                    f'Covalent Radius,'
                    f'{",".join(str(n) for n in range(number_descriptors))}'
                    '\n'
                )

        with open('anion-external-data.csv', 'a') as anion_data:

            anion_data.write(
                f'{index:03},'
                f'{dG},'
                f'{dipole_interactions},'
                f'{non_polar_dipole_interactions},'
                f'{freely_rotating_dipole_interactions},'
                f'{freely_rotating_dipoles_interactions},'
                f'{atoms[central_atom].covalent_radius},'
                f'{acsf_descriptor}'
                '\n'
            )

    except FileNotFoundError:
        pass
