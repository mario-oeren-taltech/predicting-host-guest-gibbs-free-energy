import json
import os

from ase.io import read
from dscribe.descriptors import ACSF # , SOAP, LMBTR

from complexes.complex_guest_octahedral_anion import ComplexGuestOctahedralAnion
from complexes.complex_guest_spherical_anion import ComplexGuestSphericalAnion
from complexes.complex_guest_tetrahedral_anion import ComplexGuestTetrahedralAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms


# Set up the descriptor parameters.
element_symbols = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Sb', 'I', 'Re']
r_cut = 15.0

# Set up the SOAP descriptor.
# n_max = 3
# l_max = 3
# weighting = {'function': 'pow', 'r0': 0.503, 'c': 1.0, 'd': 1.0, 'm': 2.0}
#
# soap = SOAP(
#     species=element_symbols,
#     periodic=False,
#     r_cut=r_cut,
#     n_max=n_max,
#     l_max=l_max,
#     weighting=weighting
# )

# Set up the ACSF descriptor.
g2 = [[1, 1], [1, 2], [1, 3]]
g4 = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

acsf = ACSF(
    species=element_symbols, r_cut=r_cut, g2_params=g2, g4_params=g4
)

# Set up the LMBTR descriptor.
# lmbtr = LMBTR(
#     species=element_symbols,
#     geometry={"function": "distance"},
#     grid={"min": 0, "max": 5, "n": 100, "sigma": 0.1},
#     weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
#     periodic=False,
#     normalization="l2",
# )

# Empty the previous dataset (if it exists).
with open('anion-data.csv', 'w') as anion_data:
    pass

# Go over every potential data point.
for index in range(500):

    try:

        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'anions-{index:03}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'anions-{index:03}-geometry.xyz'),
                                   os.path.join(os.getcwd(), f'anions-{index:03}-charges'))
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

        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'anions-{index:03}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        if index == 0:

            # Open the anion data file.
            with open('anion-data.csv', 'w') as anion_data:
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

        with open('anion-data.csv', 'a') as anion_data:

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

# Empty the previous dataset (if it exists).
with open('anion-not-caviton-data.csv', 'w') as anion_data:
    pass

# Go over every potential data point.
for index in range(500):

    try:

        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'anions-{index:03}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'anions-{index:03}-geometry.xyz'),
                                   os.path.join(os.getcwd(), f'anions-{index:03}-charges'))
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

        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'anions-{index:03}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        if index == 0:

            # Open the anion data file.
            with open('anion-data.csv', 'w') as anion_data:
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

            # Open the anion data file.
            with open('anion-not-caviton-data.csv', 'w') as anion_data:
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

        with open('anion-data.csv', 'a') as anion_data:

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

        with open('anion-not-caviton-data.csv', 'a') as anion_data:

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

# Go over every potential data point.
for index in range(500):

    try:

        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'anions-not-caviton-{index:03}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'anions-not-caviton-{index:03}-geometry.xyz'),
                                   os.path.join(os.getcwd(), f'anions-not-caviton-{index:03}-charges'))
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

        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'anions-not-caviton-{index:03}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        with open('anion-not-caviton-data.csv', 'a') as anion_data:

            anion_data.write(
                f'{index + 105:03},'
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
