import json
import os

from ase.io import read
from dscribe.descriptors import ACSF, SOAP, LMBTR

from complexes.complex_guest_octahedral_anion import ComplexGuestOctahedralAnion
from complexes.complex_guest_spherical_anion import ComplexGuestSphericalAnion
from complexes.complex_guest_tetrahedral_anion import ComplexGuestTetrahedralAnion
from complexes.guest import Guest
from molecular_structure.molecular_structure import make_list_of_atoms


# Set up the descriptor parameters.
element_symbols = [
    'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
]

r_cut = 15.0

# Set up the ACSF descriptor.
g2 = [[1, 1], [1, 2], [1, 3]]
g4 = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

acsf = ACSF(
    species=element_symbols, r_cut=r_cut, g2_params=g2, g4_params=g4
)

#SOAP descriptor
soap = SOAP(
    species=element_symbols,
    r_cut=r_cut,
    n_max=8,
    l_max=6,
    sigma=0.5,
    periodic=False
)


# LMBTR descriptor
lmbtr = LMBTR(
    species=element_symbols,
    geometry={"function": "distance"},
    grid={"min": 0, "max": 5, "n": 100, "sigma": 0.1},
    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
    periodic=False,
    normalization="l2",
)


# Empty the previous dataset (if it exists).
with open('full_dataset.csv', 'w') as anion_data:
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
        london_dispersion_force = 0
        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'anions-{index:03}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        # Compute SOAP descriptor
        # soap_descriptor = soap.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(soap_descriptor)
        # soap_descriptor = ','.join([str(value) for value in soap_descriptor])

        # Compute LMBTR descriptor for a given structure and central atom
        # lmbtr_descriptor = lmbtr.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(lmbtr_descriptor)
        # lmbtr_descriptor = ','.join([str(value) for value in lmbtr_descriptor])

        if index == 0:

            # Open the anion data file.
            with open('full_dataset.csv', 'w') as anion_data:
                anion_data.write(
                    f'Anion Index,'
                    f'Experimental dG,'
                    f'Dipole Interactions,'
                    f'Non-polar Dipole Interactions,'
                    f'Freely rotating Dipole Interactions,'
                    f'Freely rotating Dipoles Interactions,'
                    f'london_dispersion_force,'
                    f'Covalent Radius,'
                    f'{",".join(str(n) for n in range(number_descriptors))}'
                    '\n'
                )



        with open('full_dataset.csv', 'a') as anion_data:

            anion_data.write(
                f'{index + 105:03},'
                f'{dG},'
                f'{dipole_interactions},'
                f'{non_polar_dipole_interactions},'
                f'{freely_rotating_dipole_interactions},'
                f'{freely_rotating_dipoles_interactions},'
                f'{london_dispersion_force},'
                f'{atoms[central_atom].covalent_radius},'
                f'{acsf_descriptor}'
                #f'{soap_descriptor}'
                #f'{lmbtr_descriptor}'   
                
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
        london_dispersion_force = 0
        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'anions-not-caviton-{index:03}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        # Compute SOAP descriptor
        # soap_descriptor = soap.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(soap_descriptor)
        # soap_descriptor = ','.join([str(value) for value in soap_descriptor])

        # Compute LMBTR descriptor for a given structure and central atom
        # lmbtr_descriptor = lmbtr.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(lmbtr_descriptor)
        # lmbtr_descriptor = ','.join([str(value) for value in lmbtr_descriptor])

        with open('full_dataset.csv', 'a') as anion_data:

            anion_data.write(
                f'{index + 105:03},'
                f'{dG},'
                f'{dipole_interactions},'
                f'{non_polar_dipole_interactions},'
                f'{freely_rotating_dipole_interactions},'
                f'{freely_rotating_dipoles_interactions},'
                f'{london_dispersion_force},'
                f'{atoms[central_atom].covalent_radius},'
                f'{acsf_descriptor}'
                #f'{soap_descriptor}'
                #f'{lmbtr_descriptor}'   
                
                '\n'
            )

    except FileNotFoundError:
        pass



for index in list(range(0, 2000)):

    try:
        print(index)
        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'organic-molecules-{index:04}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'organic-molecules-{index:04}-geometry.xyz'),
                                   os.path.join(os.getcwd(), f'organic-molecules-{index:04}-charges'))
        guest = Guest(central_atom, vertex_atoms)


        host_guest_complex = ComplexGuestTetrahedralAnion(atoms, guest, solvent, homo_lumo_gap)



        dipole_interactions = host_guest_complex.get_dipole_interactions(6.0)
        non_polar_dipole_interactions = host_guest_complex.get_non_polar_dipole_interactions(6.0)
        freely_rotating_dipole_interactions = host_guest_complex.get_freely_rotating_dipole_interactions(6.0)
        freely_rotating_dipoles_interactions = host_guest_complex.get_freely_rotating_dipoles_interactions(6.0)
        london_dispersion_force = host_guest_complex.get_london_dispersion_force(6.0)

        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'organic-molecules-{index:04}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        # Compute SOAP descriptor
        # soap_descriptor = soap.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(soap_descriptor)
        # soap_descriptor = ','.join([str(value) for value in soap_descriptor])

        # Compute LMBTR descriptor for a given structure and central atom
        # lmbtr_descriptor = lmbtr.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(lmbtr_descriptor)
        # lmbtr_descriptor = ','.join([str(value) for value in lmbtr_descriptor])

        with open('full_dataset.csv', 'a') as anion_data:

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
                #f'{soap_descriptor}'
                #f'{lmbtr_descriptor}'   
                
                '\n'
            )

    except FileNotFoundError:
        pass


for index in list(range(0, 4000)):

    try:
        print(index)
        # Get the experimental and computational data.
        with open(os.path.join(os.getcwd(), f'dataset-{index:04}-information.json'), 'r') as information_file:
            complex_information = json.load(information_file)

        central_atom = complex_information.get('Guest - Central Atom')
        vertex_atoms = list(complex_information.get('Guest - Vertex Atoms'))
        solvent = complex_information.get('Solvent')
        homo_lumo_gap = complex_information.get('HOMO-LUMO Gap')

        dG = complex_information.get('dG')

        # Register the ComplexGuestAnion object.
        atoms = make_list_of_atoms(os.path.join(os.getcwd(), f'dataset-{index:04}-geometry.xyz'),
                                   os.path.join(os.getcwd(), f'dataset-{index:04}-charges'))
        guest = Guest(central_atom, vertex_atoms)


        host_guest_complex = ComplexGuestTetrahedralAnion(atoms, guest, solvent, homo_lumo_gap)



        dipole_interactions = host_guest_complex.get_dipole_interactions(6.0)
        non_polar_dipole_interactions = host_guest_complex.get_non_polar_dipole_interactions(6.0)
        freely_rotating_dipole_interactions = host_guest_complex.get_freely_rotating_dipole_interactions(6.0)
        freely_rotating_dipoles_interactions = host_guest_complex.get_freely_rotating_dipoles_interactions(6.0)
        london_dispersion_force = host_guest_complex.get_london_dispersion_force(6.0)

        # Get the descriptors.
        structure = read(os.path.join(os.getcwd(), f'dataset-{index:04}-geometry.xyz'))

        acsf_descriptor = acsf.create(structure, centers=[central_atom])[0]
        number_descriptors = len(acsf_descriptor)
        acsf_descriptor = ','.join([str(value) for value in acsf_descriptor])

        # Compute SOAP descriptor
        # soap_descriptor = soap.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(soap_descriptor)
        # soap_descriptor = ','.join([str(value) for value in soap_descriptor])

        # Compute LMBTR descriptor for a given structure and central atom
        # lmbtr_descriptor = lmbtr.create(structure, centers=[central_atom])[0]
        # number_descriptors = len(lmbtr_descriptor)
        # lmbtr_descriptor = ','.join([str(value) for value in lmbtr_descriptor])

        with open('full_dataset.csv', 'a') as anion_data:

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
                #f'{soap_descriptor}'
                #f'{lmbtr_descriptor}'   
                
                '\n'
            )

    except FileNotFoundError:
        pass