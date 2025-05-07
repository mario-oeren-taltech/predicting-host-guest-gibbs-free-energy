from molecular_structure.atom import Atom


def get_structure_coordinates(coordinates_file_path: str) -> tuple[list[str], list[list[float]]]:
    """
    The function returns a list of coordinates obtained from a Cartesian coordinates file.

    :param coordinates_file_path: The full path of the Cartesian coordinates file.
    :raises FileNotFoundError: The Coordinates file does not exist.
    :return: The list of coordinates (as a list).
    """

    molecule_elements = []
    molecule_coordinates = []

    with open(coordinates_file_path, "r") as file:
        for line_number, line in enumerate(file, 1):
            if line_number > 2:  # Skip the first two lines.
                file_lines = line.split()
                molecule_elements.append(file_lines[0])
                molecule_coordinates.append([float(coordinate) for coordinate in file_lines[1:]])

    return molecule_elements, molecule_coordinates


def get_partial_charges(charge_file_path: str) -> list[float]:
    """
    Registers the charges from a charge file.

    :param charge_file_path: The full path to the file with partial charges.
    :return: A list with partial charges.
    """

    charges = []

    with open(charge_file_path, 'r') as charges_file:
        for line in charges_file:
            charges.append(float(line.strip()))

    return charges


def make_list_of_atoms(coordinate_file_path: str, charge_file_path: str) -> list[Atom]:
    """
    The function forms a list of Atom objects from the coordinate and atom charge files.

    :param coordinate_file_path: The full path to the Cartesian coordinates file.
    :param charge_file_path: The full path to the atom charges file.
    :return: A list of Atom objects.
    """

    elements, coordinates = get_structure_coordinates(coordinate_file_path)
    charges = get_partial_charges(charge_file_path)
    indices = list(range(len(coordinates)))

    return [Atom(element, coordinate, charge, index) for element, coordinate, charge, index in zip(
        elements, coordinates, charges, indices)]
