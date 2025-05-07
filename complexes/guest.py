class Guest:
    """
    A class that represents the atom indices of the guest anion, cation, or a neutral molecule.
    """

    def __init__(self, central_atom: int | None, vertex_atoms: list[int]):
        """
        :param central_atom: The central atom index of the guest (None for molecules or asymmetric ions).
        :param vertex_atoms: Atom indices of the guest except the central atom.
        """

        self.central_atom = central_atom
        self.vertex_atoms = vertex_atoms

        self.atoms = [self.central_atom] + self.vertex_atoms
