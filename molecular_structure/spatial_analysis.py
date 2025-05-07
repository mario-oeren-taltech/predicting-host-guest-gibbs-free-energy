from numpy import arccos, arctan2, clip, cross, dot, floating, linalg, ndarray


def normalise_vector(vector: ndarray) -> ndarray:
    """
    The function normalises a vector.

    :param vector: The given vector as an NumPy array.
    :return: The normalised vector
    """

    if all(float(member) in [0.0, -0.0] for member in vector):
        return vector

    return vector / linalg.norm(vector)


def get_distance(coordinates_a: ndarray, coordinates_b: ndarray) -> floating:
    """
    Calculates the Euclidean distance between two points.

    :param coordinates_a: Coordinates of the first point.
    :param coordinates_b: Coordinates of the second point.
    :return: The Euclidean distance between the given points.
    """

    return linalg.norm(coordinates_b - coordinates_a)


def get_angle(vector_a: ndarray, vector_b: ndarray) -> float:
    """
    Calculate the angle between two given vectors.

    :param vector_a: The first vector as an NumPy array.
    :param vector_b: The second vector as an NumPy array.
    :raises RuntimeError: The length of one or both of the vectors is zero.
    :return: The angle between the given vectors in radians.
    """

    # Normalise the vectors.
    vector_a_normalised, vector_b_normalised = linalg.norm(vector_a), linalg.norm(vector_b)

    # Raise an error if zero-length vector is detected.
    if vector_a_normalised == 0.0 or vector_b_normalised == 0.0:
        raise RuntimeError('cannot normalise a zero-length vector.')

    # Return the angle in radians (use the clip function to avoid invalid input to arccos function).
    return arccos(clip(dot(vector_a, vector_b) / (vector_a_normalised * vector_b_normalised), -1.0, 1.0))


def get_dihedral_angle(vector_a: ndarray, vector_b: ndarray, vector_c: ndarray) -> float:
    """
    Calculate the angle between two given vectors.

    :param vector_a: The first vector as an NumPy array.
    :param vector_b: The second vector as an NumPy array.
    :param vector_c: The third vector as an NumPy array.
    :raises RuntimeError: The length of one or both of the vectors is zero.
    :return: The angle between the given vectors in radians.
    """

    # Raise an error if zero-length vector is detected.
    if linalg.norm(vector_a) == 0.0 or linalg.norm(vector_b) == 0.0 or linalg.norm(vector_c) == 0.0:
        raise RuntimeError('cannot normalise a zero-length vector.')

    vector_b_normalised = normalise_vector(vector_b)

    projection_1 = vector_a - dot(vector_a, vector_b_normalised) * vector_b_normalised
    projection_2 = vector_c - dot(vector_c, vector_b_normalised) * vector_b_normalised

    return float(arctan2(dot(cross(vector_b_normalised, projection_1), projection_2), dot(projection_1, projection_2)))
