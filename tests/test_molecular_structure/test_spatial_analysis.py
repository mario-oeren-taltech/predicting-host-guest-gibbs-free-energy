from unittest import TestCase

from numpy import array, array_equal, rad2deg, round

from molecular_structure.spatial_analysis import get_distance, get_angle, get_dihedral_angle, normalise_vector


class TestSpatialAnalysis(TestCase):

    def test_normalise_vector(self):
        """
        Test the normalisation of vectors.
        """

        vector = array([0.0, 0.0, 0.0])
        self.assertTrue(array_equal(array([0.0, 0.0, 0.0]), normalise_vector(vector)))

        vector = array([1.0, 2.0, 3.0])
        self.assertTrue(array_equal(array([0.267261, 0.534522, 0.801784]), round(normalise_vector(vector), 6)))

        vector = array([0.1, 2.3, 4.5])
        self.assertTrue(array_equal(array([0.019784, 0.455022, 0.89026]), round(normalise_vector(vector), 6)))

    def test_get_distance(self):
        """
        Test the calculation of angles between two points.
        """

        point_a, point_b = array([0.0, 0.0, 0.0]), array([0.0, 0.0, 0.0])
        self.assertEqual(round((get_distance(point_a, point_b)), 6), 0.000000)

        point_a, point_b = array([1.0, 0.0, 0.0]), array([0.0, 0.0, 0.0])
        self.assertEqual(round((get_distance(point_a, point_b)), 6), 1.000000)

        point_a, point_b = array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0])
        self.assertEqual(round((get_distance(point_a, point_b)), 6), 5.196152)

    def test_get_angle(self):
        """
        Test the calculation of angles between two vectors.
        """

        vector_a, vector_b = array([1.0, 0.0, 0.0]), array([0.0, 1.0, 0.0])
        self.assertEqual(round((get_angle(vector_a, vector_b)), 6), 1.570796)  # In radians.
        self.assertEqual(rad2deg(get_angle(vector_a, vector_b)), 90.0)  # In degrees.

        vector_a, vector_b = array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0])
        self.assertEqual(round((get_angle(vector_a, vector_b)), 6), 0.225726)  # In radians.

        with self.assertRaises(RuntimeError):
            vector_a, vector_b = array([0.0, 0.0, 0.0]), array([0.0, 0.0, 0.0])
            get_angle(vector_a, vector_b)

        with self.assertRaises(RuntimeError):
            vector_a, vector_b = array([0.0, 0.0, 0.0]), array([1.0, 0.0, 0.0])
            get_angle(vector_a, vector_b)

        with self.assertRaises(RuntimeError):
            vector_a, vector_b = array([1.0, 0.0, 0.0]), array([0.0, 0.0, 0.0])
            get_angle(vector_a, vector_b)

    def test_get_dihedral_angle(self):
        """
        Test the calculation of dihedral angles between three vectors.
        """

        point_a, point_b = array([0.0, 0.0, 1.0]), array([0.0, 0.0, 0.0])
        point_c, point_d = array([0.0, 1.0, 0.0]), array([1.0, 0.0, 0.0])

        vector_a = point_b - point_a
        vector_b = point_b - point_c
        vector_c = point_d - point_c

        self.assertEqual(round((get_dihedral_angle(vector_a, vector_b, vector_c)), 6), 1.570796)  # In radians.
        self.assertEqual(rad2deg(get_dihedral_angle(vector_a, vector_b, vector_c)), 90.0)  # In degrees.

        point_a, point_b = array([0.1, 2.3, 4.5]), array([6.7, 8.9, 0.1])
        point_c, point_d = array([2.3, 4.5, 6.7]), array([8.9, 0.1, 2.3])

        vector_a = point_b - point_a
        vector_b = point_b - point_c
        vector_c = point_d - point_c

        self.assertEqual(round((get_dihedral_angle(vector_a, vector_b, vector_c)), 6), 1.808737)  # In radians.

        with self.assertRaises(RuntimeError):
            vector_a, vector_b, vector_c = array([0.0, 0.0, 0.0]), array([0.0, 0.0, 0.0]), array([0.0, 0.0, 0.0])
            get_dihedral_angle(vector_a, vector_b, vector_c)

        with self.assertRaises(RuntimeError):
            vector_a, vector_b, vector_c = array([0.0, 0.0, 0.0]), array([0.0, 1.0, 0.0]), array([0.0, 0.0, 1.0])
            get_dihedral_angle(vector_a, vector_b, vector_c)

        with self.assertRaises(RuntimeError):
            vector_a, vector_b, vector_c = array([1.0, 0.0, 0.0]), array([0.0, 0.0, 0.0]), array([0.0, 0.0, 1.0])
            get_dihedral_angle(vector_a, vector_b, vector_c)

        with self.assertRaises(RuntimeError):
            vector_a, vector_b, vector_c = array([1.0, 0.0, 0.0]), array([1.0, 0.0, 0.0]), array([0.0, 0.0, 0.0])
            get_dihedral_angle(vector_a, vector_b, vector_c)
