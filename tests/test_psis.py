import unittest
import numpy as np

from psis import __version__
from psis import psis


def test_version():
    assert __version__ == '0.1.0'


class TestProjectionSeparabilityIndices(unittest.TestCase):

    def test_perfect_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        input_positive = np.array(['sample1'])
        input_formula = 'median'

        expected_psi_p = 0.0286
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels, input_positive, input_formula)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_mixed_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample2', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample1'])
        input_positive = np.array(['sample1'])
        input_formula = 'median'

        expected_psi_p = 0.8857
        expected_psi_roc = 0.5625
        expected_psi_pr = 0.5015
        expected_psi_mcc = 0.5000

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels, input_positive, input_formula)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_no_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample2', 'sample1', 'sample2', 'sample1', 'sample2', 'sample1', 'sample2'])
        input_positive = np.array(['sample1'])
        input_formula = 'median'

        expected_psi_p = 0.6857;
        expected_psi_roc = 0.6250;
        expected_psi_pr = 0.6673;
        expected_psi_mcc = 0.0000;

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels, input_positive, input_formula)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_multiclass_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample2', 'sample2', 'sample3', 'sample3', 'sample4', 'sample4'])
        input_positive = np.array(['sample2', 'sample3', 'sample4'])
        input_formula = 'median'

        expected_psi_p = 0.3333
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels, input_positive, input_formula)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_multiclass_mean_centered_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample3', 'sample3', 'sample2', 'sample2', 'sample1', 'sample1', 'sample1', 'sample1'])
        input_positive = np.array(['sample2', 'sample3'])
        input_formula = 'mean'

        expected_psi_p = 0.2828
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels, input_positive, input_formula)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_multiclass_mode_centered_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample3', 'sample3', 'sample2', 'sample2', 'sample1', 'sample1', 'sample1', 'sample1'])
        input_positive = np.array(['sample2', 'sample3'])
        input_formula = 'mode'

        expected_psi_p = 0.2828
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels, input_positive)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_multiclass_separation_default_args(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample3', 'sample3', 'sample2', 'sample2', 'sample1', 'sample1', 'sample1', 'sample1'])

        expected_psi_p = 0.2828
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        psi = psis.ProjectionSeparabilityIndices(input_matrix, input_labels)
        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = psi.calculate()

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))
