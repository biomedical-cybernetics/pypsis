import unittest

import numpy as np
from sklearn.datasets import load_iris

from psis import indices
from tests import sample_data


class TestTrustworthinessComputation(unittest.TestCase):

    def test_centroid_based_trustworthiness(self):
        matrix, labels, positives = sample_data._swiss_roll_sample_data()
        projection = 'centroid'
        formula = 'median'
        iterations = 50
        seed = 100

        model = indices.compute_trustworthiness(matrix, labels, positives, projection_type=projection,
                                                center_formula=formula, iterations=iterations, seed=seed)

        # psi-p
        self.assertEqual(1.2428266254044471e-40, model['psi_p']['value'])
        self.assertEqual(0.817396238671282, model['psi_p']['max'])
        self.assertEqual(0.017227740338773223, model['psi_p']['min'])
        self.assertEqual(0.22880354422930693, model['psi_p']['std'])
        self.assertEqual(0.0196078431372549, model['psi_p']['p_value'])

        # psi-roc
        self.assertEqual(0.8739845884072535, model['psi_roc']['value'])
        self.assertEqual(0.5754732669816965, model['psi_roc']['max'])
        self.assertEqual(0.504562187438256, model['psi_roc']['min'])
        self.assertEqual(0.018282702477521843, model['psi_roc']['std'])
        self.assertEqual(0.0196078431372549, model['psi_roc']['p_value'])

        # psi-pr
        self.assertEqual(0.7829196344855981, model['psi_pr']['value'])
        self.assertEqual(0.3184859473523561, model['psi_pr']['max'])
        self.assertEqual(0.2704583081970596, model['psi_pr']['min'])
        self.assertEqual(0.009793843589861682, model['psi_pr']['std'])
        self.assertEqual(0.0196078431372549, model['psi_pr']['p_value'])

        # psi-mcc
        self.assertEqual(0.628615966394596, model['psi_mcc']['value'])
        self.assertEqual(0.10585904641997314, model['psi_mcc']['max'])
        self.assertEqual(0.008667971955796558, model['psi_mcc']['min'])
        self.assertEqual(0.02324139114770061, model['psi_mcc']['std'])
        self.assertEqual(0.0196078431372549, model['psi_mcc']['p_value'])

    def test_lda_based_trustworthiness(self):
        matrix, labels, positives = sample_data._swiss_roll_sample_data()
        projection = 'lda'
        iterations = 50
        seed = 100

        model = indices.compute_trustworthiness(matrix, labels, positives, projection_type=projection,
                                                iterations=iterations, seed=seed)

        # psi-p
        self.assertEqual(1.009261421351576e-40, model['psi_p']['value'])
        self.assertEqual(0.6521271550825026, model['psi_p']['max'])
        self.assertEqual(0.010606992074642272, model['psi_p']['min'])
        self.assertEqual(0.1627033162824601, model['psi_p']['std'])
        self.assertEqual(0.0196078431372549, model['psi_p']['p_value'])

        # psi-roc
        self.assertEqual(0.9176549415996585, model['psi_roc']['value'])
        self.assertEqual(0.5816329751950958, model['psi_roc']['max'])
        self.assertEqual(0.513258579532901, model['psi_roc']['min'])
        self.assertEqual(0.015973959781501835, model['psi_roc']['std'])
        self.assertEqual(0.0196078431372549, model['psi_roc']['p_value'])

        # psi-pr
        self.assertEqual(0.818340188219321, model['psi_pr']['value'])
        self.assertEqual(0.3271676715034102, model['psi_pr']['max'])
        self.assertEqual(0.2864985337058604, model['psi_pr']['min'])
        self.assertEqual(0.00936130564040644, model['psi_pr']['std'])
        self.assertEqual(0.0196078431372549, model['psi_pr']['p_value'])

        # psi-mcc
        self.assertEqual(0.6401887507956918, model['psi_mcc']['value'])
        self.assertEqual(0.11584620474627294, model['psi_mcc']['max'])
        self.assertEqual(0.014460273138448517, model['psi_mcc']['min'])
        self.assertEqual(0.025160919960162023, model['psi_mcc']['std'])
        self.assertEqual(0.0196078431372549, model['psi_mcc']['p_value'])


class TestIndicesComputation(unittest.TestCase):

    def test_wrong_inputs(self):
        cases = [
            # wrong matrix
            {
                'matrix': [[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]],
                'labels': np.array(
                    ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2']),
                'positives': np.array(['sample1']),
                'center': 'median'
            },
            # wrong labels
            {
                'matrix': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]]),
                'labels': list(
                    ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2']),
                'positives': np.array(['sample1']),
                'center': 'median'
            },
            # wrong positives
            {
                'matrix': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]]),
                'labels': np.array(
                    ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2']),
                'positives': 'sample1',
                'center': 'median'
            }
        ]

        for case in cases:
            self.assertRaises(
                TypeError,
                indices.compute_psis,
                case['matrix'],
                case['labels'],
                case['positives'],
                case['center']
            )

    def test_wrong_center_formula(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        input_positive = np.array(['sample1'])
        input_projection = 'centroid'
        input_formula = 'fake-formula'

        self.assertWarns(SyntaxWarning, indices.compute_psis, input_matrix, input_labels, input_positive,
                         input_projection, input_formula)

    def test_centroid_based_perfect_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        input_positive = np.array(['sample1'])
        input_formula = 'median'
        input_projection_type = 'centroid'

        expected_psi_p = 0.0286
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection_type,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_lda_based_perfect_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        input_positive = np.array(['sample1'])
        input_formula = ''  # ignored
        input_projection_type = 'lda'

        expected_psi_p = 0.0286
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection_type,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_centroid_based_high_dimensional_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        input_positive = np.array(['sample1'])
        input_formula = 'median'
        input_projection_type = 'centroid'

        expected_psi_p = 0.0286
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection_type,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_lda_based_high_dimensional_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        input_positive = np.array(['sample1'])
        input_formula = ''  # ignored
        input_projection_type = 'lda'

        expected_psi_p = 0.0286
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection_type,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_centroid_based_mixed_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample2', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample1'])
        input_positive = np.array(['sample1'])
        input_formula = 'median'

        expected_psi_p = 0.8857
        expected_psi_roc = 0.5625
        expected_psi_pr = 0.5015
        expected_psi_mcc = 0.5000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_formula)

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

        expected_psi_p = 0.6857
        expected_psi_roc = 0.6250
        expected_psi_pr = 0.6673
        expected_psi_mcc = 0.0000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_lda_based_multiclass_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample2', 'sample2', 'sample3', 'sample3', 'sample4', 'sample4'])
        input_positive = np.array(['sample2', 'sample3', 'sample4'])
        input_formula = ''  # ignored
        input_projection_type = 'lda'

        expected_psi_p = 0.3333
        expected_psi_roc = 1.0000
        expected_psi_pr = 1.0000
        expected_psi_mcc = 1.0000

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection_type,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_centroid_based_high_dimensional_multiclass_separation(self):
        data = load_iris()

        samples = np.empty(len(data.target), dtype=object)
        samples[data.target == 0] = data.target_names[0]
        samples[data.target == 1] = data.target_names[1]
        samples[data.target == 2] = data.target_names[2]
        positives = np.unique(samples)
        positives = np.delete(positives, 0)

        input_matrix = data.data
        input_labels = samples
        input_positive = positives
        input_formula = 'median'
        input_projection = 'centroid'

        expected_psi_p = 0.0000
        expected_psi_roc = 0.9723
        expected_psi_pr = 0.9707
        expected_psi_mcc = 0.8080

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection,
                                                                                           input_formula)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_lda_based_high_dimensional_multiclass_separation(self):
        data = load_iris()

        samples = np.empty(len(data.target), dtype=object)
        samples[data.target == 0] = data.target_names[0]
        samples[data.target == 1] = data.target_names[1]
        samples[data.target == 2] = data.target_names[2]
        positives = np.unique(samples)
        positives = np.delete(positives, 0)

        input_matrix = data.data
        input_labels = samples
        input_positive = positives
        input_formula = ''  # ignored
        input_projection = 'lda'

        expected_psi_p = 0.0000
        expected_psi_roc = 0.9975
        expected_psi_pr = 0.9976
        expected_psi_mcc = 0.9644

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_projection,
                                                                                           input_formula)

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

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_formula)

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

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           input_positive,
                                                                                           input_formula)

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

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels)

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))

    def test_multidimensional_mean_centered_separation(self):
        input_matrix, input_labels, input_positives = sample_data._smokers_sample_data()

        expected_psi_p = 0.0
        expected_psi_roc = 0.7999
        expected_psi_pr = 0.713
        expected_psi_mcc = 0.4207

        actual_psi_p, actual_psi_roc, actual_psi_pr, actual_psi_mcc = indices.compute_psis(input_matrix,
                                                                                           input_labels,
                                                                                           center_formula='mean')

        self.assertEqual(expected_psi_p, round(actual_psi_p, 4))
        self.assertEqual(expected_psi_roc, round(actual_psi_roc, 4))
        self.assertEqual(expected_psi_pr, round(actual_psi_pr, 4))
        self.assertEqual(expected_psi_mcc, round(actual_psi_mcc, 4))


if __name__ == '__main__':
    unittest.main()
