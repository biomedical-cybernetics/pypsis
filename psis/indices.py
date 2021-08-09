import warnings
import numpy as np
from scipy import stats
from sklearn import metrics


def _mode_distribution(data_clustered):
    mode_dist = np.empty([0])
    _, dims = data_clustered.shape
    for ix in range(dims):
        kde = stats.gaussian_kde(data_clustered[:, ix])
        xi = np.linspace(data_clustered.min(), data_clustered.max(), 100)
        p = kde(xi)
        ind = np.argmax([p])
        mode_dist = np.append(mode_dist, xi[ind])
    return mode_dist


def _find_positive_classes(sample_labels):
    positives, positions = np.unique(sample_labels, return_inverse=True)
    max_pos = np.bincount(positions).argmax()
    positives = np.delete(positives, max_pos)
    return positives


def _nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n / k * _nchoosek(n - 1, k - 1)
    return round(r)


def _compute_mcc(labels, scores, positives):
    total_positive = np.sum(labels == positives)
    total_negative = np.sum(labels != positives)
    negative_class = np.unique(labels[labels != positives]).item()
    true_labels = labels[np.argsort(scores)]

    ps = np.array([positives] * total_positive)
    ng = np.array([negative_class] * total_negative)

    coefficients = np.empty([0])
    for ix in range(0, 2):
        if ix == 0:
            predicted_labels = np.concatenate((ps, ng), axis=0)
        else:
            predicted_labels = np.concatenate((ng, ps), axis=0)
        coefficients = np.append(coefficients, metrics.matthews_corrcoef(true_labels, predicted_labels))

    mcc = np.max(coefficients)

    return mcc


def _create_line_between_centroids(centroid1, centroid2):
    line = np.vstack([centroid1, centroid2])
    return line


def _project_points_on_line(point, line):
    # centroids
    a = line[0]
    b = line[1]

    # deltas
    ap = point - a
    ab = b - a

    # projection
    projected_point = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

    return projected_point


def _convert_points_to_one_dimension(points):
    start_point = None
    _, dims = points.shape

    for ix in range(dims):
        if np.unique(points[:, ix]).size != 1:
            start_point = np.array(points[np.argmin(points[:, ix], axis=0), :]).reshape(1, dims)
            break

    if start_point is None:
        raise RuntimeError('impossible to set projection starting point')

    v = np.zeros(np.shape(points)[0])
    for ix in range(dims):
        v = np.add(v, np.power(points[:, ix] - np.min(start_point[:, ix]), 2))

    v = np.sqrt(v)

    return v


def _compute_mannwhitney(scores_c1, scores_c2):
    mw = stats.mannwhitneyu(scores_c1, scores_c2)  # method="exact"
    return mw


def _compute_auc_aupr(labels, scores, positives):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=positives)
    auc = metrics.auc(fpr, tpr)
    if auc < 0.5:
        auc = 1 - auc
        flipped_scores = 2 * np.mean(scores) - scores
        precision, recall, thresholds = metrics.precision_recall_curve(labels, flipped_scores, pos_label=positives)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=positives)
    aupr = metrics.auc(recall, precision)
    return auc, aupr


def compute_null_model(data_matrix, sample_labels, positive_classes=None, center_formula='median', iterations=1,
                       seed=None):
    psi_p, psi_roc, psi_pr, psi_mcc = compute_psis(data_matrix, sample_labels, positive_classes, center_formula)
    initial_values = dict(psi_p=psi_p, psi_roc=psi_roc, psi_pr=psi_pr, psi_mcc=psi_mcc)

    total_samples = len(sample_labels)

    if seed is not None:
        np.random.seed(seed)

    permutations = dict(psi_p=np.empty([0]), psi_roc=np.empty([0]), psi_pr=np.empty([0]), psi_mcc=np.empty([0]))
    for ix in range(iterations):
        permuted_positions = np.random.permutation(total_samples)
        permuted_samples = sample_labels[permuted_positions]
        perm_p, perm_roc, perm_pr, perm_mcc = compute_psis(data_matrix, permuted_samples, positive_classes,
                                                           center_formula)
        permutations['psi_p'] = np.append(permutations['psi_p'], perm_p)
        permutations['psi_roc'] = np.append(permutations['psi_roc'], perm_roc)
        permutations['psi_pr'] = np.append(permutations['psi_pr'], perm_pr)
        permutations['psi_mcc'] = np.append(permutations['psi_mcc'], perm_mcc)

    delta_degrees_of_freedom = 0
    if iterations > 2:
        delta_degrees_of_freedom = 1

    model_results = dict(psi_p=None, psi_roc=None, psi_pr=None, psi_mcc=None)

    for name, value in permutations.items():
        if name == 'psi_p':
            p_val = (np.sum(permutations[name] < initial_values[name]) + 1) / (iterations + 1)
        else:
            p_val = (np.sum(permutations[name] > initial_values[name]) + 1) / (iterations + 1)

        model_results[name] = dict(
            value=initial_values[name],
            permutations=permutations[name],
            max=np.max(permutations[name]),
            min=np.min(permutations[name]),
            std=np.std(permutations[name], ddof=delta_degrees_of_freedom),
            p_value=p_val
        )

    return model_results


def compute_psis(data_matrix, sample_labels, positive_classes=None, center_formula='median'):
    # TODO: Validate numpy arrays and non-empty args

    if positive_classes is None:
        positive_classes = _find_positive_classes(sample_labels)
    if center_formula != 'mean' and center_formula != 'median' and center_formula != 'mode':
        warnings.warn('invalid center formula: median will be applied')
        center_formula = 'median'

    # checking range of dimensions
    total_samples, dimensions_number = data_matrix.shape
    if len(sample_labels) != total_samples:
        raise RuntimeError("the number of sample labels does not match the number of rows in the matrix")

    # obtaining unique sample labels
    unique_labels = np.unique(sample_labels)
    number_unique_labels = len(unique_labels)

    # clustering data according to sample labels
    sorted_labels = np.empty([0], dtype=str)
    data_clustered = list()
    for k in range(number_unique_labels):
        idxes = np.where(sample_labels == unique_labels[k])
        sorted_labels = np.append(sorted_labels, sample_labels[idxes])
        data_clustered.append(data_matrix[idxes])

    mw_values = np.empty([0])
    auc_values = np.empty([0])
    aupr_values = np.empty([0])
    mcc_values = np.empty([0])
    clusters_projections = [np.empty([0, dimensions_number])] * number_unique_labels
    pairwise_group_combinations = _nchoosek(number_unique_labels, 2)

    n = 0
    m = 1
    for index_group_combination in range(pairwise_group_combinations):
        centroid_cluster_1 = centroid_cluster_2 = None
        if center_formula == 'median':
            centroid_cluster_1 = np.median(data_clustered[n], axis=0)
            centroid_cluster_2 = np.median(data_clustered[m], axis=0)
        elif center_formula == 'mean':
            centroid_cluster_1 = np.mean(data_clustered[n], axis=0)
            centroid_cluster_2 = np.mean(data_clustered[m], axis=0)
        elif center_formula == 'mode':
            centroid_cluster_1 = _mode_distribution(data_clustered[n])
            centroid_cluster_2 = _mode_distribution(data_clustered[m])

        if centroid_cluster_1 is None or centroid_cluster_2 is None:
            raise RuntimeError('impossible to set clusters centroids')
        elif (centroid_cluster_1 == centroid_cluster_2).all():
            raise RuntimeError('clusters have the same centroid: no line can be traced between them')

        clusters_line = _create_line_between_centroids(centroid_cluster_1, centroid_cluster_2)

        clusters_projections[n] = clusters_projections[m] = np.empty([0, dimensions_number])

        for o in range(np.shape(data_clustered[n])[0]):
            proj = _project_points_on_line(data_clustered[n][o], clusters_line)
            clusters_projections[n] = np.vstack([clusters_projections[n], proj])
        for o in range(np.shape(data_clustered[m])[0]):
            proj = _project_points_on_line(data_clustered[m][o], clusters_line)
            clusters_projections[m] = np.vstack([clusters_projections[m], proj])

        size_cluster_n = len(data_clustered[n])
        size_cluster_m = len(data_clustered[m])

        cluster_projection_1d = _convert_points_to_one_dimension(
            np.vstack([clusters_projections[n], clusters_projections[m]]))

        dp_scores_cluster_1 = cluster_projection_1d[0:size_cluster_n]
        dp_scores_cluster_2 = cluster_projection_1d[size_cluster_n:size_cluster_n + size_cluster_m]
        dp_scores = np.concatenate([dp_scores_cluster_1, dp_scores_cluster_2])

        mw = _compute_mannwhitney(dp_scores_cluster_1, dp_scores_cluster_2)
        mw_values = np.append(mw_values, mw.pvalue)

        # sample membership
        samples_cluster_n = sample_labels[np.where(sample_labels == unique_labels[n])[0]]
        samples_cluster_m = sample_labels[np.where(sample_labels == unique_labels[m])[0]]
        sample_labels_membership = np.concatenate((samples_cluster_n, samples_cluster_m), axis=0)

        current_positive_class = None
        for o in range(len(positive_classes)):
            if np.any(sample_labels_membership == positive_classes[o]):
                current_positive_class = positive_classes[o]
                break

        if current_positive_class is None:
            raise RuntimeError('impossible to set the current positive class')

        auc, aupr = _compute_auc_aupr(sample_labels_membership, dp_scores, current_positive_class)
        auc_values = np.append(auc_values, auc)
        aupr_values = np.append(aupr_values, aupr)

        mcc = _compute_mcc(sample_labels_membership, dp_scores, current_positive_class)
        mcc_values = np.append(mcc_values, mcc)

        m = m + 1
        if m > (number_unique_labels - 1):
            n = n + 1
            m = n + 1

    delta_degrees_of_freedom = 0
    if number_unique_labels > 2:
        delta_degrees_of_freedom = 1

    psi_p = (np.mean(mw_values) + np.std(mw_values, ddof=delta_degrees_of_freedom)) / (
            np.std(mw_values, ddof=delta_degrees_of_freedom) + 1)
    psi_roc = np.mean(auc_values) / (np.std(auc_values, ddof=delta_degrees_of_freedom) + 1)
    psi_pr = np.mean(aupr_values) / (np.std(aupr_values, ddof=delta_degrees_of_freedom) + 1)
    psi_mcc = np.mean(mcc_values) / (np.std(mcc_values, ddof=delta_degrees_of_freedom) + 1)

    return psi_p, psi_roc, psi_pr, psi_mcc
