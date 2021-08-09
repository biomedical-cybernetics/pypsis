# Projection Separability Indices

*This python package is based on the MATLAB project
named [projection-separability-indices](https://github.com/biomedical-cybernetics/projection-separability-indices).*

## Description

The projection separability indices (PSIs) are statistical-based measures specifically designed to assess and quantify
the group separability of data samples in a geometrical space of dimensionality reduction analyses based on embedding
algorithms. Currently, this package implements four different PSIs for evaluating group separability:

* **psi-p**: Based on Mann-Whitney U-test p-value [1]
* **psi-roc**: Based on Area Under the ROC-Curve [2]
* **psi-pr**: Based on Area Under the Precision-Recall Curve [3]
* **psi-mcc**: Based on the Matthews Correlation Coefficient [4]

> [1] H. B. Mann and D. R. Whitney, “On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other,” Ann. Math. Stat., vol. 18, no. 1, pp. 50–60, 1947, doi: 10.1214/aoms/1177730491.
>
> [2] J. S. Hanley and B. J. McNeil, “The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve,” Radiology, vol. 143, no. 1, pp. 29–36, 1982.
>
> [3] V. Raghavan, P. Bollmann, and G. S. Jung, “A critical investigation of recall and precision as measures of retrieval system performance,” ACM Trans. Inf. Syst., vol. 7, no. 3, pp. 205–229, 1989, doi: 10.1145/65943.65945.
>
> [4] B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” BBA - Protein Struct., vol. 405, no. 2, pp. 442–451, 1975, doi: 10.1016/0005-2795(75)90109-9.

## Installation

Run the following to install:

```shell
pip install psis
```

## Usage

### Compute indices

```python
import numpy as np
from psis import indices

"""
Simulated embedding obtained by a dimension reduction method.
In this example, only two dimensions are used. however, an arbitrary 
number of dimensions can be evaluated
"""
embedding = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])

"""
List of sample labels (groups/classes).
In this example, only two different groups are used. However, an arbitrary 
number of classes can be evaluated
"""
labels = np.array(['group1', 'group1', 'group1', 'group1', 'group2', 'group2', 'group2', 'group2'])

"""
List of positive samples.
Depending on the study, positive classes are usually ranked as 
the labels for which a particular prediction is desired. 

For instance: 
- sick patients (positive class) versus controls (negative class)
- burnout (positive class), depression (positive class), versus control (negative class)
 
If you are not sure which are your positive classes, then omit this input and the 
algorithm will take the groups with the lower number of samples as positive
"""
positives = np.array(['group1'])

"""
Base approach for defining the groups' centroids.

Available options are:
- mean
- median [default]
- mode
"""
center_formula = 'median'

# Group separability evaluation
psi_p, psi_roc, psi_pr, psi_mcc = indices.compute_psis(embedding, labels, positives, center_formula)

print(psi_p)
print(psi_roc)
print(psi_pr)
print(psi_mcc)
```

### Computing null model

```python
from sklearn.datasets import load_iris
from psis import indices

# Sample data. Details at: https://scikit-learn.org/stable/datasets/toy_dataset.html
data = load_iris()

# Number of iteration for the Null model
iterations = 50

# Random seed (for reproducibility)
seed = 10

# Group separability evaluation.
# In this example, the evaluation of group separability is directly
# assessed in the High-Dimensional (HD) space
results = indices.compute_null_model(data.data, data.target, iterations=iterations, seed=seed)

# Accessing the results
# In this example only 'psi_roc' is evaluated. The other indices' results can be 
# accessed in the same way
print(results['psi_roc']['value']) # Initial index value
print(results['psi_roc']['min']) # Minimum permuted value
print(results['psi_roc']['max']) # Maximum permuted value
print(results['psi_roc']['p_value']) # Separability significance (p-value)
```

## Issues

Please, report any issue at [psis/issues](https://github.com/biomedical-cybernetics/pypsis/issues)