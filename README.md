# Ensemble training with MLP

## Overview

Based on the provided code snippets, here's a summary of what each file appears to do:


* `adaboost-mlp.py`: This script implements a simplified version of the AdaBoost algorithm using multi-layer perceptrons (MLPs) as the weak learners. It trains multiple MLPs with different weights and combines their predictions to make the final prediction.

* `bagging-weight.py`: This script implements a form of bagging (bootstrap aggregating) using decision trees as the base learners. It calculates the accuracy of the weighted bagging approach.


* `bagging-weight-mlp.py`: This script is a variation of the `bagging-weight.py` script, but it uses MLPClassifier as the base learners instead of decision trees.
This implementation works most like `boosting`.

* `bagging-weight-mlp.py`: This script is a variation of the `bagging-weight.py` script, but it uses multi-layer perceptrons (MLPs).

* `gb_mlp.py`: This script implements a form of gradient boosting using MLPs as the base learners. It trains MLPs sequentially to predict the residuals of the ensemble's predictions.


## Extra files

* `data.py`: This file seems to contain functions for generating data, but the actual implementation is not provided in the snippet. It imports necessary libraries like `numpy` and `sklearn`.
