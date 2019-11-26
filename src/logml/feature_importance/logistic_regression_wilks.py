
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles
from ..datasets import InOut


class LogisticRegressionWilks(MlFiles):
    '''
    Estimate feature importance based on a model.
    '''

    def __init__(self, datasets, null_model_variables, tag):
        self.datasets = datasets
        self.null_model_variables = null_model_variables
        self.tag = tag
        self.x, self.y = datasets.get_xy()  # Note: We use the full dataset
        self.loss_base = None
        self.p_values = dict()
        self.model_null = None

    def __call__(self):
        # Base performance
        self._debug(f"Logistic regression Wilks ({self.tag}): Start, null model variables {self.null_model_variables}")
        self.model_null = self.model_fit()
        cols = list(self.x_train.columns)
        cols_count = len(cols)
        for i in cols:
            c = cols[i]
            self.p_values[c] = self.p_value(c)
        # List of items sorted by importance (most important first)
        self.importance = sorted(self.performance.items(), key=lambda kv: kv[1], reverse=True)
        perf_array = np.array(perf)
        self.performance_norm = perf_array / self.loss_base if self.loss_base > 0.0 else perf_array
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def change_dataset(self, col):
        """ Change datasets for column 'col' """
        x_train = self.x_train.drop(col, axis=1)
        x_val = self.x_val.drop(col, axis=1)
        return x_train, self.y_train

    def p_value(self, cols):
        """
        Calculate the p-value using column 'cols'
        """
        self._debug(f"Logistic regression Wilks ({self.tag}): Column {c}, p_value {p_value}")
        return 1.0
