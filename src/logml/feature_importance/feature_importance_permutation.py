
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import traceback

from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles
from .feature_importance_model import FeatureImportanceModel


class FeatureImportancePermutation(FeatureImportanceModel):
    '''
    Estimate feature importance based on a model.
    How it works: Suffle a column and analyze how model performance is
    degraded. Most important features will make the model perform much
    worse when shuffled, unimportant features will not affect performance

    To estimate a p-value, it uses a ranked test by comparing to resutls from
    randomly shuffled columns
    '''

    def __init__(self, model, model_name, rand_columns, num_iterations=1):
        super().__init__(model, model_name, num_iterations)
        self.importance_name = 'permutation'
        self.rand_columns = rand_columns

    def calc_importances(self):
        """
        Calculate all feature importances, based on performance results
        """
        # Calculate importance based an all results
        rand_cols_set = set(self.rand_columns)
        null_values = np.array([v for c in self.rand_columns for v in self.performance[c]])
        self.importances = [self._calc_importance(c) for c in self.columns]
        self.pvalues = [self.pvalue(c, null_values) for c in self.columns]
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def _calc_importance(self, col_name):
        """ Calculate one feature importance, for column col_name """
        results = np.array(self.performance[col_name])
        return results.mean()

    def dataset_change(self, col_name):
        """ Change datasets for column 'col_name' """
        col_ori = self.datasets.shuffle_input(col_name)
        return col_ori

    def dataset_restore(self, col_name, col_ori):
        """ Restore column 'col_name' using values 'col_ori' """
        self.datasets.shuffle_input(col_name, col_ori)

    def get_pvalues(self):
        return pd.Series(self.pvalues, index=self.columns)

    def initialize(self):
        """ Initialzie the model (the model is trained only once) """
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Initialize. Model fit")
        self.model.model_train()
        self._error(f"DF: {self.datasets.dataset}")

    def loss(self):
        """
        Train (if necesary) and calculate loss
        In this case, there is no training, just evaluate the loss
        """
        self.model.model_eval_validate()
        return self.model.eval_validate

    def pvalue(self, col_name, null_values):
        if col_name in self.rand_columns:
            return 1.0
        try:
            results = np.array(self.performance[col_name])
            u, p = scipy.stats.mannwhitneyu(results, null_values, alternative='greater')
            self._debug(f"Mann-Whitney statistic '{col_name}': p-value={p}, U-test={u}, results: {results}")
            return p
        except ValueError as v:
            self._warning(f"Error calculating Mann-Whitney's U-statistic, column '{col_name}': {v}. Results: {results}")
        return 1.0
