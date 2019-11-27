
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from scipy.stats import chi2
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from statsmodels.discrete.discrete_model import Logit

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
        self.model_null_results = None

    def __call__(self):
        # Base performance
        self._debug(f"Logistic regression Wilks ({self.tag}): Start, null model variables {self.null_model_variables}")
        # Fit 'null' model
        self.model_null, self.model_null_results = self.model_fit()
        # Create 'alt' models (one per column)
        null_vars = set(self.null_model_variables)
        cols = list(self.x.columns)
        cols_count = len(cols)
        for i in range(cols_count):
            c = cols[i]
            self._debug(f"Logistic regression Wilks ({self.tag}): Column {i} / {cols_count}, '{c}'")
            if c in null_vars:
                self._debug(f"Logistic regression Wilks ({self.tag}): Null variable '{c}', skipped")
                continue
            if c in self.datasets.outputs:
                self._debug(f"Logistic regression Wilks ({self.tag}): Output variable '{c}', skipped")
                continue
            self.p_values[c] = self.p_value(c)
        return len(self.p_values) > 0

    def get_pvalues(self):
        """ Get all p-values as a vector """
        return np.array([self.p_values.get(c, 1.0) for c in self.x.columns])

    def model_fit(self, alt_model_variables=None):
        """ Fit a model using 'null_model_variables' + 'alt_model_variables' """
        cols = list(self.null_model_variables)
        if alt_model_variables:
            cols.append(alt_model_variables)
        x = self.x[cols]
        logit_model = Logit(self.y, x)
        res = logit_model.fit()
        return logit_model, res

    def p_value(self, cols):
        """ Calculate the p-value using column 'cols' """
        model_alt, model_alt_res = self.model_fit(cols)
        d = 2.0 * (model_alt_res.llf - self.model_null_results.llf)
        p_value = chi2.sf(d, 1)
        self._debug(f"Logistic regression Wilks ({self.tag}): Columns {cols}, log-likelihood null: {self.model_null_results.llf}, log-likelihood alt: {model_alt_res.llf}, p_value: {p_value}")
        return p_value
