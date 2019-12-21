
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from scipy.stats import chi2
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.multitest import fdrcorrection

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
        self.columns = list(self.x.columns)
        self.loss_base = None
        self.model_null = None
        self.model_null_results = None
        self.p_values = dict()  # Dictionary of 'raw' p-values
        self.p_values_corrected = None  # Array of FDR-corrected p-values (sorted by 'columns')
        self.rejected = None  # Arrays of bool signaling 'rejected' null hypothesis for FDR-corrected p-values

    def __call__(self):
        # Base performance
        self._debug(f"Logistic regression Wilks ({self.tag}): Start, null model variables {self.null_model_variables}")
        # Fit 'null' model
        self.model_null, self.model_null_results = self.model_fit()
        if self.model_null is None:
            self._error(f"Logistic regression Wilks ({self.tag}): Could not fit null model, skipping")
            return False
        # Create 'alt' models (one per column)
        null_vars = set(self.null_model_variables)
        cols_count = len(self.columns)
        for i in range(cols_count):
            c = self.columns[i]
            if c in null_vars:
                self._debug(f"Logistic regression Wilks ({self.tag}): Null variable '{c}', skipped")
                continue
            if c in self.datasets.outputs:
                self._debug(f"Logistic regression Wilks ({self.tag}): Output variable '{c}', skipped")
                continue
            self.p_values[c] = self.p_value(c)
            self._info(f"Logistic regression Wilks ({self.tag}): Column {i} / {cols_count}, '{c}', pvalue: {self.p_values[c]}")
        self.fdr()
        return len(self.p_values) > 0

    def fdr(self):
        """ Perform multiple testing correction using FDR """
        self.rejected, self.p_values_corrected = fdrcorrection(self.get_pvalues())

    def get_pvalues(self):
        """ Get all p-values as a vector """
        return np.array([self.p_values.get(c, 1.0) for c in self.columns])

    def model_fit(self, alt_model_variables=None):
        """ Fit a model using 'null_model_variables' + 'alt_model_variables' """
        try:
            cols = list(self.null_model_variables)
            if alt_model_variables:
                cols.append(alt_model_variables)
            x = self.x[cols]
            logit_model = Logit(self.y, x)
            res = logit_model.fit(disp=0)
            return logit_model, res
        except np.linalg.LinAlgError as e:
            self._error(f"Logistic regression (Wilks): Linear Algebra exception.\nException: {e}\n{traceback.format_exc()}")
            return None, None

    def p_value(self, cols):
        """ Calculate the p-value using column 'cols' """
        model_alt, model_alt_res = self.model_fit(cols)
        if model_alt is None:
            self._error(f"Logistic regression Wilks ({self.tag}): Could not fit alt model for column/s {cols}, returning p-value=1.0")
            return 1.0
        d = 2.0 * (model_alt_res.llf - self.model_null_results.llf)
        p_value = chi2.sf(d, 1)
        self._debug(f"Logistic regression Wilks ({self.tag}): Columns {cols}, log-likelihood null: {self.model_null_results.llf}, log-likelihood alt: {model_alt_res.llf}, p_value: {p_value}")
        return p_value
