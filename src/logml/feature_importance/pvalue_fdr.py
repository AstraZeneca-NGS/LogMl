
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traceback

from scipy.stats import chi2
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import fdrcorrection

from ..core.files import MlFiles
from ..datasets import InOut


class PvalueFdr(MlFiles):
    '''
    Estimate p-values and correct for FDR
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
        self.algorithm = 'P-values'

    def __call__(self):
        # Base performance
        self._debug(f"{self.algorithm} ({self.tag}): Start, null model variables {self.null_model_variables}")
        if not self.filter_null_variables():
            self._warning(f"{self.algorithm} ({self.tag}): No variables left for null model")
        # Create 'null' model
        if not self.fit_null_model():
            self._error(f"{self.algorithm} ({self.tag}): Could not fit null model, skipping")
            return False
        # Calculate p-values (one per column)
        null_vars = set(self.null_model_variables)
        cols_count = len(self.columns)
        for i in range(cols_count):
            c = self.columns[i]
            if c in null_vars:
                self._debug(f"{self.algorithm} ({self.tag}): Null variable '{c}', skipped")
                continue
            if c in self.datasets.outputs:
                self._debug(f"{self.algorithm} ({self.tag}): Output variable '{c}', skipped")
                continue
            self.p_values[c] = self.p_value(c)
            self._info(f"{self.algorithm} ({self.tag}): Column {i} / {cols_count}, '{c}', pvalue: {self.p_values[c]}")
        self.fdr()
        return len(self.p_values) > 0

    def fdr(self):
        """ Perform multiple testing correction using FDR """
        self.rejected, self.p_values_corrected = fdrcorrection(self.get_pvalues())

    def filter_null_variables(self):
        '''
        Filter null model variables, only keep the ones in the dataset
        Returns:
            True if there are remaining variables or the initial list was empty
            False if, after filtering, there are no remaining variables in the list
        '''
        if len(self.null_model_variables) == 0:
            return True
        x_vars = set(self.x.columns)
        null_vars = list()
        for c in self.null_model_variables:
            if c in x_vars:
                null_vars.append(c)
            else:
                self._info("Variable '{c}' does not exists in dataset, ommiting")
        self.null_model_variables = null_vars
        return len(null_vars) > 0

    def _drop_na_inf(self, cols):
        ''' Remove 'na' and 'inf' values from x '''
        x_cols = self.x[cols]
        keep = ~(pd.isna(x_cols.replace([np.inf, -np.inf], np.nan)).any(axis=1).values)
        x, y = x_cols[keep].copy(), self.y[keep].copy()
        return x, y

    def fit_null_model(self):
        ''' Fit null model '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def get_pvalues(self):
        """ Get all p-values as a vector """
        return np.array([self.p_values.get(c, 1.0) for c in self.columns])

    def p_value(self, col):
        """ Calculate the p-value using column 'col' """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")


class LogisticRegressionWilks(PvalueFdr):
    '''
    Estimate p-value from logistic regression (Wilks)
    '''

    def __init__(self, datasets, null_model_variables, tag):
        super().__init__(datasets, null_model_variables, tag)
        self.algorithm = 'Logistic regression Wilks'

    def fit_null_model(self):
        ''' Fit 'null' model '''
        self.model_null, self.model_null_results = self.model_fit()
        if self.model_null is None:
            self._error(f"{self.algorithm} ({self.tag}): Could not fit null model, skipping")
            return False
        return True

    def model_fit(self, alt_model_variables=None):
        """ Fit a model using 'null_model_variables' + 'alt_model_variables' """
        try:
            cols = list(self.null_model_variables)
            if alt_model_variables:
                cols.append(alt_model_variables)
            x, y = self._drop_na_inf(cols)
            logit_model = Logit(y, x)
            res = logit_model.fit(disp=0)
            return logit_model, res
        except np.linalg.LinAlgError as e:
            self._error(f"{self.algorithm}: Linear Algebra exception.\nException: {e}\n{traceback.format_exc()}")
            return None, None

    def p_value(self, col):
        """ Calculate the p-value using column 'col' """
        model_alt, model_alt_res = self.model_fit(col)
        if model_alt is None:
            self._error(f"{self.algorithm} ({self.tag}): Could not fit alt model for column/s {col}, returning p-value=1.0")
            return 1.0
        d = 2.0 * (model_alt_res.llf - self.model_null_results.llf)
        p_value = chi2.sf(d, 1)
        self._debug(f"{self.algorithm} ({self.tag}): Columns {col}, log-likelihood null: {self.model_null_results.llf}, log-likelihood alt: {model_alt_res.llf}, p_value: {p_value}")
        return p_value


class PvalueLinear(PvalueFdr):
    '''
    Estimate p-values based on a linear model
    '''

    def __init__(self, datasets, null_model_variables, tag):
        super().__init__(datasets, null_model_variables, tag)
        self.algorithm = 'Linear regression p-value'

    def fit_null_model(self):
        return True

    def p_value(self, col):
        """ Calculate the p-value using column 'col' """
        cols = list(self.null_model_variables)
        cols.append(col)
        x, y = self._drop_na_inf(cols)
        res = OLS(endog=y, exog=x).fit()
        return res.pvalues.loc[col]
