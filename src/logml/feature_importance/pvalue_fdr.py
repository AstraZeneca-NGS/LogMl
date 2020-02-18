
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
        self.null_model_required = False
        self.p_values = dict()  # Dictionary of 'raw' p-values
        self.p_values_corrected = None  # Array of FDR-corrected p-values (sorted by 'columns')
        self.rejected = None  # Arrays of bool signaling 'rejected' null hypothesis for FDR-corrected p-values
        self.algorithm = 'P-values'

    def __call__(self):
        # Base performance
        self._debug(f"{self.algorithm} ({self.tag}): Start, null model variables {self.null_model_variables}")
        if not self.filter_null_variables():
            if self.null_model_required:
                self._error(f"{self.algorithm} ({self.tag}): No variables left for null model")
                return False
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
        self.qq_log_plot()
        return len(self.p_values) > 0

    def fdr(self):
        """ Perform multiple testing correction using FDR """
        # Don't use 'nan' values from FDR correction
        pvals = self.get_pvalues()
        isna = np.isnan(pvals)
        # FDR on non-nan values
        p_no_nan = pvals[~isna]
        rej, pvc = fdrcorrection(p_no_nan)
        # Set 'pvalues_corrected' (use 1 for all nan values)
        pvalues_corrected = np.ones(pvals.shape)
        pvalues_corrected[~isna] = pvc
        # Set 'reject' (use 'True' for all nan values)
        reject = np.ones(pvals.shape).astype(bool)
        reject[~isna] = rej
        self.rejected, self.p_values_corrected = reject, pvalues_corrected

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
                self._info(f"{self.algorithm} ({self.tag}): Variable '{c}' does not exists in dataset, ommiting")
        self.null_model_variables = null_vars
        return len(null_vars) > 0

    def _drop_na_inf(self, cols):
        ''' Remove 'na' and 'inf' values from x '''
        x_cols = self.x[cols]
        keep = ~(pd.isna(x_cols.replace([np.inf, -np.inf], np.nan)).any(axis=1).values)
        x, y = x_cols.iloc[keep].copy(), self.y.iloc[keep].copy()
        return x, y

    def fit_null_model(self):
        ''' Fit null model '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def get_pvalues(self):
        """ Get all p-values as a vector """
        return np.array([self.p_values.get(c, np.nan) for c in self.columns])

    def p_value(self, col):
        """ Calculate the p-value using column 'col' """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def qq_log_plot(self):
        """
        QQ plot in log scale.
        Plots the log10 of expected p-values (uniform) vs log10 of
        observed p-values (probs). A 45 degree line is also shown.
        If 'probs' contains values out of [0, 1] interval, an error
        is produced.
        If 'probs' contains 0.0 values, they are replaced by the minimum
        non-zero value in probs, or 0.01/len(probs) (whichever is smaller)
        """
        probs = self.get_pvalues()
        count_oor = ((probs < 0.0) | (probs > 1.0)).sum()
        if count_oor > 0:
            self._error(f"QQ-plot:There are {count_oor} values out of range (less than 0.0 or more than 1.0)")
            return
        count_zero = (probs == 0.0).sum()
        if count_zero > 0:
            min_val = probs[probs > 0.0].min()
            epsilon = 0.01 / len(probs)
            min_val = min(min_val, epsilon)
            self._warning(f"QQ-plot:There are {count_zero} values equal to zero, replacing them by with {min_val}")
            probs[probs == 0.0] = min_val
        # Log10 of expected vs observed (sorted) p-values
        lp = -np.log10(probs)
        lr = -np.log10(np.linspace(1.0 / len(probs), 1.0, num=len(probs)))
        lp.sort()
        lr.sort()
        # Plot values and 45 degree line
        plt.plot(lr, lp, '.')
        plt.plot(lr, lr)
        title = f"QQ-plot: {self.algorithm} - {self.tag}"
        self._plot_show(title, 'qq_plot')


class LogisticRegressionWilks(PvalueFdr):
    '''
    Estimate p-value from logistic regression (Wilks)
    '''

    def __init__(self, datasets, null_model_variables, tag, class_to_analyze=None):
        super().__init__(datasets, null_model_variables, tag)
        self.algorithm = 'Logistic regression Wilks'
        self.class_to_analyze = class_to_analyze
        self.null_model_required = True
        # TODO: REMOVE DEBUGGING CODE
        self._error("SAVE DATASET")
        self.datasets.dataset.to_csv(f"dataset.LogisticRegressionWilks.{tag}.csv")

    def binarize(self, y):
        """ Make sure 'y' has values in range [0, 1]
        Returns: A tuple with < y_normalize, ok>
        Where 'ok' is True on success, False on error
        The value of y_normalized is meaningless if 'ok' is False
        """
        if self.class_to_analyze is None:
            if y.min() < y.max():
                self._debug(f"Converting to [0, 1] range: [{y.min()}, {y.max()}]")
                ybin = (y - y.min()) / (y.max() - y.min())
                return ybin, True
            else:
                self._warning(f"Cannot convert to [0, 1] range: Variable min={y.min()}, max={y.max()}")
                return None, False
        self._debug(f"Converting to binary: Class {self.class_to_analyze}")
        return (y == self.class_to_analyze).astype('float'), True

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
            y, ok = self.binarize(y)
            if not ok:
                return None, None
            # TODO: REMOVE DEBUGING CODE
            if alt_model_variables is not None:
                self._error(f"SAVE DATAFRAME: LOGIT MODEL FIT {alt_model_variables}")
                dflr = x.join(y)
                dflr.to_csv(f"pvalues_fdr_logit_{alt_model_variables}.csv")
            # TODO: REMOVE DEBUGING CODE
            logit_model = Logit(y, x, missing='drop')
            res = self._model_fit(logit_model)
            return logit_model, res
        except np.linalg.LinAlgError as e:
            self._error(f"{self.algorithm}: Could not fit logistic regression model")
            return None, None

    def _model_fit(self, logit_model):
        """ Try to fit the model using different methods """
        methods = ['newton', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg', 'basinhopping']
        for method in methods:
            try:
                res = logit_model.fit(method=method, disp=False)
                if math.isfinite(res.llf):
                    return res
                else:
                    self._debug(f"{self.algorithm}, method '{method}': Log-likekelihood is not a finite number {res.llf} for method {method}.")
            except np.linalg.LinAlgError as e:
                self._debug(f"{self.algorithm}, method '{method}': Linear Algebra exception {e}")  # "\n{traceback.format_exc()}"
            except ValueError as e:
                self._warning(f"{self.algorithm}: ValueError {e}\n{traceback.format_exc()}")
        raise np.linalg.LinAlgError("Could not fit logistic regression using any method")

    def p_value(self, col):
        """ Calculate the p-value using column 'col' """
        model_alt, model_alt_res = self.model_fit(col)
        if model_alt is None:
            self._error(f"{self.algorithm} ({self.tag}): Could not fit alt model for column/s {col}, returning p-value=1.0")
            return np.nan
        d = 2.0 * (model_alt_res.llf - self.model_null_results.llf)
        p_value = chi2.sf(d, 1)
        self._debug(f"{self.algorithm} ({self.tag}): Columns {col}, class={self.class_to_analyze}, log-likelihood null: {self.model_null_results.llf}, log-likelihood alt: {model_alt_res.llf}, p_value: {p_value}")
        return p_value


class MultipleLogisticRegressionWilks(PvalueFdr):
    """ Logistic regression for multiple classes
    This just performs multiple instances of logistic regression. Each "comparisson"
    is testing a class agains the all the others. For instance if the classes
    are: {'low', 'mid', 'high'}, run logstic regression is run three times (one
    for each "comparisson"):
        - 'low' vs ('mid' or 'high')
        - 'mid' vs ('low' or 'high')
        - 'high' vs ('low' or 'mid')
    P-values are be corrected for multiple testing, the lowest p-value for
    each "comparisson" is reported
    """
    def __init__(self, datasets, null_model_variables, tag):
        super().__init__(datasets, null_model_variables, tag)
        self.algorithm = 'Multiple Logistic regression Wilks'
        # Find classes
        self.classes = self.y.replace([np.inf, -np.inf], np.nan).unique()
        # Create a logistic regression for each class
        self.logistic_regressions_by_class = {cn: LogisticRegressionWilks(datasets, null_model_variables, tag, cn) for cn in self.classes}

    def __call__(self):
        oks = [self.logistic_regressions_by_class[cn]() for cn in self.classes]
        self.fdr()
        return len(self.p_values) > 0

    def fdr(self):
        """ Perform multiple testing correction using FDR
        In this case, each 'p_value' calculation returned a list of p-values (one
        for each 'comparisson')
        """
        # FDR Correction using all comparissons
        pvals = [pval for cn in self.classes for pval in self.logistic_regressions_by_class[cn].get_pvalues()]
        pvals = np.array(pvals)
        rejected, pvals_corr = fdrcorrection(pvals)
        # Assign 'raw' p_values: Use across comparissons
        pvals = pvals.reshape((len(self.classes), -1))
        self.p_values = {self.columns[i]: pvals[:, i].min() for i in range(len(self.columns))}
        # Assign FDR corected values: Use minimum across comparissons
        pvals_corr = pvals_corr.reshape((len(self.classes), -1))
        self.p_values_corrected = np.array([pvals_corr[:, i].min() for i in range(len(self.columns))])
        # Assign 'rejected': Use 'or' across comparissons (i.e. 'any')
        rejected = rejected.reshape((len(self.classes), -1))
        self.rejected = np.array([rejected[:, i].any() for i in range(len(self.columns))])


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
