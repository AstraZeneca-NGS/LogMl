
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns

from ..core.files import MlFiles
from ..util.etc import array_to_str


class FeatureImportanceModel(MlFiles):
    """
    Estimate feature importance based on a model.
    """

    def __init__(self, model, model_type, rand_columns, num_iterations, scatter):
        self.model = model.clone()
        self.model_type = model_type
        self.datasets = model.datasets
        self.rand_columns = rand_columns
        self.num_iterations = num_iterations
        self.loss_base = None
        self.performance = dict()
        self.importance_name = ''
        self.importances = None
        self.verbose = False
        self.is_cv = self.model.is_cv
        self.scatter = scatter

    def calc_importances(self):
        """
        Calculate all feature importances, based on performance results
        """
        # Calculate importance based an all results
        null_values = np.array([v for c in self.rand_columns for v in self.performance[c]]).ravel()
        self._debug(f"P-value null-values (Mann-Whitney statistic): {array_to_str(null_values)}")
        self.importances = [self._calc_importance(c) for c in self.columns]
        self.pvalues = [self.pvalue(c, null_values) for c in self.columns]
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def _calc_importance(self, col_name):
        """ Calculate one feature importance, for column col_name """
        results = np.array(self.performance[col_name]).ravel()
        return results.mean()

    def __call__(self):
        if self.num_iterations < 1:
            self._fatal_error("Number of iterations should be an integer number greater than 0: num_iterations={self.num_iterations}")
        # Base performance
        self.scatter.set_subsection(f"model.{self.importance_name}.{self.model_type}")
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Start")
        self.initialize()
        self.loss_base = self.loss(is_base=True)
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Base loss = {array_to_str(self.loss_base)}")
        # Shuffle each column
        self.columns = self.datasets.get_input_names()
        cols_count = len(self.columns)
        fi_sk = self.model.get_feature_importances()
        for i, c in enumerate(self.columns):
            if not self.scatter.should_run():
                continue
            self._info(f"Feature importance ({self.importance_name}, {self.model_type}): Column {i} / {cols_count}, column name '{c}', raw importance: {fi_sk[i]}")
            # Only estimate importance of input variables
            if c not in self.datasets.outputs:
                self.losses(c)
        self._error("SAVE PARTIAL RESULTS")
        self._error("MOVE THIS TO GATHER STEP")
        # self.calc_importances()
        # self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def dataset_change(self, col_name):
        """ Change datasets for column 'col_name' """
        raise Exception("Unimplemented!")

    def dataset_restore(self, col_name, col_ori):
        """ Restore column 'col_name' using values from col_ori """
        raise Exception("Unimplemented!")

    def get_importances(self):
        return pd.Series(self.importances, index=self.columns)

    def get_pvalues(self):
        return pd.Series(self.pvalues, index=self.columns)

    def get_weight(self):
        """ Weight used when combinig different models for feature importance """
        return self.loss_base.ravel().mean() if self.is_cv else self.loss_base

    def initialize(self):
        pass

    def losses(self, column_name):
        """
        Calculate loss after changing the dataset for 'column_name'.
        Repeat 'num_iterations' and store results.
        """
        perf, loss = list(), list()
        for i in range(self.num_iterations):
            # Change dataset, evaluate performance, restore original dataset
            ori = self.dataset_change(column_name)
            loss_i = self.loss()
            self.dataset_restore(column_name, ori)
            # Performance is the loss difference respect to self.loss_base
            # (the higher the loss difference, the more important the variable)
            # Note that loss can be an array (in case of cross-validation), so perf_i can be an array too
            perf_i = loss_i - self.loss_base
            self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column '{column_name}', iteration {i+1} / {self.num_iterations}, losses: {array_to_str(loss_i)}, performance: {array_to_str(perf_i)}")
            perf.append(perf_i)
            loss.append(perf_i)
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column '{column_name}', losses: {array_to_str(loss)}, performance: {array_to_str(np.array(perf))}")
        self.performance[column_name] = perf

    def loss(self, is_base=False):
        """
        Calculate loss. Re-train model if necesary
        is_base: Indicates if this is the 'base' model loss used for comparisson (i.e. self.model)
        Returns: A loss value or multiple loss values if the model uses cross-validation
        """
        raise Exception("Unimplemented!")

    def plot(self):
        " Plot importance distributions "
        names = np.array([self.columns])
        imp = np.array(self.importances)
        # Show bar plot
        fig = plt.figure()
        y_pos = np.arange(len(imp))
        try:
            plt.barh(y_pos, imp, tick_label=self.columns)
            self._plot_show(f"Feature importance {self.importance_name}: {self.model_type}", 'dataset_feature_importance_dropcolumn', fig, count_vars_y=len(self.performance))
        except Exception as e:
            self._error(f"Feature importance {self.importance_name}, {self.model_type}: Exception trying to bar plot, exception: {e}, y_pos: {y_pos}, importances: {imp}")
        # Plot performance histogram
        values = [v for vs in self.performance.values() for v in vs]
        values = np.array(values).ravel()
        fig = plt.figure()
        try:
            # Sometimes the distribution cannot be created (e.g. an exception is thrown when calculating the kernel density)
            sns.distplot(values)
            self._plot_show(f"Feature importance {self.importance_name}: {self.model_type}: Performance histogram", 'dataset_feature_importance_dropcolumn_histo', fig)
        except Exception as e:
            self._error(f"Feature importance {self.importance_name}, {self.model_type}: Exception while trying to create distribution plot, exception: {e}, values to plot: {values}")

    def pvalue(self, col_name, null_values):
        """
        Calculate p-values based on a Mann-Whitney U-statistic
        Test if loss values for input 'col_name' are higher than the ones for
        the 'null_model' (randomly shuffled variables). If the p-value is
        low, it might indicate that the variable is 'important'
        """
        if len(null_values) == 0:
            return np.nan
        if col_name in self.rand_columns:
            return 1.0
        try:
            results = np.array(self.performance[col_name]).ravel()
            u, p = scipy.stats.mannwhitneyu(results, null_values, alternative='greater')
            self._debug(f"P-value '{col_name}' (Mann-Whitney statistic): p-value={p}, U-test={u}, results: {array_to_str(results)}")
            return p
        except ValueError as v:
            self._warning(f"Error calculating Mann-Whitney's U-statistic, column '{col_name}': {v}. Results: {results}")
        return 1.0
