import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns

from ..core.files import MlFiles
from ..core.scatter_gather import scatter, scatter_all, gather
from ..util.etc import array_to_str


class FeatureImportanceModel(MlFiles):
    """
    Estimate feature importance based on a model.
    """

    def __init__(self, model_factory, rand_columns, num_iterations):
        self.model_factory = model_factory
        self.model = None
        self.model_type = model_factory.model_type
        self.datasets = model_factory.datasets
        self.rand_columns = rand_columns
        self.num_iterations = num_iterations
        self.loss_base = None
        self.performance = dict()
        self.importance_name = ''
        self.importances = None
        self.init_new_model_force = False
        self.verbose = False
        self.is_cv = False

    def __call__(self):
        if self.num_iterations < 1:
            self._fatal_error("Number of iterations should be an integer number greater than 0: num_iterations={self.num_iterations}")
        # Base performance
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Start")
        self.initialize()
        self.loss_base = self._loss_base()
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Base loss = {array_to_str(self.loss_base)}")
        # Shuffle each column
        self.columns = self.datasets.get_input_names()
        cols_count = len(self.columns)
        # fi_sk = self.model.get_feature_importances()
        for i, c in enumerate(self.columns):
            # Only estimate importance of input variables
            if c not in self.datasets.outputs:
                try:
                    perf, perf_indx = self.performances(c)
                except TypeError:
                    perf = None
                if perf is not None:
                    self.performance[c] = perf, perf_indx
                    self._info(f"Feature importance ({self.importance_name}, {self.model_type}): Column {i} / {cols_count}, column name '{c}', performances: {perf}")
        self.importances = self._importances()
        self.pvalues = self._pvalues()
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return self.importances

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

    def get_performances(self, column_name):
        """Prepare performance detailed data per column for detailed performance table"""
        column_indexes, column_values = list(), list()
        for performance_values in self.performance.values():
            perf_values, perf_indexes = performance_values

            # union lists of column indexes in one list and add column_name to each column index
            # ('x1', 'Iteration_1') -> ('x2', 'Iteration_1', 'importance_permutation')
            column_indexes += [indx + (column_name, ) for indx in perf_indexes]
            # union lists of column values in one list
            try:
                flat_list = [item for sublist in perf_values for item in sublist]
                column_values += flat_list
            except Exception:
                column_values += perf_values

        multi_indexes = pd.MultiIndex.from_tuples(column_indexes)
        return pd.Series(column_values, index=multi_indexes).unstack()

    def get_weight(self):
        """ Weight used when combining different models for feature importance """
        return self.loss_base.ravel().mean() if self.is_cv else self.loss_base

    def _importance(self, col_name):
        """ Calculate one feature importance, for column col_name """
        perf, perf_indx = self.performance[col_name]
        results = np.array(perf).ravel()
        return results.mean()

    @gather
    def _importances(self):
        """ Calculate all feature importances, based on performance results """
        return [self._importance(c) for c in self.columns]

    def initialize(self):
        is_cv = self.initialize_model()
        if is_cv is not None:
            self.is_cv = is_cv
        self._debug(f"Feature importance ({self.importance_name}: Initialize, is_cv={self.is_cv}")

    @scatter_all
    def initialize_model(self):
        self.model = self.model_factory.get(force=self.init_new_model_force)
        return self.model.is_cv

    def losses(self, column_name):
        """
        Calculate loss and perfor values after changing the dataset for 'column_name'.
        Repeat 'num_iterations' and store results.
        """
        loss, perf, perf_indx = list(), list(), list()
        for i in range(self.num_iterations):
            # Change dataset, evaluate performance, restore original dataset
            ori = self.dataset_change(column_name)
            loss_i = self.loss()
            self.dataset_restore(column_name, ori)
            # Performance is the loss difference respect to self.loss_base
            # (the higher the loss difference, the more important the variable)
            # Note that loss can be an array (in case of cross-validation), so perf_i can be an array too
            perf_i = loss_i - self.loss_base
            self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column '{column_name}', is_cv: {self.is_cv}, iteration {i+1} / {self.num_iterations}, losses: {array_to_str(loss_i)}, performance: {array_to_str(perf_i)}")

            perf_indx_i = self._save_performance_data_for_detailed_table(i, perf_i, column_name)
            for indx in perf_indx_i:
                perf_indx.append(indx)

            perf.append(perf_i)
            loss.append(perf_i)
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column '{column_name}', losses: {array_to_str(loss)}, performance: {array_to_str(np.array(perf))}")
        return loss, perf, perf_indx

    def _save_performance_data_for_detailed_table(self, iteration_number, perf_i, column_name):
        """Collect table indexes and performance values for each cross-validation"""
        indxs = list()
        if self.is_cv:
            for cv_number, cv_perfm_value in enumerate(perf_i, start=1):
                indxs.append((column_name, f'CV_{cv_number}', f'Iteration_{iteration_number + 1}'))
        else:
            if self.num_iterations <= 1:
                indxs.append((column_name,))
            else:
                indxs.append((column_name, f'Iteration_{iteration_number + 1}'))
        return indxs

    @scatter_all
    def _loss_base(self):
        lb = self.loss(is_base=True)
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Loss base '{lb}")
        return lb

    def loss(self, is_base=False):
        """
        Calculate loss. Re-train model if necesary
        is_base: Indicates if this is the 'base' model loss used for comparison (i.e. self.model)
        Returns: A loss value or multiple loss values if the model uses cross-validation
        """
        raise Exception("Unimplemented!")

    @scatter
    def performances(self, column_name):
        """
        Example returned values:
        loss = [0.00046228115333546427]
        perf = [0.00046228115333546427]
        perf_indx = [('x3', 'Iteration1')]
        """
        loss, perf, perf_indx = self.losses(column_name)
        return perf, perf_indx

    def plot(self):
        """ Plot importance distributions """
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
        perf_values = [performance_value[0] for performance_value in self.performance.values()]
        values = [v for vs in perf_values for v in vs]
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
            results = np.array(self.performance[col_name][0]).ravel()
            u, p = scipy.stats.mannwhitneyu(results, null_values, alternative='greater')
            self._debug(f"P-value '{col_name}' (Mann-Whitney statistic): p-value={p}, U-test={u}, results: {array_to_str(results)}")
            return p
        except ValueError as v:
            self._warning(f"Error calculating Mann-Whitney's U-statistic, column '{col_name}': {v}. Results: {results}")
        return 1.0

    @gather
    def _pvalues(self):
        """ Calculate all pvalues """
        null_values = np.array([v for c in self.rand_columns for v in self.performance[c][0]]).ravel()
        self._debug(f"P-value null-values (Mann-Whitney statistic): {array_to_str(null_values)}")
        return [self.pvalue(c, null_values) for c in self.columns]

