
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traceback

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles
from ..datasets import InOut


class FeatureImportanceModel(MlFiles):
    '''
    Estimate feature importance based on a model.
    '''

    def __init__(self, model, model_type, num_iterations=1):
        self.model = model.clone()
        self.model_type = model_type
        self.datasets = model.datasets
        self.num_iterations = num_iterations
        self.loss_base = None
        self.performance = dict()
        self.importance_name = ''
        self.importances = None
        self.verbose = False

    def calc_importances(self):
        """
        Calculate all feature importances, based on performance results
        """
        # Calculate importance based an all results
        self.importances = [self._calc_importance(c) for c in self.columns]
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def _calc_importance(self, col_name):
        """ Calculate one feature importance, for column col_name """
        results = self.performance[col_name]
        return np.array(results).mean()

    def __call__(self):
        # Base performance
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Start")
        self.initialize()
        self.loss_base = self.loss()
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Base loss = {self.loss_base}")
        # Shuffle each column
        self.columns = self.datasets.get_input_names()
        cols_count = len(self.columns)
        for i, c in enumerate(self.columns):
            self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column {i} / {cols_count}, column name '{c}'")
            # Only estimate importance of input variables
            if c not in self.datasets.outputs:
                self.losses(c)
        self.calc_importances()
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def dataset_change(self, col_name):
        """ Change datasets for column 'col_name' """
        raise Exception("Unimplemented!")

    def dataset_restore(self, col_name, col_ori):
        """ Restore column 'col_name' using values from col_ori """
        raise Exception("Unimplemented!")

    def get_importances(self):
        return pd.Series(self.importances, index=self.columns)

    def initialize(self):
        pass

    def losses(self, column_name):
        """
        Calculate loss after changing the dataset for 'column_name'.
        Repeat 'num_iterations' and store results.
        """
        perf = list()
        for i in range(self.num_iterations):
            # Change dataset, evaluate performance, restore originl dataset
            ori = self.dataset_change(column_name)
            loss = self.loss()
            self.dataset_restore(column_name, ori)
            # Performance is the loss difference respect to self.loss_base
            # (the higher the loss, the more important the variable)
            perf_i = loss - self.loss_base
            self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column '{column_name}', iteration {i} / {self.num_iterations}, performance {perf_i}")
            perf.append(perf_i)
        self.performance[column_name] = perf

    def loss(self):
        """ Calculate loss. Re-train model if necesary """
        raise Exception("Unimplemented!")

    def plot(self, x=None):
        " Plot importance distributions "
        names = np.array([self.columns])
        imp = np.array(self.importances)
        self._error(f"IMP: {imp}\nNAMES: {names}")
        # Show bar plot
        fig = plt.figure()
        y_pos = np.arange(len(imp))
        plt.barh(y_pos, imp, tick_label=self.columns)
        self._plot_show(f"Feature importance {self.importance_name}: {self.model_type}", 'dataset_feature_importance_dropcolumn', fig, count_vars_y=len(self.performance))
        # Plot performance histogram
        fig = plt.figure()
        values = [v for vs in self.performance.values() for v in vs]
        values = np.array(values)
        sns.distplot(values)
        self._plot_show(f"Feature importance {self.importance_name}: {self.model_type}: Performance histogram", 'dataset_feature_importance_dropcolumn_histo', fig)
