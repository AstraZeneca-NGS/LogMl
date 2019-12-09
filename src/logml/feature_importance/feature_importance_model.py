
import math
import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, model, model_type):
        self.model = model.clone()
        self.model_type = model_type
        self.datasets = model.datasets
        self.loss_base = None
        self.performance = dict()
        self.importance_name = ''
        self.importance = None
        self.verbose = False

    def __call__(self):
        # Base performance
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Start")
        self.initialize()
        self.loss_base = self.loss()
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Base loss = {self.loss_base}")
        # Shuffle each column
        perf = list()
        cols = list(self.datasets.get_input_names())
        cols_count = len(cols)
        for i in range(cols_count):
            c = cols[i]
            # Only estimate importance of input variables
            if c in self.datasets.outputs:
                continue
            # Change dataset, evaluate performance, restore originla dataset
            ori = self.dataset_change(c)
            loss_c = self.loss()
            self.dataset_restore(c, ori)
            # Performance is the loss difference respect to self.loss_base
            # (the higher the loss, the more important the variable)
            perf_c = loss_c - self.loss_base
            self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Column {i} / {cols_count}, column name '{c}', performance {perf_c}")
            perf.append(perf_c)
            self.performance[c] = perf_c
        # List of items sorted by importance (most important first)
        self.importance = sorted(self.performance.items(), key=lambda kv: kv[1], reverse=True)
        perf_array = np.array(perf)
        self.performance_norm = perf_array / self.loss_base if self.loss_base > 0.0 else perf_array
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): End")
        return True

    def dataset_change(self, col_name):
        """ Change datasets for column 'col_name' """
        raise Exception("Unimplemented!")

    def dataset_restore(self, col_name, col_ori):
        """ Restore column 'col_name' using values from col_ori """
        raise Exception("Unimplemented!")

    def initialize(self):
        pass

    def loss(self):
        """ Train (if necesary) and calculate loss """
        raise Exception("Unimplemented!")

    def plot(self, x=None):
        " Plot importance distributions "
        imp_x = np.array([f[0] for f in self.importance])
        imp_y = np.array([f[1] for f in self.importance])
        # Show bar plot
        fig = plt.figure()
        plt.barh(imp_x, imp_y)
        self._plot_show(f"Feature importance {self.importance_name}: {self.model_type}", 'dataset_feature_importance_dropcolumn', fig, count_vars_y=len(self.performance))
        # Plot performance histogram
        fig = plt.figure()
        sns.distplot(self.performance_norm)
        self._plot_show(f"Feature importance {self.importance_name}: {self.model_type}: Performance histogram", 'dataset_feature_importance_dropcolumn_histo', fig)
