
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

    def __init__(self, model):
        self.model = model.clone()
        self.model_name = model.model_name
        self.x_train, self.y_train = model.datasets.get_train_xy()
        self.x_val, self.y_val = model.datasets.get_validate_xy()
        self.loss_base = None
        self.performance = dict()
        self.importance_name = ''
        self.importance = None
        self.verbose = False

    def __call__(self):
        # Base performance
        self._debug(f"Feature importance ({self.importance_name}): Start")
        self.initialize()
        self.loss_base = self.loss(self.x_train, self.y_train, self.x_val, self.y_val)
        # Shuffle each column
        perf = list()
        cols = list(self.x_train.columns)
        cols_count = len(cols)
        for i in range(cols_count):
            c = cols[i]
            # Change dataset
            x_train, y_train, x_val, y_val = self.change_dataset(c)
            # Performance after modifying column 'c'
            loss_c = self.loss(x_train, y_train, x_val, y_val)
            # Performance is the loss difference respect to self.loss_base
            # (the higher the loss, the more important the variable)
            perf_c = loss_c - self.loss_base
            self._debug(f"Feature importance ({self.importance_name}): Column {i} / {cols_count}, column name '{c}', performance {perf_c}")
            perf.append(perf_c)
            self.performance[c] = perf_c
        # List of items sorted by importance (most important first)
        self.importance = sorted(self.performance.items(), key=lambda kv: kv[1], reverse=True)
        perf_array = np.array(perf)
        self.performance_norm = perf_array / self.loss_base if self.loss_base > 0.0 else perf_array
        self._debug(f"Feature importance ({self.importance_name}): End")
        return True

    def change_dataset(self, col):
        """ Change datasets for column 'col' """
        raise Exception("Unimplemented!")

    def initialize(self):
        pass

    def loss(self, x_train, y_train, x_val, y_val):
        """ Train (if necesary) and calculate loss """
        raise Exception("Unimplemented!")

    def model_set_datasets(self, model, x_train, y_train, x_val, y_val):
        """ Create a model clone, update datasets """
        # Update datasets
        model.datasets.dataset_train_xy = InOut(x_train, y_train)
        model.datasets.dataset_validate_xy = InOut(x_val, y_val)

    def plot(self, x=None):
        " Plot importance distributions "
        imp_x = np.array([f[0] for f in self.importance])
        imp_y = np.array([f[1] for f in self.importance])
        # Show bar plot
        fig = plt.figure()
        plt.barh(imp_x, imp_y)
        self._plot_show(f"Feature importance ({self.importance_name}) {self.model_name}", 'dataset_feature_importance_dropcolumn', fig, count_vars_y=len(self.performance))
        # Plot performance histogram
        fig = plt.figure()
        sns.distplot(self.performance_norm)
        self._plot_show(f"Feature importance ({self.importance_name}) {self.model_name}: Performance histogram", 'dataset_feature_importance_dropcolumn_histo', fig)
