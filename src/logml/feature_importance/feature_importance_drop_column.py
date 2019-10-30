

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles


class FeatureImportanceDropColumn(MlFiles):
    '''
    Estimate feature importance based on a model.
    How it works: Drops a single column, re-train and analyze how model performance
    is degraded (respect to validation dataset). Most important features will
    make the model perform much worse when dropped, unimportant features will
    not affect performance
    '''

    def __init__(self, model, model_name, x_train, y_train, x_val, y_val):
        self.model = model
        self.model_name = model_name
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.performance = dict()
        self.importance = None
        self.verbose = False

    def __call__(self):
        # Base performance
        self._debug(f"Feature importance (drop-column): Start")
        try:
            score_base = self.loss(self.x_train, self.x_val)
            # Shuffle each column
            perf = list()
            cols = list(self.x_train.columns)
            cols_count = len(cols)
            for i in range(cols_count):
                c = cols[i]
                # Delete column 'c'
                x_copy = self.x_train.drop(c, axis=1)
                x_val_copy = self.x_val.drop(c, axis=1)
                # How did it perform
                score_xi = self.loss(x_copy, x_val_copy)
                # Performance is the score dofference respect to score_base
                perf_c = score_base - score_xi
                self._debug(f"Column {i} / {cols_count}, column name '{c}', performance {perf_c}")
                perf.append(perf_c)
                self.performance[c] = perf_c
            # List of items sorted by importance (most important first)
            self.importance = sorted(self.performance.items(), key=lambda kv: kv[1], reverse=True)
            perf_array = np.array(perf)
            self.performance_norm = perf_array / score_base if score_base > 0.0 else perf_array
        except Exception as e:
            self._error(f"Feature importance (drop-column): Exception '{e}'\n{traceback.format_exc()}")
            return False
        self._debug(f"Feature importance (drop-column): End")
        return True

    def plot(self, x=None):
        " Plot importance distributions "
        imp_x = np.array([f[0] for f in self.importance])
        imp_y = np.array([f[1] for f in self.importance])
        # Show bar plot
        fig = plt.figure()
        plt.barh(imp_x, imp_y)
        self._plot_show(f"Feature importance (drop-column) {self.model_name}", 'dataset_feature_importance_dropcolumn', fig, count_vars_y=len(self.performance))
        # Plot performance histogram
        fig = plt.figure()
        sns.distplot(self.performance_norm)
        self._plot_show(f"Feature importance (drop-column) {self.model_name}: Performance histogram", 'dataset_feature_importance_dropcolumn_histo', fig)

    def loss(self, x_train, x_val):
        """ Create a new model, train on 'x', calculate the loss on the validation x_val """
        model = clone(self.model)
        model.fit(x_train, self.y_train)
        return model.score(x_val, self.y_val)
