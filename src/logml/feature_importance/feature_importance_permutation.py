

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles


class FeatureImportancePermutation(MlFiles):
    '''
    Estimate feature importance based on a model.
    How it works: Suffle a column and analyze how model performance is
    degraded. Most important features will make the model perform much
    worse when shuffled, unimportant features will not affect performance
    '''

    def __init__(self, model, model_name, x, y):
        self.model = model
        self.model_name = model_name
        self.x = x
        self.y = y
        self.performance = dict()
        self.importance = None
        self.figsize = (16, 10)
        self.verbose = False

    def __call__(self):
        # Base performance
        self._debug(f"Feature importance using permutation: Start")
        x_copy = self.x.copy()
        score_base = self.loss(x_copy)
        # Shuffle each column
        perf = list()
        cols = list(self.x.columns)
        cols_count = len(cols)
        for i in range(cols_count):
            c = cols[i]
            # Shuffle column 'c'
            x_copy = self.x.copy()
            xi = np.random.permutation(x_copy[c])
            x_copy[c] = xi
            # How did it perform
            score_xi = self.loss(x_copy)
            # Performance is the score dofference respect to score_base
            perf_c = score_base - score_xi
            self._debug(f"Column {i} / {cols_count}, column name '{c}', performance {perf_c}")
            perf.append(perf_c)
            self.performance[c] = perf_c
        # List of items sorted by importance (most important first)
        self.importance = sorted(self.performance.items(), key=lambda kv: kv[1], reverse=True)
        perf_array = np.array(perf)
        self.performance_norm = perf_array / score_base if score_base > 0.0 else perf_array
        self._debug(f"Feature importance using permutation: End")
        return True

    def most_important(self, importance_threshold=None, ratio_to_most_important=100, df=None):
        """
        Select features to keep either using an absolute value or
        a ratio to most important feature
        """
        if ratio_to_most_important is not None:
            most_important = fi.importance[0]
            importance_threshold = most_important[1] / ratio_to_most_important

        important_features = [f[0] for f in self.importance if f[1] > importance_threshold]
        unimportant_features = [f[0] for f in self.importance if f[0] not in important_features]
        return important_features, unimportant_features

    def plot(self, x=None):
        " Plot importance distributions "
        imp_x = np.array([f[0] for f in self.importance])
        imp_y = np.array([f[1] for f in self.importance])
        # Show bar plot
        plt.figure(figsize=self.figsize)
        plt.barh(imp_x, imp_y)
        self._plot_show(f"Feature importance {self.model_name}", 'dataset_feature_importance')

    def loss(self, x):
        return self.model.score(x, self.y)

    def __repr__(self):
        return "\n".join([f"{f[0]} : {f[1]}" for f in fi.importance])
