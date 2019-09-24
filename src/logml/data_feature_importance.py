#!/usr/bin/env python

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import subprocess
import warnings

from IPython.core.display import Image, display
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_graphviz

from .config import CONFIG_DATASET_FEATURE_IMPORTANCE
from .files import MlFiles
from .feature_importance import FeatureImportance


class DataFeatureImportance(MlFiles):
    '''
    Perform feature importance / feature selection analysis
    '''

    def __init__(self, datasets, config, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_FEATURE_IMPORTANCE)
        self.datasets = datasets
        self.model_type = model_type
        if set_config:
            self._set_from_config()

    def __call__(self):
        ''' Feature importance '''
        if not self.enable:
            self._info(f"Dataset feature importance / feature selection disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        self._info("Feature importance / feature selection (model_type={self.model_type}): Start")
        self.x, self.y = self.datasets.get_train_xy()
        self.feature_importance(self.random_forest(), 'RandomForest')
        self.feature_importance(self.extra_trees(), 'ExtraTrees')
        self.feature_importance(self.gradient_boosting(), 'GradientBoosting')
        self.tree_graph()
        self._info("Feature importance / feature selection: End")
        return True

    def extra_trees(self, n_estimators=100):
        ''' Create a ExtraTrees model '''
        if self.model_type == 'regression':
            m = ExtraTreesRegressor(n_estimators=n_estimators)
        elif self.model_type == 'classification':
            m = ExtraTreesClassifier(n_estimators=n_estimators)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def gradient_boosting(self):
        ''' Create a ExtraTrees model '''
        if self.model_type == 'regression':
            m = GradientBoostingRegressor()
        elif self.model_type == 'classification':
            m = GradientBoostingClassifier()
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def feature_importance(self, model, model_name):
        """ Feature importance analysis """
        self._info(f"Feature importance based on {model_name}")
        self.feature_importance_model(model, model_name)
        fi = FeatureImportance(model, model_name, self.x, self.y)
        if not fi():
            self._info("Could not analyze feature importance using RandomForest")
        fi.plot()
        return True

    def feature_importance_model(self, model, model_name):
        ''' Show model built-in feature importance '''
        field_name = f"importance_{model_name}"
        imp_df = pd.DataFrame({field_name: model.feature_importances_}, index=self.x.columns)
        imp_df.sort_values(by=[field_name], ascending=False, inplace=True)
        display(imp_df)
        return True

    def random_forest(self, n_estimators=100, max_depth=None, bootstrap=True):
        ''' Create a RandomForest model '''
        if self.model_type == 'regression':
            m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        elif self.model_type == 'classification':
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def tree_graph(self, max_depth=3, file_dot='tree.dot', file_png='tree.png'):
        """ Simple tree representation """
        # Train a single tree with all the samples
        m = self.random_forest(n_estimators=1, max_depth=max_depth, bootstrap=False)
        # Export the tree to a graphviz 'dot' format
        str_tree = export_graphviz(m.estimators_[0],
                                   out_file='tree.dot',
                                   feature_names=self.x.columns,
                                   filled=True,
                                   rounded=True)
        print(f"str_tree={str_tree}")
        print(f"Created dot file: '{file_dot}'")
        # Convert 'dot' to 'png'
        args = ['dot', '-Tpng', file_dot, '-o', 'tree.png']
        subprocess.run(args)
        print(f"Created image: '{file_png}'")
        display(Image(filename=file_png))
