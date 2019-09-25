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
from sklearn.feature_selection import SelectFdr, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsIC
from sklearn.tree import export_graphviz

from .config import CONFIG_DATASET_FEATURE_IMPORTANCE
from .files import MlFiles
from .feature_importance import FeatureImportance

EPSILON = 1.0e-4

class DataFeatureImportance(MlFiles):
    '''
    Perform feature importance / feature selection analysis
    '''

    def __init__(self, datasets, config, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_FEATURE_IMPORTANCE)
        self.datasets = datasets
        self.model_type = model_type
        self.regularization_model_cv = 10
        if set_config:
            self._set_from_config()

    def __call__(self):
        ''' Feature importance '''
        if not self.enable:
            self._info(f"Dataset feature importance / feature selection disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        self._info("Feature importance / feature selection (model_type={self.model_type}): Start")
        self.x, self.y = self.datasets.get_train_xy()
        self.regularization_models()
        # self.select_fdr()
        # self.feature_importance(self.random_forest(), 'RandomForest')
        # self.feature_importance(self.extra_trees(), 'ExtraTrees')
        # self.feature_importance(self.gradient_boosting(), 'GradientBoosting')
        # self.tree_graph()
        self._info("Feature importance / feature selection: End")
        return True

    def extra_trees(self, n_estimators=100):
        ''' Create a ExtraTrees model '''
        if self.is_regression():
            m = ExtraTreesRegressor(n_estimators=n_estimators)
        elif self.is_classification():
            m = ExtraTreesClassifier(n_estimators=n_estimators)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def feature_importance(self, model, model_name):
        """ Feature importance analysis """
        self._debug(f"Feature importance based on {model_name}")
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

    def gradient_boosting(self):
        ''' Create a ExtraTrees model '''
        if self.is_regression():
            m = GradientBoostingRegressor()
        elif self.is_classification():
            m = GradientBoostingClassifier()
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def is_classification(self):
        return self.model_type == 'classification'

    def is_regression(self):
        return self.model_type == 'regression'

    def plot_ic_criterion(self, model, name, color):
        '''
        Plot AIC/BIC criterion
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        '''
        alpha_ = model.alpha_ + EPSILON
        alphas_ = model.alphas_ + EPSILON
        criterion_ = model.criterion_
        plt.plot(-np.log10(alphas_), criterion_, '--', color=color, linewidth=3, label=f"{name} criterion")
        plt.axvline(-np.log10(alpha_), color=color, linewidth=3, label=f"alpha: {name} estimate")
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')

    def plot_lars(self, model_aic, model_bic):
        '''
        Plot LARS AIC/BIC criterion
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        '''
        alpha_bic_ = model_bic.alpha_
        plt.figure()
        self.plot_ic_criterion(model_aic, 'AIC', 'b')
        self.plot_ic_criterion(model_bic, 'BIC', 'r')
        plt.legend()
        plt.title('Information-criterion for model selection')
        plt.show()

    def regularization_models(self):
        ''' Feature importance analysis based on regularization models (Lasso, Ridge, Lars, etc.) '''
        self._debug(f"Feature importance based on regularization")
        # LassoCV
        lassocv = self.regularization_model(LassoCV(cv=self.regularization_model_cv))
        self.plot_lasso_alphas(lassocv)
        # RidgeCV
        ridgecv = self.regularization_model(RidgeCV(cv=self.regularization_model_cv))
        # LARS
        lars_aic = self.regularization_model(LassoLarsIC(criterion='aic'))
        lars_bic = self.regularization_model(LassoLarsIC(criterion='bic'))
        self.plot_lars(lars_aic, lars_bic)

    def regularization_model(self, model):
        ''' Fit a modelularization model and show non-zero coefficients '''
        model.fit(self.x, self.y)
        keep = (model.coef_ != 0.0)
        model_name = model.__class__.__name__
        field_name = f"coeficient_{model_name}"
        coef_df = pd.DataFrame({field_name: model.coef_[keep]}, index=self.x.columns[keep])
        coef_df.sort_values(by=[field_name], ascending=False, inplace=True)
        display(coef_df)
        return model

    def plot_lasso_alphas(self, model):
        '''
        Plot LassoCV model alphas
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        '''
        m_log_alphas = -np.log10(model.alphas_ + EPSILON)
        plt.figure()
        plt.plot(m_log_alphas, model.mse_path_, ':')
        plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model.alpha_ + EPSILON), linestyle='--', color='k', label='alpha: CV estimate')
        plt.legend()
        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: coordinate descent')
        plt.axis('tight')
        plt.show()

    def random_forest(self, n_estimators=100, max_depth=None, bootstrap=True):
        ''' Create a RandomForest model '''
        if self.is_regression():
            m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        elif self.is_classification():
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def select_fdr(self):
        '''
        User SelectFdr to calculate a feature importance rank
        Use different functions, depending on model_type and inputs
        '''
        # Select functions to use
        if self.is_regression():
            funcs = [f_regression, mutual_info_regression]
        elif self.is_classification():
            funcs = [f_classif, mutual_info_classif]
            # Chi^2 only works on non-negative values
            if (X < 0).all(axis=None):
                funcs.append(chi2)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")

        # Apply all functions
        for f in funcs:
            self.select_fdr_f(f)
        return True

    def select_fdr_f(self, score_function):
        ''' Select features using FDR (False Discovery Rate) '''
        # skb = SelectFdr(score_func=score_function, k="all")
        skb = SelectFdr(score_func=score_function)
        fit = skb.fit(self.x, self.y)
        keep = skb.get_support()
        fname = score_function.__name__
        skb_df = pd.DataFrame({f"scores_{fname}": skb.scores_, f"p_values_{fname}": skb.pvalues_, f"keep_{fname}": keep}, index=self.x.columns)
        skb_df.sort_values(by=[f"scores_{fname}", f"p_values_{fname}"], inplace=True, ascending=False)
        display(skb_df)

    def tree_graph(self, max_depth=3, file_dot='tree.dot', file_png='tree.png'):
        """ Simple tree representation """
        self._info(f"Tree graph: Random Forest")
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
