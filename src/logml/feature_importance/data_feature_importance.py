#!/usr/bin/env python

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import subprocess
import warnings

from boruta import BorutaPy
from IPython.core.display import Image, display
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFdr, SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE, RFECV
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsIC
from sklearn.tree import export_graphviz

from ..core.config import CONFIG_DATASET_FEATURE_IMPORTANCE
from ..core.files import MlFiles
from .feature_importance_from_model import FeatureImportanceFromModel
from ..util.results_df import ResultsDf

EPSILON = 1.0e-4


class DataFeatureImportance(MlFiles):
    '''
    Perform feature importance / feature selection analysis
    '''

    def __init__(self, datasets, config, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_FEATURE_IMPORTANCE)
        self.datasets = datasets
        self.is_fim_random_forest = True
        self.is_fim_extra_trees = True
        self.is_fim_gradient_boosting = True
        self.is_regularization_lasso = True
        self.is_regularization_ridge = True
        self.is_regularization_lars = True
        self.is_rfe_model = True
        self.is_rfe_model_lasso = True
        self.is_rfe_model_ridge = True
        self.is_rfe_model_lars_aic = True
        self.is_rfe_model_lars_bic = True
        self.is_rfe_model_random_forest = True
        self.is_rfe_model_extra_trees = True
        self.is_rfe_model_gradient_boosting = True
        self.is_select = True
        self.is_tree_graph = True
        self.model_type = model_type
        self.regularization_model_cv = 10
        self.rfe_model_cv = 0
        self.tree_graph_max_depth = 4
        if set_config:
            self._set_from_config()
        self.results = None
        self.x, self.y = None, None

    def boruta(self):
        ''' Calculate feature improtance using Boruta algorithm '''
        if not self.is_classification():
            self._debug("Boruta algorithm only for classification")
            return
        self._info(f"Feature importance: Boruta algorithm")
        model = self.fit_random_forest()
        boruta = BorutaPy(model, n_estimators='auto', verbose=2)
        boruta.fit(self.x, self.y)
        self.results.add_col('boruta_support', boruta.support_)
        self.results.add_col('boruta_rank', boruta.ranking_)

    def __call__(self):
        ''' Feature importance '''
        if not self.enable:
            self._debug(f"Dataset feature importance / feature selection disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        self._info(f"Feature importance / feature selection (model_type={self.model_type}): Start")
        self.x, self.y = self.datasets.get_train_xy()
        self.results = ResultsDf(self.x.columns)
        self.feature_importance()
        print(f"BORUTA: NOT WORKING")
        # FIXME:  self.boruta()
        self.regularization_models()
        self.select()
        self.recursive_feature_elimination()
        # Use ranks from all previously calculated models
        self._info(f"Feature importance: Adding 'rank of rank_sum' column")
        self.results.add_rank_of_ranksum()
        # Show a decition tree of the most important variables (first levels)
        self.tree_graph()
        # Display results
        self.results.sort('rank_of_ranksum')
        self._display(self.results.df)
        # Save results
        file_csv = self.datasets.get_file_name('feature_importance', ext=f"csv")
        self._save_csv(file_csv, "Dataset feature importance", self.results.df, save_index=True)
        self._info("Feature importance / feature selection: End")
        return True

    def feature_importance(self):
        ''' Feature importance using several models '''
        if self.is_fim_random_forest:
            self.feature_importance_model(self.fit_random_forest(), 'RandomForest')
        if self.is_fim_extra_trees:
            self.feature_importance_model(self.fit_extra_trees(), 'ExtraTrees')
        if self.is_fim_gradient_boosting:
            self.feature_importance_model(self.fit_gradient_boosting(), 'GradientBoosting')

    def feature_importance_model(self, model, model_name):
        """
        Two feature importance analysis:
            1) Using sklean 'model.feature_importances_'
            2) Using FeatureImportanceFromModel class
        """
        self._debug(f"Feature importance: Based on '{model_name}'")
        fi = FeatureImportanceFromModel(model, model_name, self.x, self.y)
        if not fi():
            self._info("Could not analyze feature importance using RandomForest")
        self.results.add_col(f"importance_model_{model_name}", fi.performance_norm)
        self.results.add_col_rank(f"importance_model_rank_{model_name}", fi.performance_norm, reversed=True)
        fi.plot()
        self.feature_importance_skmodel(model, model_name)
        return True

    def feature_importance_skmodel(self, model, model_name):
        ''' Show model built-in feature importance '''
        self._info(f"Feature importance: Based on SkLean '{model_name}'")
        self.results.add_col(f"feature_importances_sk_{model_name}", model.feature_importances_)
        self.results.add_col_rank(f"feature_importances_sk_rank_{model_name}", model.feature_importances_, reversed=True)

    def fit_lars_aic(self):
        model = LassoLarsIC(criterion='aic')
        model.fit(self.x, self.y)
        return model

    def fit_lars_bic(self):
        model = LassoLarsIC(criterion='bic')
        model.fit(self.x, self.y)
        return model

    def fit_lasso(self):
        model = LassoCV(cv=self.regularization_model_cv)
        model.fit(self.x, self.y)
        return model

    def fit_extra_trees(self, n_estimators=100):
        ''' Create a ExtraTrees model '''
        if self.is_regression():
            m = ExtraTreesRegressor(n_jobs=-1, n_estimators=n_estimators)
        elif self.is_classification():
            m = ExtraTreesClassifier(n_jobs=-1, n_estimators=n_estimators)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def fit_gradient_boosting(self):
        ''' Create a ExtraTrees model '''
        if self.is_regression():
            m = GradientBoostingRegressor()
        elif self.is_classification():
            m = GradientBoostingClassifier()
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def fit_random_forest(self, n_estimators=100, max_depth=None, bootstrap=True):
        ''' Create a RandomForest model '''
        if self.is_regression():
            m = RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        elif self.is_classification():
            m = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', bootstrap=bootstrap)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        m.fit(self.x, self.y)
        return m

    def fit_ridge(self):
        model = RidgeCV(cv=self.regularization_model_cv)
        model.fit(self.x, self.y)
        return model

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
        self._plot_show('Information-criterion for model selection', 'dataset_feature_importance')

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
        plt.axis('tight')
        self._plot_show('Mean square error per fold: coordinate descent', 'dataset_feature_importance')

    def recursive_feature_elimination(self):
        ''' Use RFE to estimate parameter importance based on model '''
        self._debug(f"Feature importance: Recursive feature elimination")
        if not self.is_rfe_model:
            return
        if self.is_regression():
            if self.is_rfe_model_lasso:
                self.recursive_feature_elimination_model(self.fit_lasso(), 'Lasso')
            if self.is_rfe_model_ridge:
                self.recursive_feature_elimination_model(self.fit_ridge(), 'Ridge')
            if self.is_rfe_model_lars_aic:
                self.recursive_feature_elimination_model(self.fit_lars_aic(), 'Lars_AIC')
            if self.is_rfe_model_lars_bic:
                self.recursive_feature_elimination_model(self.fit_lars_bic(), 'Lars_BIC')
        if self.is_rfe_model_random_forest:
            self.recursive_feature_elimination_model(self.fit_random_forest(), 'RandomForest')
        if self.is_rfe_model_extra_trees:
            self.recursive_feature_elimination_model(self.fit_extra_trees(), 'ExtraTrees')
        if self.is_rfe_model_gradient_boosting:
            self.recursive_feature_elimination_model(self.fit_gradient_boosting(), 'GradientBoosting')

    def recursive_feature_elimination_model(self, model, model_name):
        ''' Use RFE to estimate parameter importance based on model '''
        self._info(f"Feature importance: Recursive Feature Elimination, model '{model_name}'")
        if self.rfe_model_cv > 1:
            rfe = RFECV(model, min_features_to_select=1, cv=self.rfe_model_cv)
        else:
            rfe = RFE(model, n_features_to_select=1)
        fit = rfe.fit(self.x, self.y)
        self.results.add_col(f"rfe_{model_name}", fit.ranking_)

    def regularization_models(self):
        ''' Feature importance analysis based on regularization models (Lasso, Ridge, Lars, etc.) '''
        if not self.is_regression():
            return
        self._info(f"Feature importance: Regularization")
        # LassoCV
        if self.is_regularization_lasso:
            lassocv = self.regularization_model(self.fit_lasso())
            self.plot_lasso_alphas(lassocv)
        # RidgeCV
        if self.is_regularization_ridge:
            ridgecv = self.regularization_model(self.fit_ridge())
        # LARS
        if self.is_regularization_lars:
            lars_aic = self.regularization_model(self.fit_lars_aic(), 'Lars_AIC')
            lars_bic = self.regularization_model(self.fit_lars_bic(), 'Lars_BIC')
            self.plot_lars(lars_aic, lars_bic)

    def regularization_model(self, model, model_name=None):
        ''' Fit a modelularization model and show non-zero coefficients '''
        if not model_name:
            model_name = model.__class__.__name__
        self._info(f"Feature importance: Regularization '{model_name}'")
        imp = np.abs(model.coef_)
        self.results.add_col(f"regularization_coef_{model_name}", imp)
        self.results.add_col_rank(f"regularization_rank_{model_name}", imp, reversed=True)
        return model

    def select(self):
        '''
        User Select (Fdr or K-best) to calculate a feature importance rank
        Use different functions, depending on model_type and inputs
        '''
        # Select functions to use, defined as dictionary {function: has_pvalue}
        if not self.is_select:
            return True
        if self.is_regression():
            funcs = {f_regression: True, mutual_info_regression: False}
        elif self.is_classification():
            funcs = {f_classif: True, mutual_info_classif: False}
            # Chi^2 only works on non-negative values
            if (self.x < 0).all(axis=None):
                funcs.append(chi2)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        # Apply all functions
        for f, has_pvalue in funcs.items():
            self.select_f(f, has_pvalue)
        return True

    def select_f(self, score_function, has_pvalue):
        ''' Select features using FDR (False Discovery Rate) or K-best '''
        # skb = SelectFdr(score_func=score_function, k="all")
        fname = score_function.__name__
        self._debug(f"Select FDR: '{fname}'")
        # select = SelectFdr(score_func=score_function)
        if has_pvalue:
            select = SelectFdr(score_func=score_function)
        else:
            select = SelectKBest(score_func=score_function, k='all')
        fit = select.fit(self.x, self.y)
        keep = select.get_support()
        field_name = f"scores_{fname}"
        self.results.add_col(f"selectf_scores_{fname}", select.scores_)
        if has_pvalue:
            self.results.add_col(f"selectf_p_values_{fname}", select.pvalues_)
            self.results.add_col(f"selectf_keep_{fname}", keep)
        self.results.add_col_rank(f"selectf_rank_{fname}", select.scores_, reversed=True)

    def tree_graph(self, file_dot='tree.dot', file_png='tree.png'):
        """ Simple tree representation """
        if not self.is_tree_graph:
            return
        self._info(f"Tree graph: Random Forest")
        # Train a single tree with all the samples
        m = self.fit_random_forest(n_estimators=1, max_depth=self.tree_graph_max_depth, bootstrap=False)
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
        self._display(Image(filename=file_png))
