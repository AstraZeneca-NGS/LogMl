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

EPSILON = 1.0e-4


class ResultsDf:
    def __init__(self, index):
        self.index = index
        self.df = pd.DataFrame({}, index=index)

    def add_col(self, name, vals):
        ''' Add column 'name:vals' to dataframe '''
        df_new = pd.DataFrame({name: vals}, index=self.index)
        self.df = self.df.join(df_new)

    def add_col_rank(self, name, vals, reversed=False):
        ''' Add a column ranked by value '''
        temp = vals.argsort()
        ranks = np.empty_like(temp)
        if reversed:
            temp = temp[::-1]
        ranks[temp] = np.arange(len(vals))
        self.add_col(name, ranks)

    def add_rank_of_ranksum(self):
        ''' Add a column with the rank of the sum of all columns having 'rank' in the name '''
        len = self.df.shape[0]
        ranks = np.zeros(len)
        for c in self.df.columns:
            if 'rank' in c:
                ranks = ranks + self.df[c]
        self.add_col_rank("rank_of_ranksum", ranks)


# TODO: Add a summary table with results from all feature importance methods
class DataFeatureImportance(MlFiles):
    '''
    Perform feature importance / feature selection analysis
    '''

    def __init__(self, datasets, config, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_FEATURE_IMPORTANCE)
        self.datasets = datasets
        self.model_type = model_type
        self.regularization_model_cv = 10
        self.rfe_model_cv = 5
        self.tree_graph_max_depth = 4
        if set_config:
            self._set_from_config()
        self.x, self.y = self.datasets.get_train_xy()
        self.results = ResultsDf(self.x.columns)

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
            self._info(f"Dataset feature importance / feature selection disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        self._info("Feature importance / feature selection (model_type={self.model_type}): Start")
        self.feature_importance()
        self.boruta()
        self.regularization_models()
        self.select()
        self.recursive_feature_elimination()
        # Use ranks from all previously calculated models
        self.results.add_rank_of_ranksum()
        res = self.results.df
        res.sort_values('rank_of_ranksum', inplace=True)
        display(res)
        # Show a decitin tree of the most important variables (first levels)
        self.tree_graph()
        self._info("Feature importance / feature selection: End")
        return True

    def feature_importance(self):
        ''' Feature importance using several models '''
        self.feature_importance_model(self.fit_random_forest(), 'RandomForest')
        self.feature_importance_model(self.fit_extra_trees(), 'ExtraTrees')
        self.feature_importance_model(self.fit_gradient_boosting(), 'GradientBoosting')

    def feature_importance_model(self, model, model_name):
        """
        Two feature importance analysis:
            1) Using sklean 'model.feature_importances_'
            2) Using FeatureImportanceFromModel class
        """
        self._debug(f"Feature importance based on {model_name}")
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
            m = ExtraTreesRegressor(n_estimators=n_estimators)
        elif self.is_classification():
            m = ExtraTreesClassifier(n_estimators=n_estimators)
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
            m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        elif self.is_classification():
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', bootstrap=bootstrap)
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
        plt.show()

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

    def recursive_feature_elimination(self):
        ''' Use RFE to estimate parameter importance based on model '''
        self._debug(f"Feature importance: Recursive feature elimination")
        if self.is_regression():
            self.recursive_feature_elimination_model(self.fit_lasso(), 'Lasso')
            self.recursive_feature_elimination_model(self.fit_ridge(), 'Ridge')
            self.recursive_feature_elimination_model(self.fit_lars_aic(), 'Lars_AIC')
            self.recursive_feature_elimination_model(self.fit_lars_bic(), 'Lars_BIC')
        self.recursive_feature_elimination_model(self.fit_random_forest(), 'RandomForest')
        self.recursive_feature_elimination_model(self.fit_extra_trees(), 'ExtraTrees')
        self.recursive_feature_elimination_model(self.fit_gradient_boosting(), 'GradientBoosting')

    def recursive_feature_elimination_model(self, model, model_name):
        ''' Use RFE to estimate parameter importance based on model '''
        self._debug(f"Feature importance: Recursive feature elimination, model '{model_name}'")
        if self.rfe_model_cv > 1:
            rfe = RFECV(model, min_features_to_select=1, cv=self.rfe_model_cv)
        else:
            rfe = RFE(model, n_features_to_select=1, cv=self.rfe_model_cv)
        fit = rfe.fit(self.x, self.y)
        self.results.add_col(f"rfe_{model_name}", fit.ranking_)

    def regularization_models(self):
        ''' Feature importance analysis based on regularization models (Lasso, Ridge, Lars, etc.) '''
        self._debug(f"Feature importance based on regularization")
        # LassoCV
        lassocv = self.regularization_model(self.fit_lasso())
        self.plot_lasso_alphas(lassocv)
        # RidgeCV
        ridgecv = self.regularization_model(self.fit_ridge())
        # LARS
        lars_aic = self.regularization_model(self.fit_lars_aic())
        lars_bic = self.regularization_model(self.fit_lars_bic())
        self.plot_lars(lars_aic, lars_bic)

    def regularization_model(self, model):
        ''' Fit a modelularization model and show non-zero coefficients '''
        keep = (model.coef_ != 0.0)
        model_name = model.__class__.__name__
        self.results.add_col(f"regularization_coef_{model_name}", model.coef_)
        self.results.add_col_rank(f"regularization_rank_{model_name}", model.coef_, reversed=True)
        return model

    def select(self):
        '''
        User Select (Fdr or K-best) to calculate a feature importance rank
        Use different functions, depending on model_type and inputs
        '''
        # Select functions to use, defined as dictionary {function: has_pvalue}
        if self.is_regression():
            funcs = {f_regression: True, mutual_info_regression: False}
        elif self.is_classification():
            funcs = {f_classif: True, mutual_info_classif: False}
            # Chi^2 only works on non-negative values
            if (X < 0).all(axis=None):
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

    def tree_graph(self, max_depthfile_dot='tree.dot', file_png='tree.png'):
        """ Simple tree representation """
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
        display(Image(filename=file_png))
