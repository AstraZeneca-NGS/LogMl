#!/usr/bin/env python

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import subprocess
import traceback
import warnings

from boruta import BorutaPy
from IPython.core.display import Image, display
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFdr, SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE, RFECV
from sklearn.tree import export_graphviz

from ..core.config import CONFIG_DATASET_FEATURE_IMPORTANCE
from ..core.files import MlFiles
from .feature_importance_permutation import FeatureImportancePermutation
from .feature_importance_drop_column import FeatureImportanceDropColumn
from .pvalue_fdr import get_categories, LogisticRegressionWilks, MultipleLogisticRegressionWilks, PvalueLinear
from ..models.sklearn_model import ModelSkExtraTreesRegressor, ModelSkExtraTreesClassifier
from ..models.sklearn_model import ModelSkGradientBoostingRegressor, ModelSkGradientBoostingClassifier
from ..models.sklearn_model import ModelSkLassoLarsAIC, ModelSkLassoLarsBIC, ModelSkLassoCV, ModelSkRidgeCV
from ..models.sklearn_model import ModelSkRandomForestRegressor, ModelSkRandomForestClassifier
from ..util.results_df import ResultsRankDf

EPSILON = 1.0e-4


class DataFeatureImportance(MlFiles):
    '''
    Perform feature importance analysis.

    We perform several different approaches using "classic statistics" as
    well as "model based" approaches.

    Model-based methods, such as "drop column" or "permutation", is weighted
    according to the "loss" of the model (on the validation dataset). This means
    that poorly performing models will contribute less than better performing
    models.
    '''

    def __init__(self, config, datasets, model_type, tag, set_config=True):
        super().__init__(config, CONFIG_DATASET_FEATURE_IMPORTANCE)
        self.datasets = datasets
        self.enable_na = True
        self.dropcol_iterations_extra_trees = 1
        self.dropcol_iterations_gradient_boosting = 1
        self.dropcol_iterations_random_forest = 1
        self.is_dropcol_extra_trees = True
        self.is_dropcol_gradient_boosting = True
        self.is_dropcol_random_forest = True
        self.is_fip_extra_trees = True
        self.is_fip_gradient_boosting = True
        self.is_fip_random_forest = True
        self.is_linear_pvalue = True
        self.is_model_dropcol = True
        self.is_model_permutation = True
        self.is_model_skmodel = True
        self.is_permutation_extra_trees = True
        self.is_permutation_gradient_boosting = True
        self.is_permutation_random_forest = True
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
        self.is_skmodel_extra_trees = True
        self.is_skmodel_gradient_boosting = True
        self.is_skmodel_random_forest = True
        self.is_select = True
        self.is_tree_graph = True
        self.is_wilks = True
        self.linear_pvalue_null_model_variables = list()
        self.model_type = model_type
        self.random_inputs_ratio = 1.0  # Add one rand column for each real column
        self.permutation_iterations_extra_trees = 10
        self.permutation_iterations_gradient_boosting = 3
        self.permutation_iterations_random_forest = 10
        self.regularization_model_cv = 10
        self.rfe_model_cv = 0
        self.tree_graph_max_depth = 4
        self.weight_max = 10.0
        self.weight_min = 1.0
        self.wilks_null_model_variables = list()
        if set_config:
            self._set_from_config()
        self.results = None
        self.tag = tag
        self.weights = dict()
        self.x, self.y = None, None
        self.x_train, self.y_train = None, None

    def boruta(self):
        ''' Calculate feature improtance using Boruta algorithm '''
        if not self.is_classification():
            self._debug("Boruta algorithm only for classification")
            return
        self._info(f"Feature importance {self.tag}: Boruta algorithm")
        model = self.fit_random_forest()
        boruta = BorutaPy(model, n_estimators='auto', verbose=2)
        boruta.fit(self.x_train, self.y_train)
        self.results.add_col('boruta_support', boruta.support_)
        self.results.add_col_rank('boruta_rank', boruta.ranking_)

    def __call__(self):
        ''' Feature importance '''
        if not self.enable:
            self._debug(f"Feature importance {self.tag} disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        if self.tag == 'na' and not self.enable_na:
            self._debug(f"Feature importance {self.tag} disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable_na='{self.enable_na}'")
            return True
        # Add random inputs (shuffled columns)
        self.random_inputs_added = self.random_inputs_add()
        self._info(f"Feature importance {self.tag} (model_type={self.model_type}): Start")
        self.x, self.y = self.datasets.get_xy()
        self.x_train, self.y_train = self.datasets.get_train_xy()
        inputs = [c for c in self.datasets.get_input_names() if c not in self.datasets.outputs]
        self.results = ResultsRankDf(inputs)
        self.feature_importance_models()
        # FIXME:  self.boruta()
        self.regularization_models()
        self.select()
        self.recursive_feature_elimination()
        self.pvalue_linear()
        self.wilks()
        # Show a decision tree of the most important variables (first levels)
        self.tree_graph()
        # Perform re-weighting, then display and save results
        loss_ori = self.reweight_results()
        self.show_and_save_results(loss_ori)
        # Restore dataset (remove added random inputs)
        if self.random_inputs_added:
            self._debug(f"Removing shuffled inputs: {self.random_inputs_added}")
            self.datasets.remove_inputs(self.random_inputs_added)
        self._info(f"Feature importance {self.tag}: End")
        return True

    def feature_importance_drop_column(self, model, model_name, config_tag):
        """ Feature importance using 'drop column' analysis """
        conf = f"is_dropcol_{config_tag}"
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} (drop column) using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        self._debug(f"Feature importance {self.tag} (drop column): Based on '{model_name}', config '{conf}'")
        num_iterations = self.__dict__[f"dropcol_iterations_{config_tag}"]
        fi = FeatureImportanceDropColumn(model, f"{self.tag}_{model_name}", self.random_inputs_added, num_iterations)
        if not fi():
            self._info(f"Could not analyze feature importance (drop column) using {model_name}")
            return
        self._info(f"Feature importance (drop column), {model_name} , weight {fi.get_weight()}")
        imp = fi.get_importances()
        self.results.add_col(f"importance_dropcol_{model_name}", imp)
        self.results.add_col_rank(f"importance_dropcol_rank_{model_name}", imp, weight=fi.get_weight(), reversed=True)
        self.results.add_col(f"importance_dropcol_pvalue_{model_name}", fi.get_pvalues())
        fi.plot()
        return True

    def feature_importance_models(self):
        ''' Feature importance using several models '''
        if self.is_fip_random_forest:
            self.feature_importance_model(self.fit_random_forest(), 'RandomForest', 'random_forest')
        else:
            self._debug(f"Feature importance 'Random forest': is_fip_random_forest={self.is_fip_random_forest}, skipping")
        if self.is_fip_extra_trees:
            self.feature_importance_model(self.fit_extra_trees(), 'ExtraTrees', 'extra_trees')
        else:
            self._debug(f"Feature importance 'Extra trees': is_fip_extra_trees={self.is_fip_extra_trees}, skipping")
        if self.is_fip_gradient_boosting:
            self.feature_importance_model(self.fit_gradient_boosting(), 'GradientBoosting', 'gradient_boosting')
        else:
            self._debug(f"Feature importance 'Gradient boosting': is_fip_gradient_boosting={self.is_fip_gradient_boosting}, skipping")

    def feature_importance_model(self, model, model_name, config_tag):
        """ Perform feature importance analyses based on a (trained) model """
        conf = f"is_fip_{config_tag}"
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        if self.is_model_permutation:
            self.feature_importance_permutation(model, model_name, config_tag)
        if self.is_model_dropcol:
            self.feature_importance_drop_column(model, model_name, config_tag)
        if self.is_model_skmodel:
            self.feature_importance_skmodel(model, model_name, config_tag)

    def feature_importance_permutation(self, model, model_name, config_tag):
        """ Feature importance using 'permutation' analysis """
        conf = f"is_permutation_{config_tag}"
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} (permutation) using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        self._debug(f"Feature importance {self.tag} (permutation): Based on '{model_name}'")
        num_iterations = self.__dict__[f"permutation_iterations_{config_tag}"]
        fi = FeatureImportancePermutation(model, f"{self.tag}_{model_name}", self.random_inputs_added, num_iterations)
        if not fi():
            self._info(f"Could not analyze feature importance (permutation) using {model.model_name}")
            return
        self._info(f"Feature importance (permutation), {model_name} , weight {fi.get_weight()}")
        imp = fi.get_importances()
        self.results.add_col(f"importance_permutation_{model_name}", imp)
        self.results.add_col_rank(f"importance_permutation_rank_{model_name}", imp, weight=fi.get_weight(), reversed=True)
        self.results.add_col(f"importance_permutation_pvalue_{model_name}", fi.get_pvalues())
        fi.plot()
        return True

    def feature_importance_skmodel(self, model, model_name, config_tag):
        ''' Show model built-in feature importance '''
        conf = f"is_skmodel_{config_tag}"
        weight = model.eval_validate if model.model_eval_validate() else None
        skmodel = model.model
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} (skmodel importance) using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        self._info(f"Feature importance (sklearn): Based on '{model_name}', weight {weight}")
        fi = model.get_feature_importances()
        self.results.add_col(f"importance_skmodel_{model_name}", fi)
        self.results.add_col_rank(f"importance_skmodel_rank_{model_name}", fi, weight=weight, reversed=True)

    def fit_lars_aic(self, cv_enable=None):
        m = ModelSkLassoLarsAIC(self.config, self.datasets)
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_lars_bic(self, cv_enable=None):
        m = ModelSkLassoLarsBIC(self.config, self.datasets)
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_lasso(self, cv_enable=None):
        m = ModelSkLassoCV(self.config, self.datasets, cv=self.regularization_model_cv)
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_extra_trees(self, n_estimators=100, cv_enable=None):
        ''' Create a ExtraTrees model '''
        if self.is_regression():
            m = ModelSkExtraTreesRegressor(self.config, self.datasets, n_jobs=-1, n_estimators=n_estimators)
        elif self.is_classification():
            m = ModelSkExtraTreesClassifier(self.config, self.datasets, n_jobs=-1, n_estimators=n_estimators)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_gradient_boosting(self, cv_enable=None):
        ''' Create a ExtraTrees model '''
        if self.is_regression():
            m = ModelSkGradientBoostingRegressor(self.config, self.datasets)
        elif self.is_classification():
            m = ModelSkGradientBoostingClassifier(self.config, self.datasets)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_random_forest(self, n_estimators=100, max_depth=None, bootstrap=True, cv_enable=None):
        ''' Create a RandomForest model '''
        if self.is_regression():
            m = ModelSkRandomForestRegressor(self.config, self.datasets, n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        elif self.is_classification():
            m = ModelSkRandomForestClassifier(self.config, self.datasets, n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', bootstrap=bootstrap)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_ridge(self, cv_enable=None):
        m = ModelSkRidgeCV(self.config, self.datasets, cv=self.regularization_model_cv)
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
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
        fig = plt.figure()
        self.plot_ic_criterion(model_aic, 'AIC', 'b')
        self.plot_ic_criterion(model_bic, 'BIC', 'r')
        plt.legend()
        self._plot_show('Information-criterion for model selection', 'dataset_feature_importance', fig)

    def plot_lasso_alphas(self, model):
        '''
        Plot LassoCV model alphas
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        '''
        m_log_alphas = -np.log10(model.alphas_ + EPSILON)
        fig = plt.figure()
        plt.plot(m_log_alphas, model.mse_path_, ':')
        plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model.alpha_ + EPSILON), linestyle='--', color='k', label='alpha: CV estimate')
        plt.legend()
        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.axis('tight')
        self._plot_show('Mean square error per fold: coordinate descent', 'dataset_feature_importance', fig)

    def pvalue_linear(self):
        """ Calculate p-values using linear regression """
        if not self.is_linear_pvalue:
            return True
        # Only for regression models
        if not self.is_regression():
            self._debug("Linear regression (p-value): Not a regression model, skipping")
            return True
        self._info(f"Linear regression (p-value) {self.tag}: Start")
        plin = PvalueLinear(self.datasets, self.linear_pvalue_null_model_variables, self.tag)
        ok = plin()
        if ok:
            self.results.add_col(f"linear_coefficient", plin.get_coefficients())
            self.results.add_col(f"linear_p_values", plin.get_pvalues())
            self.results.add_col(f"linear_p_values_fdr", plin.p_values_corrected)
            self.results.add_col(f"linear_significant", plin.rejected)
            self.results.add_col_rank(f"linear_p_values_rank", plin.get_pvalues(), reversed=False)
        self._info(f"Linear regression (p-value) {self.tag}: End")
        return ok

    def random_inputs_add(self):
        '''
        Add random columns to a dataset
        Return list of names of columns added
        '''
        added_columns = list()
        if self.random_inputs_ratio <= 0.0:
            return added_columns
        for c in self.datasets.get_input_names():
            if self.random_inputs_ratio < 1.0:
                # Add shuffled column with probability 'random_inputs_ratio'
                if np.random.rand() <= self.random_inputs_ratio:
                    c_new = f"__rand_{c}"  # New input name starts with '__rand_'
                    self._debug(f"Adding shuffled input from '{c}': '{c_new}'")
                    self.datasets.shuffle_input(c, new_name=c_new)
                    added_columns.append(c_new)
            if self.random_inputs_ratio >= 1.0:
                for i in range(int(self.random_inputs_ratio)):
                    c_new = f"__rand_{c}_{i+1}"
                    self._debug(f"Adding shuffled input from '{c}': '{c_new}'")
                    self.datasets.shuffle_input(c, new_name=c_new)
                    added_columns.append(c_new)
        return added_columns

    def recursive_feature_elimination(self):
        ''' Use RFE to estimate parameter importance based on model '''
        self._debug(f"Feature importance {self.tag}: Recursive feature elimination")
        if not self.is_rfe_model:
            return
        if self.is_regression():
            if self.is_rfe_model_lasso:
                self.recursive_feature_elimination_model(self.fit_lasso(cv_enable=False), 'Lasso')
            if self.is_rfe_model_ridge:
                self.recursive_feature_elimination_model(self.fit_ridge(cv_enable=False), 'Ridge')
            if self.is_rfe_model_lars_aic:
                self.recursive_feature_elimination_model(self.fit_lars_aic(cv_enable=False), 'Lars_AIC')
            if self.is_rfe_model_lars_bic:
                self.recursive_feature_elimination_model(self.fit_lars_bic(cv_enable=False), 'Lars_BIC')
        if self.is_rfe_model_random_forest:
            self.recursive_feature_elimination_model(self.fit_random_forest(cv_enable=False), 'RandomForest')
        if self.is_rfe_model_extra_trees:
            self.recursive_feature_elimination_model(self.fit_extra_trees(cv_enable=False), 'ExtraTrees')
        if self.is_rfe_model_gradient_boosting:
            self.recursive_feature_elimination_model(self.fit_gradient_boosting(cv_enable=False), 'GradientBoosting')

    def recursive_feature_elimination_model(self, model, model_name):
        ''' Use RFE to estimate parameter importance based on model '''
        weight = model.eval_validate if model.model_eval_validate() else None
        self._debug(f"Feature importance {self.tag}: Recursive Feature Elimination, model '{model_name}', x.shape={self.x.shape}, x.shape={self.y.shape}, weight={weight}")
        skmodel = model.model
        if self.rfe_model_cv > 1:
            rfe = RFECV(skmodel, min_features_to_select=1, cv=self.rfe_model_cv)
        else:
            rfe = RFE(skmodel, n_features_to_select=1)
        fit = rfe.fit(self.x, self.y)
        self._info(f"Feature importance {self.tag}: Recursive Feature Elimination '{model_name}', weight {weight}")
        self.results.add_col_rank(f"rfe_rank_{model_name}", fit.ranking_, weight=weight)

    def regularization_models(self):
        ''' Feature importance analysis based on regularization models (Lasso, Ridge, Lars, etc.) '''
        if not self.is_regression():
            return
        self._debug(f"Feature importance {self.tag}: Regularization")
        # LassoCV
        if self.is_regularization_lasso:
            lassocv = self.regularization_model(self.fit_lasso(cv_enable=False))
            self.plot_lasso_alphas(lassocv)
        # RidgeCV
        if self.is_regularization_ridge:
            ridgecv = self.regularization_model(self.fit_ridge(cv_enable=False))
        # LARS
        if self.is_regularization_lars:
            lars_aic = self.regularization_model(self.fit_lars_aic(cv_enable=False), 'Lars_AIC')
            lars_bic = self.regularization_model(self.fit_lars_bic(cv_enable=False), 'Lars_BIC')
            self.plot_lars(lars_aic, lars_bic)

    def regularization_model(self, model, model_name=None):
        ''' Fit a modularization model and show non-zero coefficients '''
        skmodel = model.model
        weight = model.eval_validate if model.model_eval_validate() else None
        if not model_name:
            model_name = skmodel.__class__.__name__
        self._info(f"Feature importance {self.tag}: Regularization '{model_name}, weight {weight}'")
        imp = np.abs(skmodel.coef_)
        self.results.add_col(f"regularization_coef_{model_name}", imp)
        self.results.add_col_rank(f"regularization_rank_{model_name}", imp, weight=weight, reversed=True)
        return skmodel

    def reweight_results(self):
        '''
        Rank results according to all methods
            - Re-weight: Current weights are 'loss functions' from each
                methods, i.e. lower is better. We need the opposite, i.e. higher
                weight means the result of a methods is more important.
            - Perform weighted ranking
        '''
        self._debug(f"Feature importance {self.tag}: Re-weighting")
        names = self.results.get_weight_names()
        loss_ori = dict(self.results.weights)
        w = self.results.get_weights()
        if len(w) > 0:
            weight_delta = self.weight_max - self.weight_min if self.weight_max > self.weight_min else 1.0
            # Flip weights and scale relative to the best loss (minimum weight)
            w_min = np.abs(w).min()
            if w_min == 0:
                w_min = 1.0
            wp_adj = w / -w_min
            # Correct range using a scaling factor if weights span more than weight_delta
            wp_adj_delta = wp_adj.max() - wp_adj.min()
            corr = weight_delta / wp_adj_delta if wp_adj_delta > weight_delta else 1.0
            # New weights
            wp = corr * (wp_adj - wp_adj.min()) + self.weight_min
            self._debug(f"Feature importance {self.tag}: Re-weighting: weight_delta={weight_delta}, w_min={w_min}, wp_adj={wp_adj}, wp_adj_delta={wp_adj_delta}, corr={corr}, wp={wp}")
            # Set new weights
            for i in range(len(names)):
                self.results.add_weight(names[i], wp[i])
        # Use ranks from all previously calculated models
        self._debug(f"Feature importance {self.tag}: Adding 'rank of rank_sum' column")
        self.results.weight_default = self.weight_min
        self.results.add_rank_of_ranksum()
        # Sort by the resulting column (ranksum)
        self._debug(f"Feature importance {self.tag}: Sorting by 'rank of ranksum'")
        self.results.sort('rank_of_ranksum')
        return loss_ori

    def select(self):
        '''
        User Select (Fdr or K-best) to calculate a feature importance rank
        Use different functions, depending on model_type and inputs
        '''
        # Select functions to use, defined as dictionary {function: has_pvalue}
        # These results have no "weight" set
        if not self.is_select:
            return True
        if self.is_regression():
            funcs = {f_regression: True, mutual_info_regression: False}
        elif self.is_classification():
            funcs = {f_classif: True, mutual_info_classif: False}
            # Chi^2 only works on non-negative values
            if (self.x_train < 0).all(axis=None):
                funcs[chi2] = True
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        # Apply all functions
        for f, has_pvalue in funcs.items():
            self.select_f(f, has_pvalue)
        return True

    def select_f(self, score_function, has_pvalue):
        ''' Select features using FDR (False Discovery Rate) or K-best '''
        fname = score_function.__name__
        if has_pvalue:
            self._debug(f"Select FDR: '{fname}'")
            select = SelectFdr(score_func=score_function)
        else:
            self._debug(f"Select K-Best: '{fname}'")
            select = SelectKBest(score_func=score_function, k='all')
        self._debug(f"Select '{fname}': x.shape={self.x.shape}, y.shape={self.y.shape}")
        fit = select.fit(self.x, self.y)
        keep = select.get_support()
        field_name = f"scores_{fname}"
        self.results.add_col(f"selectf_scores_{fname}", select.scores_)
        if has_pvalue:
            self.results.add_col(f"selectf_p_values_{fname}", select.pvalues_)
            self.results.add_col(f"selectf_keep_{fname}", keep)
            self.results.add_col_rank(f"selectf_pvalue_rank_{fname}", select.pvalues_, reversed=False)
        else:
            self.results.add_col_rank(f"selectf_rank_{fname}", select.scores_, reversed=True)

    def show_and_save_results(self, loss_ori):
        ''' Show and save resutl tables '''
        if self.results.is_empty():
            self._debug(f"Feature importance {self.tag}: Enpty resutls, nothing to show or save")
            return
        # Show and save main results table
        self.results.print(f"Feature importance {self.tag}")
        fimp_csv = self.datasets.get_file_name(f'feature_importance_{self.tag}', ext=f"csv")
        self._info(f"Feature importance {self.tag}: Saving results to '{fimp_csv}'")
        self._save_csv(fimp_csv, f"Feature importance {self.tag}", self.results.df, save_index=True)
        # Show and save weights table
        fimp_weights_csv = self.datasets.get_file_name(f'feature_importance_{self.tag}_weights', ext=f"csv")
        self._info(f"Feature importance {self.tag}: Saving weights to {fimp_weights_csv}")
        weights = self.results.get_weights_table()
        if loss_ori:
            weights.add_col_dict('loss', loss_ori)
        weights.sort('weights', ascending=False)
        weights.print(f"Feature importance {self.tag} weights")
        self._save_csv(fimp_weights_csv, f"Feature importance {self.tag} weights", weights.df, save_index=True)

    def tree_graph(self, file_dot=None, file_png=None):
        """ Simple tree representation """
        if not self.is_tree_graph:
            return
        self._info(f"Tree graph {self.tag}: Random Forest")
        file_dot = self.datasets.get_file_name(f'tree_graph_{self.tag}', ext=f"dot") if file_dot is None else file_dot
        file_png = self.datasets.get_file_name(f'tree_graph_{self.tag}', ext=f"png") if file_png is None else file_png
        # Train a single tree with all the samples
        model = self.fit_random_forest(n_estimators=1, max_depth=self.tree_graph_max_depth, bootstrap=False, cv_enable=False)
        skmodel = model.model
        # Export the tree to a graphviz 'dot' format
        str_tree = export_graphviz(skmodel.estimators_[0],
                                   out_file=file_dot,
                                   feature_names=self.x_train.columns,
                                   filled=True,
                                   rounded=True)
        self._info(f"Created dot file: '{file_dot}'")
        # Convert 'dot' to 'png'
        args = ['dot', '-Tpng', file_dot, '-o', file_png]
        subprocess.run(args)
        self._info(f"Created image: '{file_png}'")
        self._display(Image(filename=file_png))

    def wilks(self):
        """ Calculate p-values using logistic regression (Wilks theorem) """
        if not self.is_wilks:
            return True
        # Wilks p-value only for (binary) classification models
        if not self.is_classification():
            self._debug("Logistic Regression (Wilks p-value): Not a classification model, skipping")
            return True
        if not self.wilks_null_model_variables:
            self._info("Logistic Regression (Wilks p-value): Null model variables undefined (config 'wilks_null_model_variables'), skipping")
            return False
        cats = get_categories(self.y)
        self._info(f"Logistic regression, Wilks {self.tag}: Start. Categories: {cats}")
        if len(cats) < 2:
            self._error(f"Logistic regression, Wilks {self.tag}: At least two categories required. Categories: {cats}")
            return False
        is_multiclass = len(cats) > 2
        if is_multiclass:
            self._info(f"Logistic regression, Wilks {self.tag}: Using multiple logistic regression, {len(cats)} categories")
            wilks = MultipleLogisticRegressionWilks(self.datasets, self.wilks_null_model_variables, self.tag)
        else:
            wilks = LogisticRegressionWilks(self.datasets, self.wilks_null_model_variables, self.tag)
        ok = wilks()
        if ok:
            self.results.add_col(f"wilks_coefficient", wilks.get_coefficients())
            self.results.add_col(f"wilks_p_values", wilks.get_pvalues())
            self.results.add_col(f"wilks_p_values_fdr", wilks.p_values_corrected)
            self.results.add_col(f"wilks_significant", wilks.rejected)
            self.results.add_col_rank(f"wilks_p_values_rank", wilks.get_pvalues(), reversed=False)
            if is_multiclass:
                self.results.add_col(f"wilks_best_category", wilks.best_category)
        self._info(f"Logistic regression, Wilks {self.tag}: End")
        return ok
