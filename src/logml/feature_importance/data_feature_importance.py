#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import subprocess

from boruta import BorutaPy
from IPython.core.display import Image
from sklearn.feature_selection import SelectFdr, SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE, RFECV
from sklearn.tree import export_graphviz

from ..core import MODEL_TYPE_CLASSIFICATION, MODEL_TYPE_REGRESSION
from ..core.config import CONFIG_DATASET_FEATURE_IMPORTANCE
from ..core.files import MlFiles
from ..core.scatter_gather import pre, scatter, gather
from .feature_importance_permutation import FeatureImportancePermutation
from .feature_importance_drop_column import FeatureImportanceDropColumn
from .pvalue_fdr import get_categories, LogisticRegressionWilks, MultipleLogisticRegressionWilks, PvalueLinear
from ..models.sklearn_model import ModelFactoryExtraTrees, ModelFactoryGradientBoosting, ModelFactoryRandomForest, \
    ModelFactory
from ..models.sklearn_model import ModelSkLarsCV, ModelSkLassoLarsAIC, ModelSkLassoLarsBIC, ModelSkLassoCV, ModelSkLassoLarsCV, ModelSkRidgeCV
from ..util.results_df import ResultsRankDf

EPSILON = 1.0e-4


class DataFeatureImportance(MlFiles):
    """
    Perform feature importance analysis.

    We perform several different approaches using "classic statistics" as
    well as "model based" approaches.

    Model-based methods, such as "drop column" or "permutation", is weighted
    according to the "loss" of the model (on the validation dataset). This means
    that poorly performing models will contribute less than better performing
    models.
    """

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
        self.is_regularization_lasso_lars = True
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
        self.model_dropcol = dict()
        self.model_permutation = dict()

        if set_config:
            self._set_from_config()
        self.results = None
        self.tag = tag
        self.weights = dict()
        self.x, self.y = None, None
        self.x_train, self.y_train = None, None

    def boruta(self):
        """ Calculate feature improtance using Boruta algorithm """
        return      # FIXME: Boruta algorithm not working properly
        if not self.is_classification():
            self._debug("Boruta algorithm only for classification")
            return
        b = self._boruta()
        if b is not None:
            self.results.add_col('boruta_support', b.support_)
            self.results.add_col_rank('boruta_rank', b.ranking_)

    @scatter
    def _boruta(self):
        self._info(f"Feature importance {self.tag}: Boruta algorithm")
        model_factory = ModelFactoryRandomForest(self.config, self.datasets, self.model_type)
        model = model_factory.get()
        boruta = BorutaPy(model, n_estimators='auto', verbose=2)
        boruta.fit(self.x_train, self.y_train)
        return boruta

    def __call__(self):
        """ Feature importance """
        if not self.enable:
            self._debug(f"Feature importance {self.tag} disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        if self.tag == 'na' and not self.enable_na:
            self._debug(f"Feature importance {self.tag} disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable_na='{self.enable_na}'")
            return True
        # Add random inputs (shuffled columns)
        self.random_inputs_added, self.datasets = self.random_inputs_add()
        # Prepare dataset
        self._info(f"Feature importance {self.tag} (model_type={self.model_type}): Start")
        self.x, self.y = self.datasets.get_xy()
        self.x_train, self.y_train = self.datasets.get_train_xy()
        # Initialize results
        inputs = [c for c in self.datasets.get_input_names() if c not in self.datasets.outputs]
        self.results = ResultsRankDf(inputs)
        # Feature importance analyses
        self.feature_importance_models()
        self.boruta()
        self.regularization_models()
        self.select()
        self.recursive_feature_elimination()
        self.pvalue_linear()
        self.wilks()
        self.tree_graph()
        # Process results from all feature importance methods: Perform re-weighting, then display and save results
        loss_ori = self.reweight_results()
        self.show_and_save_results(loss_ori)
        # Restore dataset (remove added random inputs)
        if self.random_inputs_added:
            self._debug(f"Removing shuffled inputs: {self.random_inputs_added}")
            self.datasets.remove_inputs(self.random_inputs_added)
        self._info(f"Feature importance {self.tag}: End")
        return True

    def _is_model_dropcol_provided(self):
        return True if self.model_dropcol['enable'] else False

    def _is_model_permutation_provided(self):
        return True if self.model_permutation['enable'] else False

    def feature_importance_models(self):
        """ Feature importance using several models """
        # Could set the list of models from the folders
        if self._is_model_dropcol_provided() or self._is_model_permutation_provided():
            models_dropcol = self.model_dropcol['models']
            numb_iterations = self.model_dropcol.get('num_iterations')
            if models_dropcol:
                self.feature_importance_set_models(models_dropcol, numb_iterations, 'drop_column')

            models_permutation = self.model_permutation['models']
            numb_iterations = self.model_dropcol.get('num_iterations')
            if models_permutation:
                self.feature_importance_set_models(models_permutation, numb_iterations, 'permutation')
        # or use the predefined list of models
        else:
            any_model = self.is_model_permutation or self.is_model_dropcol or self.is_model_skmodel
            if not any_model:
                self._info(f"All model based methods disabled, skipping")
                return
            if self.is_fip_random_forest:
                model_factory = ModelFactoryRandomForest(self.config, self.datasets, self.model_type)
                self.feature_importance_model(model_factory, 'random_forest')
            else:
                self._debug(f"Feature importance 'Random forest': is_fip_random_forest={self.is_fip_random_forest}, skipping")
            if self.is_fip_extra_trees:
                model_factory = ModelFactoryExtraTrees(self.config, self.datasets, self.model_type)
                self.feature_importance_model(model_factory, 'extra_trees')
            else:
                self._debug(f"Feature importance 'Extra trees': is_fip_extra_trees={self.is_fip_extra_trees}, skipping")
            if self.is_fip_gradient_boosting:
                model_factory = ModelFactoryGradientBoosting(self.config, self.datasets, self.model_type)
                self.feature_importance_model(model_factory, 'gradient_boosting')
            else:
                self._debug(f"Feature importance 'Gradient boosting': is_fip_gradient_boosting={self.is_fip_gradient_boosting}, skipping")

    def feature_importance_set_models(self, models, numb_iterations, models_type):
        for model_name, model_params in models.items():
            model_class = model_params['model']['model_class']
            model_type = model_params['model']['model_type']
            # TODO: could be something else instead model_create?
            model_functions = model_params['functions']['model_create']
            # override numb_iterations for specific model if exist
            numb_iterations = model_functions.get('numb_iterations', numb_iterations)

            model_factory = ModelFactory(
                config=self.config,
                datasets=self.datasets,
                model_type=model_type,
                model_name=model_name,
                cv_enable=None,
                **model_functions
            )
            method = getattr(self, f"feature_importance_{models_type}")
            method(model_factory, 'random_forest', numb_iterations)

    def feature_importance_model(self, model_factory, config_tag):
        """ Perform feature importance analyses based on a (trained) model """
        conf = f"is_fip_{config_tag}"
        model_name = model_factory.model_name
        if not self.__dict__[conf]:
            self._info(f"Feature importance {self.tag} using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        self._info(f"Feature importance {self.tag} using model '{model_name}'")
        if self.is_model_permutation:
            self.feature_importance_permutation(model_factory, config_tag)
        if self.is_model_dropcol:
            self.feature_importance_drop_column(model_factory, config_tag)
        if self.is_model_skmodel:
            self.feature_importance_skmodel(model_factory, config_tag)

    def feature_importance_permutation(self, model_factory, config_tag, num_iterations=None):
        """ Feature importance using 'permutation' analysis """
        conf = f"is_permutation_{config_tag}"
        model_name = model_factory.model_name
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} (permutation) using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        self._debug(f"Feature importance {self.tag} (permutation): Based on '{model_name}'")
        num_iterations = self.__dict__[f"permutation_iterations_{config_tag}"] if num_iterations is None else num_iterations
        fi = FeatureImportancePermutation(model_factory, self.random_inputs_added, num_iterations)
        res = fi()
        if res:
            imp = fi.get_importances()
            self.results.add_col(f"importance_permutation_{model_name}", imp)
            self.results.add_col_rank(f"importance_permutation_rank_{model_name}", imp, weight=fi.get_weight(), reversed=True)
            self.results.add_col(f"importance_permutation_pvalue_{model_name}", fi.get_pvalues())
            fi.plot()
        return True

    def feature_importance_drop_column(self, model_factory, config_tag, num_iterations=None):
        """ Feature importance using 'drop column' analysis """
        conf = f"is_dropcol_{config_tag}"
        model_name = model_factory.model_name
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} (drop column) using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        self._debug(f"Feature importance {self.tag} (drop column): Based on '{model_name}', config '{conf}'")
        num_iterations = self.__dict__[f"permutation_iterations_{config_tag}"] if num_iterations is None else num_iterations
        fi = FeatureImportanceDropColumn(model_factory, self.random_inputs_added, num_iterations)
        res = fi()
        if res:
            imp = fi.get_importances()
            self._info(f"Feature importance (drop column), {model_name}, weight {fi.get_weight()}")
            self.results.add_col(f"importance_dropcol_{model_name}", imp)
            self.results.add_col_rank(f"importance_dropcol_rank_{model_name}", imp, weight=fi.get_weight(), reversed=True)
            self.results.add_col(f"importance_dropcol_pvalue_{model_name}", fi.get_pvalues())
            fi.plot()
        return True

    def feature_importance_skmodel(self, model_factory, config_tag):
        """ Show model built-in feature importance """
        conf = f"is_skmodel_{config_tag}"
        model_name = model_factory.model_name
        if not self.__dict__[conf]:
            self._debug(f"Feature importance {self.tag} (skmodel importance) using model '{model_name}' disabled (config '{conf}' is '{self.__dict__[conf]}'), skipping")
            return
        res = self._feature_importance_skmodel(model_factory)
        if res is not None:
            fi, weight = res
            self.results.add_col(f"importance_skmodel_{model_name}", fi)
            self.results.add_col_rank(f"importance_skmodel_rank_{model_name}", fi, weight=weight, reversed=True)

    @scatter
    def _feature_importance_skmodel(self, model_factory):
        """ Show model built-in feature importance """
        model_name = model_factory.model_name
        model = model_factory.get()
        weight = model.eval_validate if model.model_eval_validate() else None
        self._info(f"Feature importance (sklearn), {model_name}, weight {weight}")
        fi = model.get_feature_importances()
        return fi, weight

    def fit_lars(self, cv_enable=None):
        m = ModelSkLarsCV(self.config, self.datasets, cv=self.regularization_model_cv)
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

    def fit_lasso_lars(self, cv_enable=None):
        m = ModelSkLassoLarsCV(self.config, self.datasets, cv=self.regularization_model_cv)
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_lasso_lars_aic(self, cv_enable=None):
        m = ModelSkLassoLarsAIC(self.config, self.datasets)
        if cv_enable is not None:
            m.cv_enable = cv_enable
        m.model_create()
        m.model_train()
        return m

    def fit_lasso_lars_bic(self, cv_enable=None):
        m = ModelSkLassoLarsBIC(self.config, self.datasets)
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
        return self.model_type == MODEL_TYPE_CLASSIFICATION

    def is_regression(self):
        return self.model_type == MODEL_TYPE_REGRESSION

    def plot_ic_criterion(self, modelsk, name, color):
        """
        Plot AIC/BIC criterion
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        Parameters:
            ic: A tuple having model's parameters (model.alpha_, model.alphas_, model.criterion_)
            name: Model name
            color: Plot color
        """
        alpha = modelsk.alpha_ + EPSILON
        alphas = modelsk.alphas_ + EPSILON
        criterion = modelsk.criterion_
        plt.plot(-np.log10(alphas), criterion, '--', color=color, linewidth=3, label=f"{name} criterion")
        plt.axvline(-np.log10(alpha), color=color, linewidth=3, label=f"alpha: {name} estimate")
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')

    def plot_lars(self, model_aic, model_bic):
        """
        Plot LARS AIC/BIC criterion
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        """
        fig = plt.figure()
        self.plot_ic_criterion(model_aic, 'AIC', 'b')
        self.plot_ic_criterion(model_bic, 'BIC', 'r')
        plt.legend()
        self._plot_show('Information-criterion for model selection', 'dataset_feature_importance', fig)

    def plot_lasso_alphas(self, model):
        """
        Plot LassoCV model alphas
        Ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        """
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
        res = self._pvalue_linear()
        self.pvalue_linear_add_result(res)

    @scatter
    def _pvalue_linear(self):
        """ Calculate p-values using linear regression """
        if not self.is_linear_pvalue:
            return None
        # Only for regression models
        if not self.is_regression():
            self._debug("Linear regression (p-value): Not a regression model, skipping")
            return None
        self._info(f"Linear regression (p-value) {self.tag}: Start")
        plin = PvalueLinear(self.datasets, self.linear_pvalue_null_model_variables, self.tag)
        ok = plin()
        self._info(f"Linear regression (p-value) {self.tag}: End")
        if ok:
            return plin.get_coefficients(), plin.get_pvalues(), plin.p_values_corrected, plin.rejected
        return None

    @gather
    def pvalue_linear_add_result(self, res):
        if res is None:
            return
        coef, pval, pval_corr, rejected = res
        self.results.add_col(f"linear_coefficient", coef)
        self.results.add_col(f"linear_p_values", pval)
        self.results.add_col(f"linear_p_values_fdr", pval_corr)
        self.results.add_col(f"linear_significant", rejected)
        self.results.add_col_rank(f"linear_p_values_rank", pval, reversed=False)

    @pre
    def random_inputs_add(self):
        """
        Add random columns to a dataset
        Return list of names of columns added
        """
        added_columns = list()
        if self.random_inputs_ratio <= 0.0:
            return added_columns, self.datasets
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
        return added_columns, self.datasets

    def recursive_feature_elimination(self):
        """ Use RFE to estimate parameter importance based on model """
        self._debug(f"Feature importance {self.tag}: Recursive feature elimination")
        if not self.is_rfe_model:
            return
        if self.is_regression():
            if self.is_rfe_model_lasso:
                ret = self._rfe_model_lasso()
                self._rfe_add_result(ret)
            if self.is_rfe_model_ridge:
                ret = self._rfe_model_ridge()
                self._rfe_add_result(ret)
            if self.is_rfe_model_lars_aic:
                ret = self._rfe_model_lars_aic()
                self._rfe_add_result(ret)
            if self.is_rfe_model_lars_bic:
                ret = self._rfe_model_lars_bic()
                self._rfe_add_result(ret)
        if self._is_model_dropcol_provided():
            self._rfe_models()

        # if self.is_rfe_model_random_forest:
        #     ret = self._rfe_model_random_forest()
        #     self._rfe_add_result(ret)
        # if self.is_rfe_model_extra_trees:
        #     ret = self._rfe_model_extra_trees()
        #     self._rfe_add_result(ret)
        # if self.is_rfe_model_gradient_boosting:
        #     ret = self._rfe_model_gradient_boosting()
        #     self._rfe_add_result(ret)

    def recursive_feature_elimination_model(self, model, model_name):
        """ Use RFE to estimate parameter importance based on model """
        weight = model.eval_validate if model.model_eval_validate() else None
        self._debug(f"Feature importance {self.tag}: Recursive Feature Elimination, model '{model_name}', x.shape={self.x.shape}, x.shape={self.y.shape}, weight={weight}")
        skmodel = model.model
        if self.rfe_model_cv > 1:
            rfe = RFECV(skmodel, min_features_to_select=1, cv=self.rfe_model_cv)
        else:
            rfe = RFE(skmodel, n_features_to_select=1)
        fit = rfe.fit(self.x, self.y)
        self._info(f"Feature importance {self.tag}: Recursive Feature Elimination '{model_name}', weight {weight}")
        return model_name, fit.ranking_, weight

    @gather
    def _rfe_add_result(self, res):
        model_name, ranking, weight = res
        self.results.add_col_rank(f"rfe_rank_{model_name}", ranking, weight=weight)

    def _rfe_models(self):
        for model_name, model_params in self.model_dropcol['models'].items():
            model_class = model_params['model']['model_class']
            model_type = model_params['model']['model_type']
            # TODO: could be something else instead model_create?
            model_functions = model_params['functions']['model_create']

            model_factory = ModelFactory(
                config=self.config,
                datasets=self.datasets,
                model_type=model_type,
                model_name=model_name,
                cv_enable=None,
                **model_functions
            )
            ret = self._rfe_model(model_factory)
            self._rfe_add_result(ret)

    @scatter
    def _rfe_model(self, model_factory):
        m = model_factory.get()
        return self.recursive_feature_elimination_model(m, model_factory.model_name)

    @scatter
    def _rfe_model_random_forest(self):
        mf = ModelFactoryRandomForest(self.config, self.datasets, self.model_type, cv_enable=False)
        m = mf.get()
        return self.recursive_feature_elimination_model(m, mf.model_name)

    @scatter
    def _rfe_model_extra_trees(self):
        mf = ModelFactoryExtraTrees(self.config, self.datasets, self.model_type, cv_enable=False)
        m = mf.get()
        return self.recursive_feature_elimination_model(m, mf.model_name)

    @scatter
    def _rfe_model_gradient_boosting(self):
        mf = ModelFactoryGradientBoosting(self.config, self.datasets, self.model_type, cv_enable=False)
        m = mf.get()
        return self.recursive_feature_elimination_model(m, mf.model_name)

    @scatter
    def _rfe_model_lars_aic(self):
        return self.recursive_feature_elimination_model(self.fit_lasso_lars_aic(cv_enable=False), 'Lars_AIC')

    @scatter
    def _rfe_model_lars_bic(self):
        return self.recursive_feature_elimination_model(self.fit_lasso_lars_bic(cv_enable=False), 'Lars_BIC')

    @scatter
    def _rfe_model_lasso(self):
        return self.recursive_feature_elimination_model(self.fit_lasso(cv_enable=False), 'Lasso')

    @scatter
    def _rfe_model_ridge(self):
        return self.recursive_feature_elimination_model(self.fit_ridge(cv_enable=False), 'Ridge')

    def regularization_models(self):
        """ Feature importance analysis based on regularization models (Lasso, Ridge, Lars, etc.) """
        if not self.is_regression():
            return
        self._debug(f"Feature importance {self.tag}: Regularization")
        if self.is_regularization_lars:
            res = self._regularization_lars()
            self.regularization_models_add_results(res)
        if self.is_regularization_lasso:
            res = self._regularization_lasso()
            self.regularization_models_add_results(res)
        if self.is_regularization_ridge:
            res = self._regularization_ridge()
            self.regularization_models_add_results(res)
        if self.is_regularization_lasso_lars:
            res = self._regularization_lasso_lars()
            self.regularization_models_add_results(res)
            res_aic = self._regularization_lasso_lars_aic()
            self.regularization_models_add_results(res_aic)
            res_bic = self._regularization_lasso_lars_bic()
            self.regularization_models_add_results(res_bic)
            if res_aic is not None and res_bic is not None:
                self.plot_lars(res_aic[3], res_bic[3])

    @gather
    def regularization_models_add_results(self, reg):
        imp, weight, model_name, model = reg
        self.results.add_col(f"regularization_coef_{model_name}", imp)
        self.results.add_col_rank(f"regularization_rank_{model_name}", imp, weight=weight, reversed=True)

    @scatter
    def _regularization_lars(self):
        return self.regularization_model(self.fit_lars(cv_enable=False), 'Lars')

    @scatter
    def _regularization_lasso(self):
        res = self.regularization_model(self.fit_lasso(cv_enable=False), 'Lasso')
        self.plot_lasso_alphas(res[3])
        return res

    @scatter
    def _regularization_lasso_lars(self):
        return self.regularization_model(self.fit_lasso_lars(cv_enable=False))

    @scatter
    def _regularization_lasso_lars_aic(self):
        return self.regularization_model(self.fit_lasso_lars_aic(cv_enable=False), 'Lars_AIC')

    @scatter
    def _regularization_lasso_lars_bic(self):
        return self.regularization_model(self.fit_lasso_lars_bic(cv_enable=False), 'Lars_BIC')

    @scatter
    def _regularization_ridge(self):
        return self.regularization_model(self.fit_ridge(cv_enable=False), 'Ridge')

    def regularization_model(self, model, model_name=None):
        """ Fit a modularization model and show non-zero coefficients """
        skmodel = model.model
        weight = model.eval_validate if model.model_eval_validate() else None
        if not model_name:
            model_name = skmodel.__class__.__name__
        self._info(f"Feature importance {self.tag}: Regularization '{model_name}, weight {weight}'")
        imp = np.abs(skmodel.coef_)
        return imp, weight, model_name, model.model

    @gather
    def reweight_results(self):
        """
        Rank results according to all methods
            - Re-weight: Current weights are 'loss functions' from each
                methods, i.e. lower is better. We need the opposite, i.e. higher
                weight means the result of a methods is more important.
            - Perform weighted ranking
        """
        self._debug(f"Feature importance {self.tag}: Re-weighting")
        names = self.results.get_weight_names()
        loss_ori = dict(self.results.weights)
        w = self.results.get_weights()
        if len(w) > 0:
            weight_delta = self.weight_max - self.weight_min if self.weight_max > self.weight_min else 1.0
            # Flip weights and scale relative to the best loss (minimum weight)
            self._error(f"REWEIGHT\n\tnames: {names}\n\tloss_ori: {loss_ori}\n\tw.shape = {w.shape}\n\tw: {w}")
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
        """
        User Select (Fdr or K-best) to calculate a feature importance rank
        Use different functions, depending on model_type and inputs
        """
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
            res = self.select_f(f, has_pvalue)
            if res is not None:
                (fname, scores, pvals, keep) = res
                self.results.add_col(f"selectf_scores_{fname}", scores)
                if pvals is not None:
                    self.results.add_col(f"selectf_p_values_{fname}", pvals)
                    self.results.add_col(f"selectf_keep_{fname}", keep)
                    self.results.add_col_rank(f"selectf_pvalue_rank_{fname}", pvals, reversed=False)
                else:
                    self.results.add_col_rank(f"selectf_rank_{fname}", scores, reversed=True)
        return True

    @scatter
    def select_f(self, score_function, has_pvalue):
        """ Select features using FDR (False Discovery Rate) or K-best """
        fname = score_function.__name__
        if has_pvalue:
            self._debug(f"Select FDR: '{fname}'")
            select = SelectFdr(score_func=score_function)
        else:
            self._debug(f"Select K-Best: '{fname}'")
            select = SelectKBest(score_func=score_function, k='all')
        self._debug(f"Select '{fname}': x.shape={self.x.shape}, y.shape={self.y.shape}")
        select.fit(self.x, self.y)
        keep = select.get_support()
        if has_pvalue:
            return (fname, select.scores_, select.pvalues_, keep)
        else:
            return (fname, select.scores_, None, None)

    @gather
    def show_and_save_results(self, loss_ori):
        """ Show and save resutl tables """
        if self.results.is_empty():
            self._debug(f"Feature importance {self.tag}: Enpty resutls, nothing to show or save")
            return
        # Show and save main results table
        self.results.print(f"Feature importance {self.tag}")
        fimp_csv = self.datasets.get_file(f'feature_importance_{self.tag}', ext=f"csv")
        self._info(f"Feature importance {self.tag}: Saving results to '{fimp_csv}'")
        self._save_csv(fimp_csv, f"Feature importance {self.tag}", self.results.df, save_index=True)
        # Show and save weights table
        fimp_weights_csv = self.datasets.get_file(f'feature_importance_{self.tag}_weights', ext=f"csv")
        self._info(f"Feature importance {self.tag}: Saving weights to {fimp_weights_csv}")
        weights = self.results.get_weights_table()
        if loss_ori:
            weights.add_col_dict('loss', loss_ori)
        weights.sort('weights', ascending=False)
        weights.print(f"Feature importance {self.tag} weights")
        self._save_csv(fimp_weights_csv, f"Feature importance {self.tag} weights", weights.df, save_index=True)

    @scatter
    def tree_graph(self, file_dot=None, file_png=None):
        """
        Simple tree representation
        Show a decision tree of the most important variables (first levels)
        """
        if not self.is_tree_graph:
            return
        self._info(f"Tree graph {self.tag}: Random Forest")
        file_dot = self.datasets.get_file(f'tree_graph_{self.tag}', ext=f"dot") if file_dot is None else file_dot
        file_png = self.datasets.get_file(f'tree_graph_{self.tag}', ext=f"png") if file_png is None else file_png
        # Train a single tree with all the samples
        model_factory = ModelFactoryRandomForest(self.config, self.datasets, self.model_type, n_estimators=1, max_depth=self.tree_graph_max_depth, bootstrap=False, cv_enable=False)
        model = model_factory.get()
        skmodel = model.model
        # Export the tree to a graphviz 'dot' format
        export_graphviz(skmodel.estimators_[0], out_file=str(file_dot), feature_names=self.x_train.columns, filled=True, rounded=True)
        self._info(f"Created dot file: '{file_dot}'")
        # Convert 'dot' to 'png', using graphviz command line
        try:
            args = ['dot', '-Tpng', file_dot, '-o', file_png]
            subprocess.run(args)
            self._info(f"Created image: '{file_png}'")
            self._display(Image(filename=str(file_png)))
        except Exception as e:
            self._error(f"Exception '{e}', while trying to run command line {args}. Is graphviz command line package installed?")

    def wilks(self):
        res = self._wilks()
        self.wilks_add_results(res)

    @scatter
    def _wilks(self):
        """ Calculate p-values using logistic regression (Wilks theorem) """
        if not self.is_wilks:
            return None
        # Wilks p-value only for (binary) classification models
        if not self.is_classification():
            self._debug("Logistic Regression (Wilks p-value): Not a classification model, skipping")
            return None
        if not self.wilks_null_model_variables:
            self._info("Logistic Regression (Wilks p-value): Null model variables undefined (config 'wilks_null_model_variables'), skipping")
            return None
        cats = get_categories(self.y)
        self._info(f"Logistic regression, Wilks {self.tag}: Start. Categories: {cats}")
        if len(cats) < 2:
            self._error(f"Logistic regression, Wilks {self.tag}: At least two categories required. Categories: {cats}")
            return None
        is_multiclass = len(cats) > 2
        if is_multiclass:
            self._info(f"Logistic regression, Wilks {self.tag}: Using multiple logistic regression, {len(cats)} categories")
            wilks = MultipleLogisticRegressionWilks(self.datasets, self.wilks_null_model_variables, self.tag)
        else:
            wilks = LogisticRegressionWilks(self.datasets, self.wilks_null_model_variables, self.tag)
        ok = wilks()
        self._info(f"Logistic regression, Wilks {self.tag}: End")
        if ok:
            best_cat = wilks.best_category if is_multiclass else None
            return wilks.get_coefficients(), wilks.get_pvalues(), wilks.p_values_corrected, wilks.rejected, best_cat
        return None

    @gather
    def wilks_add_results(self, res):
        if res is None:
            return
        coef, pvals, pvals_corr, rejected, best_category = res
        self.results.add_col(f"wilks_coefficient", coef)
        self.results.add_col(f"wilks_p_values", pvals)
        self.results.add_col(f"wilks_p_values_fdr", pvals_corr)
        self.results.add_col(f"wilks_significant", rejected)
        self.results.add_col_rank(f"wilks_p_values_rank", pvals, reversed=False)
        if best_category is not None:
            self.results.add_col(f"wilks_best_category", best_category)
