#!/usr/bin/env python

import copy
import datetime
import logging
import pandas as pd

from . import Config, CONFIG_DATASET, CONFIG_DATASET_EXPLORE, CONFIG_FUNCTIONS, CONFIG_LOGGER, CONFIG_MODEL
from .files import MlFiles, set_plots
from .registry import MODEL_CREATE
from ..datasets import Datasets, DatasetsDf, DataExplore
from ..feature_importance import DataFeatureImportance
from ..models import CrossValidation, HyperOpt, HYPER_PARAM_TYPES, Model, ModelSearch, SkLearnModel
from ..util.results_df import ResultsDf


class LogMl(MlFiles):
    '''
    ML Logger definition
    Note: This class is used as a singleton
    '''
    def __init__(self, config_file=None, config=None, datasets=None, verbose=False, debug=False):
        if config is None and config_file is not None:
            config = Config(config_file=config_file)
            config()
        if config is not None:
            if verbose:
                config.set_log_level(logging.INFO)
            if debug:
                config.set_log_level(logging.DEBUG)
        super().__init__(config, CONFIG_LOGGER)
        self.datasets = datasets
        self._id_counter = 0
        self.cross_validation = None
        self.dataset_feature_importance = None
        self.disable_plots = False
        self.display_model_results = True
        self.display_max_columns = 1000
        self.display_max_rows = 1000
        self.hyper_parameter_optimization = None
        self.model = None
        self.model_ori = None
        self.model_search = None
        self.model_analysis = None
        self.plots_path = 'logml_plots'
        self.save_model_results = True
        self.save_plots = True
        self.show_plots = True
        self._set_from_config()
        if self.config is not None:
            self.initialize()
        self.model_results = ResultsDf()

    def _config_sanity_check(self):
        '''
        Check parameters from config.
        Return True on success, False if there are errors
        '''
        wf_enabled = list()
        for wf_name in ['cross_validation', 'hyper_parameter_optimization', 'mode_search']:
            wf = self.__dict__.get(wf_name)
            if wf is None:
                continue
            if wf.enable:
                wf_enabled.append(wf_name)
        if len(wf_enabled) > 1:
            self._error(f"More than one workflow enabled (only one can be enabled): {wf_enabled}, config file '{self.config.config_file}'")
            return False
        return True

    def __call__(self):
        ''' Execute model trainig '''
        self._debug(f"Start")
        # Configure
        if self.config is None:
            self.config = Config()
            if not self.config():
                self._error("Could not load config")
                return False
        # Initialize
        self.initialize()
        # Dataset: Load or create dataset, augment, preprocess, split
        if not self.datasets:
            self.datasets = self._new_dataset()
        if not self.datasets():
            self._error("Could not load or create dataset")
            return False
        # Explore dataset
        if not self._dataset_explore():
            self._debug("Could not explore dataset")
        if not self._feature_importance():
            self._debug("Could not perform feature importance")
        if not self._feature_importance_na():
            self._debug("Could not perform feature importance of missing data")
        # Model Train
        if not self.models_train():
            self._error("Could not train model")
            return False
        # Show model/s results
        if self.display_model_results:
            self.model_results.sort(['validation', 'train', 'time'])
            self.model_results.print()
        if self.save_model_results and self.model_results is not None:
            m = self.model_ori if self.model is None else self.model
            file_csv = m.get_file_name('models', ext=f"csv")
            self._save_csv(file_csv, "Model resutls (CSV)", self.model_results.df, save_index=True)
        self._debug("End")
        return True

    def _dataset_explore(self):
        " Explore dataset "
        if not self.is_dataset_df():
            self._debug("Dataset exploration only available for dataset type 'df'")
            return True
        files_base = self.datasets.get_file_name(f"dataset_explore.transformed", ext='')
        self.dataset_explore_transformed = DataExplore(self.datasets.dataset, 'transformed', self.config, files_base)
        ok = self.dataset_explore_transformed()
        if self.config.get_parameters_section(CONFIG_DATASET_EXPLORE, 'is_use_ori'):
            files_base = self.datasets.get_file_name(f"dataset_explore.original", ext='')
            self.dataset_explore_original = DataExplore(self.datasets.dataset_ori, 'original', self.config, files_base)
            ok = self.dataset_explore_original() and ok
        return ok

    def _feature_importance(self):
        " Feature importance / feature selection "
        if not self.is_dataset_df():
            self._debug("Dataset feature importance only available for dataset type 'df'")
            return True
        model_type = self.model_ori.model_type
        self._info("Dataset feature importance: Full dataset")
        self.dataset_feature_importance = DataFeatureImportance(self.config, self.datasets, model_type, 'all')
        return self.dataset_feature_importance()

    def _feature_importance_na(self):
        " Feature importance / feature selection "
        if not self.is_dataset_df():
            self._debug("Dataset feature importance (missing data) is only available for dataset type 'df'")
            return True
        model_type = self.model_ori.model_type
        self._info("Dataset feature importance (missing data)")
        datasets_na = self.datasets.get_datasets_na()
        if datasets_na is None:
            self._debug("Dataset feature importance (missing data): Could not create 'missing' dataset, skipping")
            return False
        self.dataset_feature_importance_na = DataFeatureImportance(self.config, datasets_na, model_type, 'na')
        return self.dataset_feature_importance_na()

    def get_model_eval_test(self):
        ''' Get model test results '''
        return self.model.eval_test

    def get_model_eval_validate(self):
        ''' Get model validate results '''
        return self.model.eval_validate

    def initialize(self):
        ''' Initialize objects after config is setup '''
        if self.config is not None:
            self._set_from_config()
        if self.model_ori is None:
            self.model_ori = Model(self.config)
        if self.hyper_parameter_optimization is None:
            self.hyper_parameter_optimization = HyperOpt(self)
        if self.cross_validation is None:
            self.cross_validation = CrossValidation(self)
        if self.model_search is None:
            self.model_search = ModelSearch(self)
        # Table width
        pd.set_option('display.max_columns', self.display_max_columns)
        pd.set_option('display.max_rows', self.display_max_rows)
        # Set plots options
        set_plots(disable=self.disable_plots, show=self.show_plots, save=self.save_plots, path=self.plots_path)
        return self._config_sanity_check()

    def is_dataset_df(self):
        " Is a 'df' type of dataset? "
        ds_type = self.config.get_parameters(CONFIG_DATASET).get('dataset_type')
        return ds_type == 'df'

    def model_train(self, config=None, dataset=None):
        ''' Train a single model '''
        self._debug(f"Start")
        self.model = self._new_model(config, dataset)
        ret = self.model()
        # Add results
        self.model_results.add_row_df(self.model.model_results.df)
        self._debug(f"End")
        return ret

    def models_train(self):
        ''' Train (several) models '''
        if self.model_search.enable:
            self._debug(f"Model search")
            return self.model_search()
        elif self.hyper_parameter_optimization.enable:
            self._debug(f"Hyper-parameter optimization: single model")
            return self.hyper_parameter_optimization()
        elif self.cross_validation.enable:
            self._debug(f"Cross-validate: single model")
            return self.cross_validation()
        else:
            self._debug(f"Create and train: single model")
            return self.model_train()

    def _new_dataset(self):
        model_type = self.model_ori.model_type
        if self.is_dataset_df():
            self._debug(f"Using dataset class 'DatasetsDf'")
            return DatasetsDf(self.config, model_type)
        else:
            self._debug(f"Using dataset class 'Dataset'")
            return Datasets(self.config)

    def _new_model(self, config=None, datasets=None):
        ''' Create an Model: This is a factory method '''
        if config is None:
            config = self.config
        if datasets is None:
            datasets = self.datasets
        self._debug(f"Parameters: {config.parameters[CONFIG_FUNCTIONS]}")
        # Create models depending on class
        model_class = config.get_parameters_section(CONFIG_MODEL, 'model_class')
        if model_class is not None:
            model_params = config.get_parameters_functions(MODEL_CREATE)
            if model_class.startswith('sklearn'):
                return SkLearnModel(config, datasets, model_class, model_params)
        return Model(config, datasets)
