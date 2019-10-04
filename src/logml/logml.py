import copy
import datetime

from .core import Config, CONFIG_DATASET, CONFIG_FUNCTIONS, CONFIG_LOGGER, CONFIG_MODEL
from .core.files import MlFiles
from .core.registry import MODEL_CREATE
from .datasets import Datasets, DatasetsDf, DataExplore
from .feature_importance import DataFeatureImportance
from .models import CrossValidation, HyperOpt, HYPER_PARAM_TYPES, Model, ModelSearch, SkLearnModel
from .util.results_df import ResultsDf


class LogMl(MlFiles):
    '''
    ML Logger definition
    Note: This class is used as a singleton
    '''
    def __init__(self, config=None, datasets=None):
        super().__init__(config, config_section=CONFIG_LOGGER)
        self.datasets = datasets
        self._id_counter = 0
        self.cross_validation = None
        self.hyper_parameter_optimization = None
        self.model = None
        self.model_ori = None
        self.model_search = None
        self.model_analysis = None
        self._set_from_config()
        if self.config:
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
        if not self._explore():
            self._debug("Could not explore dataset")
        if not self._feature_importance():
            self._debug("Could not perform feature importance / feature selection")
        # Model Train
        if not self.models_train():
            self._error("Could not train model")
            return False
        self._debug("End")
        return True

    def _explore(self):
        " Explore dataset "
        if not self.is_dataset_df():
            self._debug("Dataset exploration only available for dataset type 'df'")
            return True
        de = DataExplore(self.datasets, self.config)
        return de()

    def _feature_importance(self):
        " Feature importance / feature selection "
        if not self.is_dataset_df():
            self._debug("Dataset feature importance only available for dataset type 'df'")
            return True
        model_type = self.model_ori.model_type
        dfi = DataFeatureImportance(self.datasets, self.config, model_type)
        return dfi()

    def get_model_validate(self):
        ''' Get model validate results '''
        return self.model.validate_results

    def get_model_test(self):
        ''' Get model test results '''
        return self.model.test_results

    def initialize(self):
        ''' Initialize objects after config is setup '''
        if self.model_ori is None:
            self.model_ori = Model(self.config)
        if self.hyper_parameter_optimization is None:
            self.hyper_parameter_optimization = HyperOpt(self)
        if self.cross_validation is None:
            self.cross_validation = CrossValidation(self)
        if self.model_search is None:
            self.model_search = ModelSearch(self)
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
        model_results = {'train': self.model.validate_results, 'validate': self.model.validate_results}
        self.model_results.add_row(model_results, f"{self.model.model_class}.{self.model._id}")
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
        if self.is_dataset_df():
            self._debug(f"Using dataset class 'DatasetsDf'")
            return DatasetsDf(self.config)
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
