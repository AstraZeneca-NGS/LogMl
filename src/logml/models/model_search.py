
import copy
import logml
import yaml

from ..core.config import CONFIG_MODEL, CONFIG_MODEL_SEARCH, CONFIG_DATASET_EXPLORE, CONFIG_DATASET_FEATURE_IMPORTANCE, CONFIG_HYPER_PARAMETER_OPTMIMIZATION, CONFIG_MODEL_ANALYSIS, CONFIG_MODEL_SEARCH
from ..core.files import MlFiles
from .sklearn_model import SkLearnModel


class ModelSearch(MlFiles):
    '''
    ModelSearch: Build several (base) models and fit the data
    Explore different combinations of models and hyper parameters
    '''
    def __init__(self, logml):
        super().__init__(logml.config, CONFIG_MODEL_SEARCH)
        self.logml = logml
        self.config = logml.config
        self.model_type = logml.model_ori.model_type
        self.models = list()
        self._set_from_config()

    def __call__(self):
        ''' Execute model trainig '''
        self._info(f"Model search: Start")
        if not self.enable:
            self._info(f"Model search disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_MODEL_SEARCH}', enable='{self.enable}'")
            return True
        ret = self.search()
        self._info(f"Model search: End")
        return ret

    def search(self):
        ''' Model search '''
        # Create a list of models
        self._info(f"Search models: Start")
        if self.model_type is None:
            raise ValueError(f"Missing 'model_type' parameter, in config file '{self.config.config_file}', section '{CONFIG_MODEL}'")
        # For each model in 'models' secction: Create a nea LogMl ofbjec with these parameters and run it
        for model_def in self.models:
            name, params = next(iter(model_def.items()))
            if 'enable' in params and not params['enable']:
                self._debug(f"Model '{name}' is disabled (enable='{params['enable']}')")
                continue
            if 'model' not in params:
                self._debug(f"Model '{name}' does not have a 'model' section, ignoring")
                continue
            model_params = params['model']
            if 'model_class' not in model_params:
                self._debug(f"Model '{name}' does not have a 'model.model_class' value, ignoring")
                continue
            model_class = model_params.get('model_class')
            if self.model_type != model_params['model_type']:
                self._debug(f"Model '{name}' does not match model type, skipping ('{self.model_type}' != '{model_params['model_type']}')")
                continue
            self._debug(f"Considering model '{name}', model_class={model_class}")
            self.search_model(model_class, params)
        self._info(f"Search models: End")
        return True

    def search_model(self, model_class, params):
        '''
        Create model and train it.
        Creates a new config, a new LogMl, adds the model to it and then runs
        the new LogMl
        '''
        self._debug(f"Searching model: model_class={model_class}\tparameters={params}")
        enable = params.get('enable', True)
        if not enable:
            self._debug(f"Searching model: Model (model_class={model_class}) disabled (enable={enable}), skipping")
            return
        # Disable some sections to avoid repetition (e.g. 'model_search' to avoid infinite recursion)
        # Create updated config
        conf = self.config.copy(disable_all=True)
        conf = conf.update_section(None, params)
        # Create datasets (shallow copy of datasets)
        self._debug(f"Creating dataset (shallow) copy")
        datasets = copy.copy(self.logml.datasets)
        datasets.enable = False  # We don't want to build dataset again
        # Create LogMl
        self._debug(f"Creating new LogMl")
        lml = logml.LogMl(config=conf, datasets=datasets)
        lml.display_model_results = False  # Don't display results each time
        # Run model
        self._debug(f"Running new LogMl")
        lml()
        self.logml.model_results.add_row_df(lml.model_results.df)  # Collect results for later use
