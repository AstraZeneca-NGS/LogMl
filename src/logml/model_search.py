
from .config import CONFIG_MODEL, CONFIG_MODEL_SEARCH
from .files import MlFiles
from .models import SkLearnModel


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
        for name in self.models:
            params = self.models[name]
            if 'model' not in params:
                self._debug(f"Model '{name}' does not have a 'model' section, ignoring")
                continue
            model_params = params['model']
            if 'model_class' not in model_params:
                self._debug(f"Model '{name}' does not have a 'model.model_class' value, ignoring")
                continue
            model_class = model_params.get('model_class')
            self._debug(f"Considering model '{name}', model_class={model_class}")
            if self.model_type == model_params['model_type']:
                self.search_model(model_class, params)
        self._info(f"Search models: End")

    def search_model(self, model_class, params):
        ''' Create model and train it '''
        self._debug(f"Searching model: model_class={model_class}\tparameters={params}")
        # Create updated config
        conf = self.config.update_section(None, params)
        self._debug(f"Searching model: New config: {conf}")
        # Create datasets
        !!! TODO: Shallow copy of datasets
        datasets = logml.datasets.copy()
        datasets.enable = False
        # Create LogMl
        logml = LogMl(config=conf, datasets=datasets)
        # Run model
        logml()
