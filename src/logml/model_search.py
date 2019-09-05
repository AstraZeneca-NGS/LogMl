
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

    def create_model(self, class_name, model_params):
        ''' Create a model or model wrapper '''
        if class_name.startswith('sklearn'):
            return SkLearnModel(self.config, self.logml.dataset, class_name, model_params)
        raise ValueError(f"Unsuported model: class_name: {class_name}")

    def create_models(self):
        """ Create a list of models for the type of problem """
        if self.model_type is None:
            raise ValueError(f"Missing 'model_type' parameter, in config file '{self.config.config_file}', section '{CONFIG_MODEL}'")
        models = list()
        # Create a list of models and set the parameters
        for class_name in self.models:
            model_params = self.models[class_name]
            models.append(self.create_model(class_name, model_params))
            if self.model_type == model_params['model_type']:
                self._debug(f"Adding model: class_name={class_name}\tmodel_params={model_params}")
        if models:
            return models
        raise ValueError(f"Unknown model type '{self.model_type}', in config file '{self.config.config_file}', section '{CONFIG_MODEL}'")

    def create_models_unsupervised(self):
        """ Create a list of models for unsupervised learning """
        return list()

    def search(self):
        ''' Model search '''
        # Create a list of models
        models = self.create_models()
        # Fit each model
        for m in models:
            # TODO: Hyper parameter tunning for each model?
            m()
            res = m.validate_results
