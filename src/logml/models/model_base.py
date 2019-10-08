
import math
import re
import traceback

from ..core.config import CONFIG_MODEL_SEARCH
from .model import Model


def camel_to_snake(name):
    ''' Convert CamelCase names to snake_case '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class ModelBase(Model):
    '''
    Base class for models used in 'model_search'
    '''

    def __init__(self, config, datasets):
        '''
        config: An Config object
        dataset: An Dataset object
        '''
        super().__init__(config, datasets, set_config=False)
        self.datasets = datasets
        self.is_save_params = False
        self.is_test_model = False  # Do not evaluate 'test' dataset
        # Set model specific paramters
        class_name = type(self).__name__
        model_name = camel_to_snake(class_name)
        self._debug(f"Class name: '{class_name}', model name: '{model_name}'")
        self.model_params = self.config.get_parameters(CONFIG_MODEL_SEARCH).get(model_name)

    def fit(self, x, y):
        ''' Fit model '''
        try:
            self._debug(f"Fitting model {self.class_name}: Start")
            self.train_results = self.model.fit(x, y)
            self._debug(f"Fitting model {self.class_name}: End")
            return True
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return False

    def model_evaluate(self, x, y, name):
        try:
            ret = 1.0 - self.model.score(x, y)
            self._debug(f"Evaluating model {self.class_name}, dataset '{name}': {ret}")
            return ret
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return math.nan
