import inspect
import sklearn
import sklearn.dummy
import sklearn.naive_bayes
import traceback

from .model import Model
from ..util.etc import camel_to_snake


class SkLearnModel(Model):
    ''' Create a wrapper for a SkLearn model '''
    def __init__(self, config, datasets, class_name, params, set_config=True):
        super().__init__(config, datasets)
        self.is_save_params = False
        # Set model specific paramters
        self.class_name = class_name
        model_name = camel_to_snake(class_name)
        self._debug(f"Class name: '{class_name}', model name: '{model_name}'")
        if set_config:
            self._set_from_config()
        # Set parameters
        if params:
            for n in params:
                self.__dict__[n] = params[n]
                self._debug(f"Setting {n} = {params[n]}")

    def default_model_create(self, x, y):
        """ Create real model from SciKit learn """
        self._info(f"Creating model based on class '{self.class_name}'")
        class_reference = eval(self.class_name)
        args_spec = inspect.getargspec(class_reference.__init__)
        args_init = args_spec.args
        self._debug(f"Class '{self.class_name}' has constructor arguments: {args_init}")
        kwargs = dict()
        for arg in args_init:
            if arg in self.__dict__:
                val = self.__dict__.get(arg)
                kwargs[arg] = val
        self._debug(f"Invoking constructor '{self.class_name}', with arguments: {kwargs}")
        self.model = eval(f"{self.class_name}(**kwargs)")
        return True

    def default_model_predict(self, x):
        """ Default implementation for '@model_predict' """
        try:
            self._debug(f"Model predict ({self.class_name}): Start")
            y_hat = self.model.predict(x)
            self._debug(f"Model predict ({self.class_name}): End")
            return y_hat
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return None

    def default_model_train(self, x, y):
        """ Fit the model using training data """
        try:
            self._debug(f"Model train ({self.class_name}): Start")
            self.train_results = self.model.fit(x, y)
            self._debug(f"Model train ({self.class_name}): End")
            return True
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return False

    def _loss(self, x, y):
        """ Return the loss """
        ret = super()._loss(x, y)
        if ret is not None:
            return ret
        # Use sklearn model's 'score'
        return 1.0 - self.model.score(x, y)
