import inspect
import sklearn

from .model_base import ModelBase


class SkLearnModel(ModelBase):
    ''' Create a wrapper for a SkLearn model '''
    def __init__(self, config, dataset, class_name, params):
        super().__init__(config, dataset)
        self.class_name = class_name
        self._set_from_config()
        # Set parameters
        for n in params:
            self.__dict__[n] = params[n]

    def model_create(self):
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
