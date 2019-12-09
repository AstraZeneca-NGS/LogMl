
import copy
import datetime
import math
import numpy as np
import sklearn
import time
import traceback

from ..core.config import CONFIG_MODEL
from ..core.files import MlFiles
from ..core.registry import MlRegistry, MODEL_CREATE, MODEL_EVALUATE, MODEL_PREDICT, MODEL_SAVE, MODEL_TRAIN
from ..util.results_df import ResultsDf


class Model(MlFiles):
    '''
    Model: A single (training) of a model
    Train a model setting specific parameters
    '''
    _id_counter = 0

    def __init__(self, config, datasets=None, set_config=True):
        '''
        Model: A model that can be created, trained and evaluated
        '''
        super().__init__(config, CONFIG_MODEL)
        self._id = self._new_id()
        self.datasets = datasets
        self.elapsed_time = 0
        self.enable = True
        self.is_save_model_pickle = False
        self.is_save_model_method = False
        self.is_save_model_method_ext = 'model'
        self.is_save_params = True
        self.is_save_test_pickle = False
        self.is_save_train_pickle = False
        self.is_save_validate_pickle = False
        self.is_test_model = True
        self.eval_test = None
        self.eval_train = None
        self.eval_validate = None
        self.metric_class = None
        self.metric_class_max = None
        self.metric_class_is_score = False
        self.model = None
        self.model_class = self.__class__.__name__
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.train_results = None
        if set_config:
            self._set_from_config()
        self.model_results = ResultsDf()

    def __call__(self):
        ''' Execute model trainig '''
        if not self.enable:
            self._info(f"Model disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_MODEL}', enable='{self.enable}'")
            return True
        ret = False
        try:
            self.file_stdout = self.get_file_name(ext="stdout")
            self.file_stderr = self.get_file_name(ext="stderr")
            self._debug(f"Redirecting stdout '{self.file_stdout}', stderr: '{self.file_stderr}'")
            self.tee()  # Open tees to log stdout/err to files
            ret = self._call()
        finally:
            self._debug(f"Restoring stdout/stderr")
            self.tee(True)  # Close tees
        return ret

    def _call(self):
        ''' Execute model trainig '''
        if self.datasets is None:
            self._error("Could not find dataset")
            return False
        # Create model & save params
        if not self.model_create():
            self._error("Could not create model")
            return False
        if not self.save_params():
            self._error("Could not save parameters")
            return False
        time_start = time.process_time()
        # Fit model and save it
        ret = self.model_train()
        if not ret:
            self._error("Could not train model")
            return False
        self.model_save()
        if not self.save_train_results():
            self._info("Could not save train results")
        # Evaluate on 'train' dataset
        if not self.model_eval_train():
            self._info("Could not evaluate on 'train' dataset")
        # Validate & save results
        if self.model_eval_validate():
            if not self.save_validate_results():
                self._info("Could not save validation results")
        else:
            self._info("Could not evaluate model on 'validate' dataset")
        # Test & save results
        if self.model_eval_test():
            if not self.save_test_results():
                self._info("Could not save test results")
        else:
            self._debug("Could not test model")
        # Update results
        time_end = time.process_time()
        self.elapsed_time = time_end - time_start
        model_results = {'train': self.eval_train, 'validation': self.eval_validate, 'time': self.elapsed_time}
        model_results.update(self.config.get_parameters_functions(MODEL_CREATE))
        self.model_results.add_row(f"{self.model_class}.{self._id}", model_results)
        return True

    def clone(self, clone_datasets=True):
        """ Clone the model """
        model_clone = copy.copy(self)
        if clone_datasets and self.datasets is not None:
            model_clone.datasets = copy.copy(self.datasets)
        return model_clone

    def _config_sanity_check(self):
        '''
        Check parameters from config.
        Return True on success, False if there are errors
        '''
        assert self.model_path is not None
        assert self.model_name is not None

    def default_model_create(x, y):
        " Default implementation for '@model_create' "
        return False

    def default_model_evaluate(self, x, y, name):
        """ Default implementation for '@model_evaluate' """
        try:
            ret = self.loss_(x, y)
            if ret is None:
                self._warning("No default loss function found ('metric_class' parameter is not configured), returning np.inf")
                ret = np.inf
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            # traceback.print_stack()
            ret = math.inf
        self._debug(f"Model evaluate {name} (default): Loss = {ret}")
        return ret

    def default_model_predict(self, x):
        """ Default implementation for '@model_predict' """
        raise Exception("No (default) method for model predict available")

    def default_model_save(self):
        " Default implementation for '@model_save' "
        if self.is_save_model_pickle:
            # Try using a pickle file to save the model
            file_model = self.get_file_name('model')
            self._debug(f"Save model: Saving to pickle file '{file_model}'")
            self._save_pickle(file_model, 'model', self.model)
            return True
        # Does the model have a 'save' function?
        if self.is_save_model_method and 'save' in dir(self.model):
            file_model = self.get_file_name('model', ext=self.is_save_model_method_ext)
            self._debug(f"Invoking model.save('{file_model}')")
            self.model.save(file_model)
        return False

    def default_model_train(self, x, y):
        """ Fit the model using training data """
        self._error("No (default) method for model train available")
        return False

    def get_file_name(self, file_type=None, ext='pkl'):
        ''' Create a file name for training data '''
        self._debug(f"file_type={file_type}, ext='{ext}'")
        return self._get_file_name(self.model_path, self.model_name, file_type, ext, _id=self._id)

    def fit(self, x, y):
        """ Fit a model (a.k.a model_train) """
        ret = self.invoke_model_train(x, y)
        if not ret:
            ret = self.default_model_train(x, y)
        return ret

    def invoke_model_create(self, x, y):
        " Invoke user defined function '@model_create' "
        if self.model is not None:
            self._debug("Model already exists, skipping")
            return True
        args = [x, y]
        (invoked, ret) = self.config.invoke(MODEL_CREATE, 'Create model', args)
        if invoked:
            self.model = ret
        return invoked

    def invoke_model_evaluate(self, x, y, name):
        ''' Invoke model evaluate '''
        try:
            args = [self.model, x, y]
            (invoked, ret) = self.config.invoke(MODEL_EVALUATE, f"Model evaluate {name}", args)
            if invoked:
                self._info(f"Model evaluate {name} returned: '{ret}'")
            return invoked, ret
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            # traceback.print_stack()
            return True, math.inf

    def invoke_model_predict(self, x):
        ''' Invoke model predict, return all predictions for input/s in x '''
        try:
            args = [self.model, x]
            (invoked, ret) = self.config.invoke(MODEL_PREDICT, f"Model predict", args)
            if invoked:
                self._info(f"Model predict returned: '{ret}'")
            return invoked, ret
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            # traceback.print_stack()
            return True, None

    def invoke_model_save(self):
        " Invoke user defined function '@model_save' "
        args = [self.model]
        (invoked, ret) = self.config.invoke(MODEL_SAVE, 'Save model', args)
        return invoked

    def invoke_model_train(self, x, y):
        " Invoke user defined function '@model_train' "
        args = [self.model, x, y]
        (invoked, ret) = self.config.invoke(MODEL_TRAIN, 'Train model', args)
        if invoked:
            self.train_results = ret
        return invoked

    def load_test_results(self):
        ''' Load test results from pickle file '''
        file_name = self.get_file_name('test_results')
        self._debug(f"Load test results: Loading pickle file '{file_name}'")
        res = self._load_pickle(file_name, 'test_results')
        return res

    def loss_(self, x, y):
        """ Return a metric loss based on a custom metric """
        if self.metric_class is None or self.metric_class == '':
            return None
        # Use the metric class (e.g. from sklearn)
        self._debug(f"Predicting")
        y_hat = self.model_predict(x)
        self._debug(f"Evaluating loss function using {self.metric_class}")
        ret = eval(f"{self.metric_class}(y, y_hat)")
        # Do we need to convert a 'score' into a 'loss' (i.e. to minimze)
        if self.metric_class_max is not None and self.metric_class_max != '':
            self._debug(f"Converting score to loss: maximum value={self.metric_class_max}")
            ret = self.metric_class_max - ret
        elif self.metric_class_is_score:
            self._debug(f"Converting score to loss: negate (metric_class_is_score={self.metric_class_is_score})")
            ret = -ret
        elif self.metric_class.endswith('_score'):
            self._debug(f"Converting score to loss: negate (name ensd with '_score')")
            ret = -ret
        return ret

    def model_create(self):
        ''' Create a model '''
        x, y = self.datasets.get_train_xy()
        if x is None:
            self._warning("Model create: Cannot get training dataset")
            return False
        ret = self.invoke_model_create(x, y)
        if ret:
            return ret
        return self.default_model_create(x, y)

    def model_evaluate(self, x, y, name):
        (invoked, ret) = self.invoke_model_evaluate(x, y, name)
        if invoked:
            return ret
        return self.default_model_evaluate(x, y, name)

    def model_eval_test(self):
        if not self.is_test_model:
            self._debug(f"Model testing disabled, skipping (is_test_model={self.is_test_model})")
            return None
        x, y = self.datasets.get_test_xy()
        if x is None:
            self._debug(f"Test dataset not found, skipping")
            return False
        ret = self.model_evaluate(x, y, 'test')
        self.eval_test = ret
        return ret is not None

    def model_eval_train(self):
        """ Evaluate on train dataset """
        self._debug(f"Model eval train: Start")
        x, y = self.datasets.get_train_xy()
        if x is None:
            self._warning("Model train: Cannot get training dataset")
            return False
        ret = self.model_evaluate(x, y, 'train')
        self.eval_train = ret
        return ret is not None

    def model_eval_validate(self):
        ''' Validate model: Evaluate on validation dataset '''
        x, y = self.datasets.get_validate_xy()
        if x is None:
            self._debug(f"Validation dataset not found, skipping")
            return False
        ret = self.model_evaluate(x, y, 'validate')
        self.eval_validate = ret
        return ret is not None

    def model_predict(self, x):
        (invoked, ret) = self.invoke_model_predict(x)
        if invoked:
            return ret
        return self.default_model_predict(x)

    def model_save(self):
        ''' Save dataset to pickle file '''
        ret = self.invoke_model_save()
        return ret if ret else self.default_model_save()

    def model_train(self):
        """ Train (a.k.a. 'fit') the model """
        self._debug(f"Model train: Start")
        x, y = self.datasets.get_train_xy()
        if x is None:
            self._warning("Model train: Cannot get training dataset")
            return False
        try:
            ret = self.fit(x, y)
            self._debug(f"Model train: End")
            return ret
        except Exception as e:
            self._error(f"Model train exception: {e}\n{traceback.format_exc()}")
            # traceback.print_stack()
            return None

    def _new_id(self):
        ''' Create a new Model._id '''
        Model._id_counter += 1
        dt = datetime.datetime.utcnow().isoformat(sep='.').replace(':', '').replace('-', '')
        return f"{dt}.{Model._id_counter}"

    def save_params(self):
        ''' Save parameters to YAML file '''
        if not self.is_save_params:
            self._debug(f"Saving parameters disabled, skipping (is_save_params='{self.is_save_params})'")
            return True
        cname = type(self).__name__.lower()
        _id = self._id if self._id else ''
        file_yaml = self.get_file_name('parameters', ext=f"yaml")
        if not file_yaml:
            return False
        self._debug(f"file_yaml='{file_yaml}'")
        params = dict(self.config.__dict__)
        self._save_yaml(file_yaml, params)
        return True

    def save_train_results(self):
        ''' Save training results to picle file '''
        if not self.is_save_train_pickle:
            return True
        if self.train_results is None:
            return False
        file_name = self.get_file_name('train_results')
        self._debug(f"Saving to pickle file '{file_name}'")
        self._save_pickle(file_name, 'train_results', self.train_results)
        return True

    def save_test_results(self):
        ''' Save test results to picle file '''
        if not self.is_save_test_pickle:
            self._debug(f"is_save_test_pickle={self.is_save_test_pickle}, skiping")
            return True
        if self.eval_test is None:
            self._debug(f"No test results available, skiping")
            return False
        file_name = self.get_file_name('test_results')
        self._debug(f"Saving to pickle file '{file_name}'")
        self._save_pickle(file_name, 'test_results', self.eval_test)
        return True

    def save_validate_results(self):
        ''' Save validation results to picle file '''
        if not self.is_save_validate_pickle:
            self._debug(f"is_save_validate_pickle={self.is_save_validate_pickle}, skiping")
            return True
        if self.eval_validate is None:
            self._debug(f"No test results available, skiping")
            return False
        file_name = self.get_file_name('validate_results')
        self._debug(f"Saving to pickle file '{file_name}'")
        self._save_pickle(file_name, 'validate_results', self.eval_validate)
        return True
