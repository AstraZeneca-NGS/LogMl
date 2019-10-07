
import datetime
import time

from ..core.config import CONFIG_MODEL
from ..core.files import MlFiles
from ..core.registry import MlRegistry, MODEL_CREATE, MODEL_EVALUATE, MODEL_SAVE, MODEL_TRAIN


class Model(MlFiles):
    '''
    Model: A single (training) of a model
    Train a model setting specific parameters
    '''
    _id_counter = 0

    def __init__(self, config, datasets=None, set_config=True):
        '''
        config: An Config object
        datasets: An Datasets object
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
        self.model = None
        self.model_class = self.__class__.__name__
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.train_results = None
        self.test_results = None
        self.validate_results = None
        if set_config:
            self._set_from_config()

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
        # Validate & save results
        if self.model_validate():
            if not self.save_validate_results():
                self._info("Could not save validation results")
        else:
            self._info("Could not run model validation")
        # Test & save results
        if self.model_test():
            if not self.save_test_results():
                self._info("Could not save test results")
        else:
            self._debug("Could not test model")
        time_end = time.process_time()
        self.elapsed_time = time_end - time_start
        return True

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
        " Default implementation for '@model_evaluate' "
        return None

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

    def fit(self, x, y):
        """ model.fit() is an alias to model.model_train() """
        !!!!! HANDLE EXCEPTIONS WHEN FITTING
        return self.invoke_model_train(x, y)

    def get_file_name(self, file_type=None, ext='pkl'):
        ''' Create a file name for training data '''
        self._debug(f"file_type={file_type}, ext='{ext}'")
        return self._get_file_name(self.model_path, self.model_name, file_type, ext, _id=self._id)

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
        !!!!! HANDLE EXCEPTIONS WHEN EVALUATING
        args = [self.model, x, y]
        (invoked, ret) = self.config.invoke(MODEL_EVALUATE, f"Model evaluate {name}", args)
        if invoked:
            self._info(f"Model evaluate {name} returned: '{ret}'")
        return invoked, ret

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

    def model_save(self):
        ''' Save dataset to pickle file '''
        ret = self.invoke_model_save()
        return ret if ret else self.default_model_save()

    def model_test(self):
        if not self.is_test_model:
            self._debug(f"Model testing disabled, skipping (is_test_model={self.is_test_model})")
            return None
        x, y = self.datasets.get_test_xy()
        if x is None:
            self._debug(f"Test dataset not found, skipping")
            return False
        ret = self.model_evaluate(x, y, 'test')
        self.test_results = ret
        return ret is not None

    def model_train(self):
        """ Train (a.k.a. 'fit') the model """
        self._debug(f"Model train: Start")
        x, y = self.datasets.get_train_xy()
        if x is None:
            self._warning("Model train: Cannot get training dataset")
            return False
        ret = self.fit(x, y)
        self._debug(f"Model train: End")
        return ret

    def model_validate(self):
        ''' Validate model: Evaluate on validation dataset '''
        x, y = self.datasets.get_validate_xy()
        if x is None:
            self._debug(f"Validation dataset not found, skipping")
            return False
        ret = self.model_evaluate(x, y, 'validate')
        self.validate_results = ret
        return ret is not None

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
        if self.test_results is None:
            self._debug(f"No test results available, skiping")
            return False
        file_name = self.get_file_name('test_results')
        self._debug(f"Saving to pickle file '{file_name}'")
        self._save_pickle(file_name, 'test_results', self.test_results)
        return True

    def save_validate_results(self):
        ''' Save validation results to picle file '''
        if not self.is_save_validate_pickle:
            self._debug(f"is_save_validate_pickle={self.is_save_validate_pickle}, skiping")
            return True
        if self.validate_results is None:
            self._debug(f"No test results available, skiping")
            return False
        file_name = self.get_file_name('validate_results')
        self._debug(f"Saving to pickle file '{file_name}'")
        self._save_pickle(file_name, 'validate_results', self.validate_results)
        return True
