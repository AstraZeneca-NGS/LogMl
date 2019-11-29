
import copy
import numpy as np
import sklearn.model_selection

from ..core.config import CONFIG_CROSS_VALIDATION, CONFIG_MODEL
from ..core.log import MlLog
from . import Model


CV_METHODS = ['KFold', 'RepeatedKFold', 'LeaveOneOut', 'LeavePOut', 'ShuffleSplit']



class ModelCv(Model):
    '''
    A Model with cross-validation capabilities
    '''
    def __init__(self, config, datasets=None, set_config=True):
        super().__init__(config, datasets, set_config)
        # Get cross-validation parameters
        self.cv_config = self.config.get_parameters(CONFIG_CROSS_VALIDATION)
        self.cv_enable = self.cv_config.get('enable', False)
        self.cv_type = next((k for k in self.cv_config.keys() if k in CV_METHODS), None)
        if not self.cv_type:
            self._error(f"No supported cross-validation method found. Options {CV_METHODS}. Cross-validation disables")
            self.cv_enable = False
        self.cv_iterator_args = dict()
        self.cv_datasets = list()   # A list of datasets for each cross-validation
        self.cv_models = list()   # A list of models for each cross-validation
        self.eval_train_std = None
        self.eval_validate_std = None
        self._get_cv_indeces()

    def _get_cv_indeces(self):
        ''' Get dataset indeces for each cross-validation iteration '''
        if not self.cv_enable:
            return False
        cv_it = self._get_cv_iterator()
        if not cv_it:
            self._error(f"Could not get cross-validation iterator for {self.cv_type}")
            return False
        # For each split...
        self._debug(f"Create cross-validation indexes: Start")
        dlen = len(self.datasets)
        self._debug(f"Dataset length: {dlen}")
        x = np.arange(dlen)
        # Get train and validate indeces for each split
        for idx_train, idx_validate in cv_it.split(x):
            self._debug(f"Create cross-validation indexes: idx_train length = {len(idx_train)}, idx_validate length = {len(idx_validate)}")
            ds = copy.copy(self.datasets)
            ds.split_idx(idx_train, idx_validate)
            self.cv_datasets.append(ds)
            self._debug(f"Cross-validation: Created datasets: {len(self.cv_datasets)}")
        self._debug(f"Create cross-validation indexes: End")
        return True

    def _get_cv_iterator(self):
        ''' Get cross-validation iterators '''
        if not self.cv_type:
            self._error(f"No supported cross-validation method found. Options {CV_METHODS}")
            return None
        self.cv_iterator_args = self.cv_config[self.cv_type]
        self._debug(f"Found cross-validation method '{self.cv_type}', with parameters {self.cv_iterator_args}")
        to_eval = f"sklearn.model_selection.{self.cv_type}(**{self.cv_iterator_args})"
        self._debug(f"Method to evaluate: {to_eval}")
        cv = eval(to_eval)
        return cv

    def model_create(self):
        """ Create model for cross-validation """
        if not self.cv_enable:
            return super().model_create()
        self.cv_models = [None] * len(self.cv_datasets)
        rets, self.cv_models = self._cross_validate_f(super().model_create, 'model')
        return all(rets)

    def _cross_validate_f(self, f, collect_name):
        """
        Run cross-validation evaluating function 'f' and collecting field 'collect_name'
        Returns a tuple of two lists: (rets, collects)
            - rest: All return values from each f() invokation
            - collects: All collected values, after each f() invokation
        """
        # Replace datasets for each cross-validation datasets
        datasets_ori = self.datasets
        model_ori = self.model
        # Initialize
        num_cv = len(self.cv_datasets)
        rets = list()
        collect = list()
        for i in range(num_cv):
            # Evaluate model (without cross-validation) on cv_dataset[i]
            self.datasets = self.cv_datasets[i]
            self.model = self.cv_models[i]
            self._debug(f"Cross-validation: Invoking function '{f.__name__}', dataset.id={id(self.datasets)}, model.id={id(self.model)}")
            rets.append(f())
            if collect_name is not None:
                collect.append(self.__dict__[collect_name])
        # Restore original datasets and model
        self.datasets = datasets_ori
        self.model = model_ori
        return rets, collect

    def model_eval_test(self):
        """ Evaluate model on 'test' dataset, using cross-validation """
        if not self.cv_enable:
            return super().model_eval_test()
        rets, losses = self._cross_validate_f(super().model_eval_test, 'eval_test')
        losses = np.array(losses)
        self.eval_test, self.eval_test_std = losses.mean(), losses.std()
        self._debug(f"Model eval test (cross-validation): losses={losses}, eval_test={self.eval_test}, eval_test_std={self.eval_test_std}")
        return all(rets)

    def model_eval_train(self):
        """ Evaluate model on 'train' dataset, using cross-validation """
        if not self.cv_enable:
            return super().model_eval_train()
        rets, losses = self._cross_validate_f(super().model_eval_train, 'eval_train')
        losses = np.array(losses)
        self.eval_train, self.eval_train_std = losses.mean(), losses.std()
        self._debug(f"Model eval train (cross-validation): losses={losses}, eval_train={self.eval_train}, eval_train_std={self.eval_train_std}")
        return all(rets)

    def model_eval_validate(self):
        """ Evaluate model on 'validate' dataset, using cross-validation """
        if not self.cv_enable:
            return super().model_eval_validate()
        rets, losses = self._cross_validate_f(super().model_eval_validate, 'eval_validate')
        losses = np.array(losses)
        self.eval_validate, self.eval_validate_std = losses.mean(), losses.std()
        self._debug(f"Model eval validate (cross-validation): losses={losses}, eval_validate={self.eval_validate}, eval_validate_std={self.eval_validate_std}")
        return all(rets)

    def model_train(self):
        """ Train models for cross-validation """
        if not self.cv_enable:
            return super().model_train()
        rets, _ = self._cross_validate_f(super().model_train, None)
        return all(rets)
