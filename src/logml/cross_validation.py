
import numpy as np
import sklearn.model_selection

from .config import CONFIG_CROSS_VALIDATION, CONFIG_MODEL
from .log import MlLog


CV_METHODS = ['KFold', 'RepeatedKFold', 'LeaveOneOut', 'LeavePOut', 'ShuffleSplit']


class CrossValidation(MlLog):
    ''' Cross validation class '''
    def __init__(self, logml):
        super().__init__(logml.config, CONFIG_CROSS_VALIDATION)
        self.logml = logml
        self.scores = list()
        self.mean = None
        self.std = None
        self._set_from_config()

    def __call__(self):
        return self.cross_validation()

    def _config_sanity_check(self):
        '''
        Check parameters from config.
        Return True on success, False if there are errors
        '''
        model_enable = self.config.get_parameters(CONFIG_MODEL).get('enable')
        if self.enable and not model_enable:
            self._fatal_error(f"Config file '{self.config.config_file}', section {CONFIG_CROSS_VALIDATION} incopnsistency: Cross-validation is enabled, but model is disabled (section {CONFIG_MODEL}, enable:{model_enable})")
        return True

    def get_cv_iterator(self):
        ''' Get cross-validation iterators '''
        cv_type = next((k for k in self.parameters.keys() if k in CV_METHODS), None)
        if not cv_type:
            self._error(f"No supported cross-validation method found. Options {CV_METHODS}")
            return None
        args = self.parameters[cv_type]
        self._debug(f"Found cross-validation method '{cv_type}', with parameters {args}")
        to_eval = f"sklearn.model_selection.{cv_type}(**{args})"
        self._debug(f"to_eval={to_eval}")
        cv = eval(to_eval)
        return cv

    def cross_validation(self):
        ''' Perform cross-validation '''
        self._debug(f"Start")
        # Get cross-valiation iterator
        cv_it = self.get_cv_iterator()
        if not cv_it:
            return False
        # For each split...
        dlen = len(self.logml.datasets)
        self._debug(f"Dataset length: {dlen}")
        x = np.arange(dlen)
        # Get train and validate indeces for each split
        for train, validate in cv_it.split(x):
            # Split train/validate datasets
            self.logml.datasets.split_idx(train, validate)
            # Train
            ret_train = self.logml.model_train()
            self._debug(f"Model train returned: {ret_train}")
            # Test
            ret_validate = self.logml.get_model_validate()
            self._debug(f"Model validate returned: {ret_validate}")
            # Save validate results
            self.scores.append(ret_validate)
        self._info(f"Cross validation: scores={self.scores}")
        self.mean, self.std = self.scores_stats()
        self._info(f"Cross validation: score mean={self.mean}, score std={self.std}")
        self.save_results()
        self._debug(f"End")
        return True

    def save_results(self):
        ''' Save cross-validation results to picle file '''
        mltrain = self.logml.model
        file_name = mltrain.get_file_name('cross_validation')
        self._debug(f"Save cross-validation results: Saving to pickle file '{file_name}'")
        results = {'scores': self.scores, 'parameters': self.parameters}
        self.logml._save_pickle(file_name, 'cross-validation', results)
        return True

    def scores_stats(self):
        ''' Calculate cross validation mean and std '''
        s = np.array(self.scores)
        return s.mean(), s.std()
