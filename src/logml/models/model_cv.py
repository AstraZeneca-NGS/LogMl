
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
        # Lists of arrays containing the indeces for train/validation datasets. There
        # is one array for each cross validation iteration
        self.idx_train = list()
        self.idx_validate = list()
        # In cross-validation, we have multiple lossses
        self.losses = list()
        self.loss_mean = None
        self.loss_std = None
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
        for train, validate in cv_it.split(x):
            self._debug(f"Create cross-validation indexes: train length = {len(train)}, validate length = {len(validate)}")
            self.idx_train.append(train)
            self.idx_validate.append(validate)
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

    def model_eval_validate(self):
        self._fatal_error("UNIMPLEMENTED!")
        # TODO: Create a copy of datasets?
        # TODO: For each self.idx_train
        # TODO:     dataset_split_idx(idx_train)
        # TODO:     dataset.inouts()?
        # TODO:     super.model_eval_validate()
        # TODO:     self.losses.append(self.eval_validate_loss)
        # TODO: self._scores_stats()
        # TODO: Restore original datasets?

    def model_train(self):
        self._fatal_error("UNIMPLEMENTED!")
        # TODO: Create a copy of datasets?
        # TODO: For each self.idx_train
        # TODO:     dataset_split_idx(idx_train)
        # TODO:     dataset.inouts()?
        # TODO:     super.model_train()
        # TODO: Restore original datasets?

    def _scores_stats(self):
        ''' Calculate cross validation mean and std '''
        s = np.array(self.scores)
        return s.mean(), s.std()
