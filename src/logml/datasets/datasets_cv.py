import copy
import numpy as np
import sklearn.model_selection

from ..core.config import CONFIG_CROSS_VALIDATION
from ..core.files import MlFiles


CV_METHODS = ['KFold', 'RepeatedKFold', 'LeaveOneOut', 'LeavePOut', 'ShuffleSplit']


class DatasetsCv(DatasetsBase):
    """ Datasets used for cross-validation
    Ths object creates a list of datasets which are used for cross-validation.
    For example, if we are using 5-fold cross validation, then we create 5
    datatsets (one for each validation). The list of datasets is stored
    in self.cv_datasets

    Note: Usually cross-validation is used when the dataset is small, so having
    many copies in memory shoudl not be prohibitive. Nevertheless, keep in
    mind that this implementation is not memory efficient.

    Creating this object always assumes that cross-validation is enables.
    Otherwise, unexpected results may ocurr
    """

    def __init__(self, config, dataset, set_config=True):
        """ Creating a DatasetsCv requires the original Datasets object, then
        creates copies and sub-sets of it.
        Usage:
            ```
            ds = Datasets(config)
            dscv = DatasetsCv(config, ds)
            ```
        """
        super().__init__(config, set_config)
        self.cv_datasets = list()   # A list of datasets for each cross-validation
        self.cv_iterator_args = dict()
        self.cv_config = self.config.get_parameters(CONFIG_CROSS_VALIDATION)
        self.cv_enable = self.cv_config.get('enable', False)
        self.cv_count = 0
        if not self.cv_enable:
            self._fatal_error(f"Creating a cross-validation dataset (DatasetCv) when cross-validation is disables, This should never happen!")
        self.cv_type = next((k for k in self.cv_config.keys() if k in CV_METHODS), None)
        if not self.cv_type:
            self._fatal_error(f"Unsupported cross-validation method found. Options {CV_METHODS}")

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

    def in_outs(self, all=True):
        ''' Get inputs & outputs for all datasets '''
        if not self.cv_enable:
            super().in_outs(all)
        else:
            for d in self:
                d.in_outs(all)

    def __iter__(self):
        """ Iterate over all datasets in cross-validation """
        return self.cv_datasets.__iter__()

    def split(self):
        ''' Split dataset for cross-valdation '''
        if not self.cv_enable:
            return super().split(all)
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
            ds = copy.copy(self)
            ds.split_idx(idx_train, idx_validate)
            self.cv_datasets.append(ds)
            self._debug(f"Cross-validation: Created datasets: {len(self.cv_datasets)}")
        self._debug(f"Create cross-validation indexes: End")
        self.cv_count = len(self.cv_datasets)
        return True
