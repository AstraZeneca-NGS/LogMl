import copy
import numpy as np
import sklearn.model_selection

from ..core import MODEL_TYPE_CLASSIFICATION
from ..core.config import CONFIG_CROSS_VALIDATION
from .datasets_base import DatasetsBase, InOut
from ..util.mem import memory


CV_METHODS = ['KFold', 'RepeatedKFold', 'LeaveOneOut', 'LeavePOut', 'ShuffleSplit', 'StratifiedKFold', 'StratifiedShuffleSplit']


def value_count_percent(y):
    vc = y.value_counts(normalize=True)
    return '{' + ', '.join([f"{i}: {(100 * r):.2f}%" for i, r in vc.iteritems()]) + '}'


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

    The object is iterable, so we can easily iterate over dataset for each
    cross-validation simply doing: `for d in self: ...`

    Most methods in this class, just invoke the original Dataset methods
    on each dataset.

    Creating a DatasetsCv requires the original Datasets object, then it
    creates copies and sub-sets of that dataset.
    Typical usage:
        ```
        ds = Datasets(config)

        dscv = DatasetsCv(config, ds)
        dscv()  # Create cross-validation datasets
        ```
    """

    def __init__(self, config, datasets, model_type, set_config=True):
        super().__init__(config, set_config)
        self.datasets = datasets    # Reference to original dataset
        self.cv_datasets = list()   # A list of datasets for each cross-validation
        self.cv_iterator_args = dict()
        self.cv_config = self.config.get_parameters(CONFIG_CROSS_VALIDATION)
        self.cv_enable = self.cv_config.get('enable', False)
        self.cv_count = 0
        self.model_type = model_type
        if not self.cv_enable:
            self._fatal_error(f"Creating a cross-validation dataset (DatasetCv) when cross-validation is disables, This should never happen!")
        self.cv_type = next((k for k in self.cv_config.keys() if k in CV_METHODS), None)
        if not self.cv_type:
            self._fatal_error(f"Unsupported cross-validation method found. Options {CV_METHODS}")

    def __call__(self):
        ok = self.datasets()
        if ok:
            self._create_cv_datasets()
        return ok

    def all(self):
        " All datasets, includng the original one and cross validations "
        ds_all = [self.datasets]
        ds_all.extend(self.cv_datasets)
        return ds_all

    def clone(self, deep=False):
        ds_clone = copy.copy(self)
        if deep:
            ds_clone.cv_datasets = [d.clone(deep) for d in self]
        return ds_clone

    def _create_cv_datasets(self):
        """ Create cross-validation datasets """
        if len(self.cv_datasets) > 0:
            self._debug(f"DatasetsCv: Creating CV datasets has been done ({len(self.cv_datasets)} datasets found). Skipping")
            return
        cv_it = self._get_cv_iterator()
        if not cv_it:
            self._error(f"Could not get cross-validation iterator for {self.cv_type}")
            return False
        # For each split...
        self._debug(f"DatasetsCv: Create cross-validation indexes: Start. {self.memory()}")
        dlen = len(self.datasets)
        self._debug(f"DatasetsCv: Dataset length={dlen}")
        x = np.arange(dlen)
        _, y = self.datasets.get_xy()  # We may need the outputs on stratified cross-validation
        # Get train and validate indeces for each split
        for idx_train, idx_validate in cv_it.split(x, y):
            self._debug(f"DatasetsCv: Create cross-validation indexes: idx_train length = {len(idx_train)}, idx_validate length = {len(idx_validate)}")
            ds = self.datasets.clone(deep=True)
            ds.split_idx(idx_train, idx_validate)
            if self.model_type == MODEL_TYPE_CLASSIFICATION:
                _, y = ds.get_train_xy()
                yperc = value_count_percent(y)
                self._debug(f"DatasetsCv: Create cross-validation {len(idx_train)}: class distribution: {yperc}")
            self.cv_datasets.append(ds)
            self._debug(f"DatasetsCv: Created datasets: {len(self.cv_datasets)}")
        self._debug(f"DatasetsCv: Create cross-validation indexes: End. {self.memory()}")
        self.cv_count = len(self.cv_datasets)

    def _get_cv_iterator(self):
        """ Get cross-validation iterators """
        if not self.cv_type:
            self._error(f"No supported cross-validation method found. Options {CV_METHODS}")
            return None
        self.cv_iterator_args = self.cv_config[self.cv_type]
        self._debug(f"DatasetsCv: Found cross-validation method '{self.cv_type}', with parameters {self.cv_iterator_args}")
        to_eval = f"sklearn.model_selection.{self.cv_type}(**{self.cv_iterator_args})"
        self._debug(f"DatasetsCv: Method to evaluate={to_eval}")
        cv = eval(to_eval)
        return cv

    def get_datasets_na(self):
        """ Create a dataset of 'missing value indicators' """
        dsna = self.clone()
        dsna.datasets = self.datasets.get_datasets_na()
        dsna.cv_datasets = [d.get_datasets_na() for d in self]
        if any([d is None for d in dsna.cv_datasets]):
            return None
        return dsna

    def get_input_names(self):
        return self.datasets.get_input_names()

    def get_output_names(self):
        return self.datasets.get_output_names()

    def __getitem__(self, key):
        """ Get item/s from the dataset.
        Key could be an int, a list or a slice.
        Returns: Selected elements, as a Numpy array
        """
        raise NotImplementedError("Unimplemented method, this methos should be overiden by a subclass!")

    def get(self):
        return self.datasets.get()

    def get_ori(self):
        return self.datasets.get_ori()

    def get_test(self):
        return self.datasets.get_test()

    def get_test_xy(self):
        return self.datasets.get_test_xy()

    def get_train(self):
        return self.datasets.get_train()

    def get_train_xy(self):
        return self.datasets.get_train_xy()

    def get_validate(self):
        return self.datasets.get_validate()

    def get_validate_xy(self):
        return self.datasets.get_validate_xy()

    def get_xy(self) -> InOut:
        return self.datasets.get_xy()

    def in_outs(self, all=True) -> None:
        return all([d.in_outs(all) for d in self])

    def __iter__(self):
        """ Iterate over all datasets in cross-validation """
        return self.cv_datasets.__iter__()

    def __len__(self):
        return len(self.datasets)

    def memory(self):
        """
        Return memory consumption as a string
        """
        out = f"Dataset {self.dataset_name} memory usage. total: {memory(self)},"
        if self.cv_datasets is not None:
            out += f", number of CVs {len(self.cv_datasets)} "
            out += ', sizes: [' + ','.join([memory(ds) for i, ds in enumerate(self.cv_datasets)]) + ']'
        return out

    def remove_inputs(self, names):
        [d.remove_inputs(names) for d in self.all()]

    def remove_samples_if_missing(self, name):
        [d.remove_samples_if_missing(name) for d in self.all()]

    def reset(self, soft=False):
        return all([d.reset(soft) for d in self.all()])

    def split_idx(self, idx_train, idx_validate, idx_test=None) -> bool:
        [d.split_idx(idx_train, idx_validate, idx_test) for d in self]

    def shuffle_input(self, name, restore=None, new_name=None):
        if restore is None:
            restore = [None] * (self.cv_count + 1)
        return [d.shuffle_input(name, r, new_name=new_name) for d, r in zip(self.all(), restore)]

    def zero_input(self, name, restore=None):
        if restore is None:
            restore = [None] * self.cv_count
        return [d.zero_input(name, r) for d, r in zip(self.all(), restore)]
