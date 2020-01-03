import copy
import numpy as np
import random

from collections import namedtuple
from ..core.config import CONFIG_DATASET
from ..core.files import MlFiles
from ..core.registry import MlRegistry, DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT, DATASET_LOAD, DATASET_PREPROCESS, DATASET_SAVE, DATASET_SPLIT, DATASET_TRANSFORM

# InOut is a named tuple containig dataset's inputs (InOut.x) and outputs (InOut.y)
InOut = namedtuple('InOut', ['x', 'y'])


class DatasetsBase(MlFiles):
    """ Define the basic operations that a 'Datasets' object should have
    Datasets object contains data to use for ML algorithms. For instance
    train models, perform data exploration, feature importance, etc.

    The 'raw dataset' is stored in `self.dataset` and there are no
    restrictions, it could be any kind of object. Typicalle, you may want
    to use a Pandas Dataframe, Numpy array, Pytorch tensor, etc.

    The 'raw dataset' is split into train, validate and test datasets. Then
    each of these (train/validate/test) is split into inputs and outputs. For
    this we use an InOut object (a named tuple) having fields 'x' and 'y'

    The Datasets object is callable, and calling the object should run all
    the steps in creating a dataset:
        1) Load or create the 'raw dataset'
        2) Save to a pickle file (for later , faster access)
        3) Transform dataset (e.g. convert input variables to numeric or one-hot)
        4) Augment dataset
        5) Preprocess dataset (e.g. normalize inputs)
        6) Split into train, validation and testing
        7) Split inputs and outputs

    For each of these steps, we first try to call a 'user defined function'. This
    is done in methods `invoke_*()`, for instance `invoke_preprocess()`,
    `invoke_split()`, ``invoke_transform()`, etc. If there is no user defined
    function defeined then a default implementation is used, for example methods
    `default_preprocess()`, `default_split()`, `default_transform()`, etc.

    """
    def __init__(self, config, set_config=True):
        super().__init__(config, CONFIG_DATASET)
        self.dataset_path = None
        self.dataset_name = None
        self.dataset_type = None
        self.dataset = None
        self.dataset_test = None
        self.dataset_train = None
        self.dataset_validate = None
        self.dataset_xy = InOut(None, None)
        self.dataset_test_xy = InOut(None, None)
        self.dataset_train_xy = InOut(None, None)
        self.dataset_validate_xy = InOut(None, None)
        self.do_not_load_pickle = False
        self.do_not_save = False
        self.enable = True
        self.is_use_all_inputs = False
        self.is_use_default_in_out = True
        self.is_use_default_preprocess = True
        self.is_use_default_split = True
        self.is_use_default_transform = True
        self.operations_done = set()
        self.operations = [DATASET_TRANSFORM, DATASET_PREPROCESS, DATASET_AUGMENT, DATASET_SPLIT]
        self.outputs = list()
        self.should_save = False
        if set_config:
            self._set_from_config()

    def augment(self):
        """ Perform augmentation step.
        Invoke a user defined function, if none available, call `default_augment()` method.
        Returns: True if the augmentation was performed, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def __call__(self) -> bool:
        '''
        Load (or create) dataset, then augment, proprocess and split
        Save at each step for faster processing / consistency
        Returns: True on success, False otherwise
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def clone(self, deep=False):
        ''' Create a copy of this Datasets object
        Arguments:
            deep: If true, perform deep copy, otherwise, shallow copy
        Returns: A copy if this Datasets object
        '''
        return copy.deepcopy(self) if deep else copy.copy(self)

    def create(self) -> bool:
        return self.invoke_create()

    def default_in_out(self, ds, name) -> InOut:
        '''
        Default method for getting inputs / outputs
        Returns: An InOut containing input and output data
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def default_load(self) -> bool:
        '''
        Load dataset from pickle file.
        It should set the following dataset values:
            self.dataset
            self.dataset_test
            self.dataset_train
            self.dataset_validate
            self.operations
            self.operations_done
        Returns: True on success, False otherwise
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def default_preprocess(self) -> bool:
        """
        Default implementation for '@dataset_preprocess'
        Returns: True on success, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def default_save(self):
        ''' Default implementation of '@dataset_save' '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def default_split(self):
        '''
        Default implementation for '@dataset_split'
        Assumptions:
            1) self.dataset object is iterable
            2) Parameter 'split_test' and 'split_validate' are defined such that
                2.a) split_test >= 0
                2.b) split_validate >= 0
                2.c) split_test + split_validate < 1
        Returns: A tuple with three lists of 'samples' (train, validate, test)
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def default_transform(self) -> bool:
        """ Default implementation for '@dataset_transform'
        Returns: True on success, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def get_file_name(self, dataset_type=None, ext='pkl'):
        ''' Create a file name for dataset '''
        self._debug(f"dataset_type={dataset_type}, ext='{ext}'")
        return self._get_file_name(self.dataset_path, self.dataset_name, dataset_type, ext)

    def __getitem__(self, key):
        """ Get item/s from the dataset.
        Key could be an int, a list or a slice.
        Returns: Selected elements, as a Numpy array
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def get(self):
        """ Get the 'raw' dataset """
        return self.dataset

    def get_datasets_na(self):
        """
        Create a dataset of missing data indicators.
        The new datasets should have the same samples and inputs, but replacing
        'missing' by 1 and 'not missing' by 0
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def get_input_names(self):
        """ Returns: A list of dataset's input names """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def get_ori(self):
        """ Get the 'raw' dataset (original) """
        return self.dataset_ori

    def get_test(self):
        """ Get the 'raw' test dataset """
        if self.dataset_test is None:
            self._info(f"Test dataset not found")
        return self.dataset_test

    def get_test_xy(self):
        """ Get the 'raw' test dataset (split InOut) """
        if self.dataset_test_xy.x is None:
            self._info(f"Test dataset not found")
        return self.dataset_test_xy

    def get_train(self):
        """ Get the 'raw' train dataset """
        if self.dataset_train is None:
            self._debug(f"Training dataset not found, using whole dataset")
            return self.dataset
        return self.dataset_train

    def get_train_xy(self):
        """ Get the 'raw' train dataset (split InOut) """
        if self.dataset_train_xy.x is None:
            self._debug(f"Training dataset not found, using whole dataset")
            return self.dataset_xy
        return self.dataset_train_xy

    def get_validate(self):
        """ Get the 'raw' validate dataset """
        if self.dataset_validate is None:
            self._debug(f"Validate dataset not found, using whole dataset")
            return self.dataset
        return self.dataset_validate

    def get_validate_xy(self):
        """ Get the 'raw' validate dataset (split InOut) """
        if self.dataset_validate_xy.x is None:
            self._debug(f"Validate dataset not found, using whole dataset")
            return self.dataset_xy
        return self.dataset_validate_xy

    def get_xy(self) -> InOut:
        """ Get the 'raw' dataset split into InOut"""
        return self.dataset_xy

    def _in_out(self, ds, name) -> InOut:
        '''
        Split dataset inputs and outputs from a 'raw dataset'
        Args:
            ds: A 'raw dataset'
            name: The name of the dataset, to be used in log messages
        Returns:
            InOut with the inputs (InOut.x) and outputs (InOut.y)
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def in_outs(self, all=True) -> None:
        '''
        Get inputs & outputs for all datasets (train, validate, test)
        Args:
            all: A boolean indicating whether 'self.datasets' should also be split
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_augment(self):
        """ Invoke user defined function for '@dataset_augment'
        The user defined function takes a 'raw dataset' as an argument and
        returns a (new and augmented) 'raw dataset'. So the user's defined
        function return value should be stored in self.dataset
        Returns:
            True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_create(self):
        """ Invoke user defined function for '@dataset_create' "
        The user defined function takes returns a (newly created) 'raw dataset'.
        So the user's defined function return value should be stored in self.dataset
        Returns:
            True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_in_out(self, ds, name):
        """ Invoke user defined function for '@dataset_inout'
        The user defined function takes a 'raw dataset' as an argument and returns
        two datasets: inputs and outputs (x, y).
        Returns: A tuple of bool, InOut
            bool: True if the user defined funciton was invoked, False otherwise
            InOut: A dataset split into (x, y). The values of x and y are `None`
            if the user defined function was not invoked.
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_load(self):
        """ Invoke user defined function fo '@dataset_load'
        The user defined function takes returns a (newly loaded) 'raw dataset'.
        So the user's defined function return value should be stored in self.dataset
        Returns:
            True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_preprocess(self):
        """ Invoke user defined function for '@dataset_preprocess'
        The user defined function takes a 'raw dataset' as an argument and
        returns a (preprocessed) 'raw dataset'. So the user's defined
        function return value should be stored in self.dataset
        Returns:
            True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_save(self):
        """ Invoke user defined function for '@dataset_save'
        The user defined function takes a 'raw dataset' as an argument.
        Returns:
            True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_split(self):
        """ Invoke user defined function for '@dataset_split'
        The user defined function takes a 'raw dataset' as an argument and returns
        three datasets: (train, validate, test). Test and or validate dataset
        can be None. These values should be stored in
        `self.dataset_train, self.dataset_validate, self.dataset_test`
        Returns:
            bool: True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def invoke_transform(self):
        """ Invoke user defined function for '@dataset_transform'
        The user defined function takes a 'raw dataset' as an argument and
        returns a (transformed) 'raw dataset'. So the user's defined
        function return value should be stored in self.dataset
        Returns:
            True if the user defined funciton was invoked, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def __len__(self):
        """ Length (number of samples) in the 'raw dataset' """
        return 0 if self.dataset is None else len(self.dataset)

    def load(self):
        ''' Try to load dataset.
        Attempts to load a dataset, first from pickle otherwise from user defined
        function.
        It should set `self.should_save` properly, e.g. if the dataset was
        loaded from a pickle file, there is no need to save it again to the
        same pickle file.
        Returns: True if the dataset is loaded, False otherwise
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def preprocess(self):
        """ Perform pre-processing step.
        Invoke a user defined function, if none available, call `default_preprocess()` method.
        Returns: True if the pre-processing was performed, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def reset(self, soft=False):
        ''' Reset fields '''
        self.dataset = None
        self.dataset_test = None
        self.dataset_train = None
        self.dataset_validate = None
        self.operations_done = set()
        if not soft:
            self.dataset_xy = InOut(None, None)
            self.dataset_test_xy = InOut(None, None)
            self.dataset_train_xy = InOut(None, None)
            self.dataset_validate_xy = InOut(None, None)
            self.operations = [DATASET_TRANSFORM, DATASET_AUGMENT, DATASET_PREPROCESS, DATASET_SPLIT]
            self.outputs = list()
            self.should_save = False

    def save(self):
        """ Try to save dataset.
        First use user defined function otherwise save to pickle otherwise
        Returns: True if the dataset was saved, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def shuffle_input(self, name, restore=None):
        """
        Shuffle input 'name' from a dataset.
        For instance, shuffle column 'name' from a dataframe.
        If 'restore' is provided, the data is restored (un-shuffle)
        Args:
            name: Input variable name
            restore: If not None, the input should be restored from this object's data
        Return:
            The original values om that column
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def split(self):
        """ Split dataset into train, test, validate
        First try to use user defined function otherwise call `default_split()` method
        Returns: True if the dataset was split, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def split_idx(self, idx_train, idx_validate, idx_test=None) -> bool:
        ''' Split datasets.
        First split into train, validate and test datasets. Then split inputs
        and outputs (i.e. should invoke `self.in_outs()`).
        Args:
            idx_train: Index (list or array) indicating samples in train dataset
            idx_validate: Index (list or array) indicating samples in validate dataset
            idx_test: Index (list or array) indicating samples in test dataset
        '''
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def transform(self):
        """ Perform transform step.
        Invoke a user defined function, if none available, call `default_transform()` method.
        Returns: True if the transform was performed, False otherwise
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def zero_input(self, name, restore):
        """
        Set all inputs to zero
        For instance, set to zero column 'name' from a dataframe
        If 'restore' is provided, the data is restored (un-zeroed)
        Args:
            name: Input variable name
            restore: If not None, the input should be restored from this object's data
        Return:
            The original values
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")
