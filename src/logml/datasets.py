import numpy as np
import random

from collections import namedtuple
from .config import CONFIG_DATASET
from .files import MlFiles
from .registry import MlRegistry, DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT, DATASET_LOAD, DATASET_PREPROCESS, DATASET_SAVE, DATASET_SPLIT, DATASET_TRANSFORM

InOut = namedtuple('InOut', ['x', 'y'])

class Datasets(MlFiles):
    '''
    Datasets class cotaining trainig, test and validation datasets
        self.dataset             : Original dataset
        self.dataset_test        : Test dataset (split from self.dataset)
        self.dataset_test_xy     : Test dataset, inputs and outputs
        self.dataset_train       : Train dataset (split of self.dataset_
        self.dataset_train_xy    : Train dataset inputs and outputs
        self.dataset_validate    : Valiation dataset (split from self.dataset)
        self.dataset_validate_xy : Validation dataset inputs and outputs
    '''
    def __init__(self, config, set_config=True):
        super().__init__(config, CONFIG_DATASET)
        self.dataset_path = None
        self.dataset_name = None
        self.dataset = None
        self.dataset_xy = InOut(None, None)
        self.dataset_test = None
        self.dataset_test_xy = InOut(None, None)
        self.dataset_train = None
        self.dataset_train_xy = InOut(None, None)
        self.dataset_validate = None
        self.dataset_validate_xy = InOut(None, None)
        self.do_not_load_pickle = False
        self.do_not_save = False
        self.is_use_default_in_out = True
        self.is_use_default_split = True
        self.is_use_default_transform = True
        self.operations = [DATASET_TRANSFORM, DATASET_AUGMENT, DATASET_PREPROCESS, DATASET_SPLIT]
        self.operations_done = set()
        self.outputs = list()
        self.should_save = False
        if set_config:
            self._set_from_config()

    def augment(self):
        return self.invoke_augment()

    def __call__(self):
        '''
        Load (or create) dataset, then augment, proprocess and split
        Save at each step for faster processing / consistency
        '''
        self._debug("Start")
        self.should_save = False
        if self.load():
            self._debug("Dataset loaded")
        elif self.create():
            self._debug("Dataset created")
            self.should_save = True
        else:
            self._debug("Could not load or create dataset")
            return False
        # Perform operations
        for op in self.operations:
            if self.do(op):
                self.should_save = True
        # Get inputs / outputs
        self.dataset_xy = self.in_out(self.dataset, 'all')
        self.dataset_test_xy = self.in_out(self.dataset_test, 'test')
        self.dataset_train_xy = self.in_out(self.dataset_train, 'train')
        self.dataset_validate_xy = self.in_out(self.dataset_validate, 'validate')
        if self.should_save:
            self.save()
        self._debug("End")
        return True

    def _config_sanity_check(self):
        '''
        Check parameters from config.
        Return True on success, False if there are errors
        '''
        if self.dataset_path is None:
            self._fatal_error("Missing 'dataset_path' parameters in config file '{self.config.config_file}', section {CONFIG_DATASET}")
        if self.dataset_name is None:
            self._fatal_error("Missing 'dataset_name' parameters in config file '{self.config.config_file}', section {CONFIG_DATASET}")
        return True

    def create(self):
        return self.invoke_create()

    def default_in_out(self, df, name):
        ''' Default method for getting inputs / outputs '''
        return InOut(None, None)

    def default_load(self):
        ''' Load dataset from pickle file. Return new dataset '''
        if self.do_not_load_pickle:
            return False
        file_name = self.get_file_name()
        self._debug(f"Load dataset from file '{file_name}'")
        ds = self._load_pickle(file_name, 'Load dataset')
        if ds:
            # Copy data from loaded dataset
            self.dataset = ds.dataset
            self.dataset_test = ds.dataset_test
            self.dataset_train = ds.dataset_train
            self.dataset_validate = ds.dataset_validate
            self.operations = ds.operations
            self.operations_done = ds.operations_done
        return ds is not None

    def default_save(self):
        ''' Default implementation of '@dataset_save' '''
        return self._save_pickle(self.get_file_name(), 'Save dataset', self)

    def default_split(self):
        '''
        Default implementation for '@dataset_split'
        Assumptions:
            1) self.dataset object is iterable
            2) Parameter 'split_test' and 'split_validate' are defined such that
                2.a) split_test > 0
                2.b) split_validate > 0
                2.c) split_test + split_validate < 1
        It returns three list of 'samples': train, validate, test
        '''
        # Is datasets iterable?
        self._debug(f"Using default split method")
        # Are split parameters defined?
        kwargs = self.config.get_parameters_functions(DATASET_SPLIT)
        for key in ['split_test', 'split_validate']:
            if key not in kwargs:
                self._debug(f"Cannot run default _split: Parameter '{key}' not in defined for section '{DATASET_SPLIT}' in YAML file")
                return False
        split_test, split_validate = kwargs['split_test'], kwargs['split_validate']
        # Split dataset into three lists
        idx_train, idx_validate, idx_test = list(), list(), list()
        for idx in range(len(self.dataset)):
            r = random.random()
            if r <= split_validate:
                idx_validate.append(idx)
            elif r <= split_test + split_validate:
                idx_test.append(idx)
            else:
                idx_train.append(idx)
        idx_train, idx_validate, idx_test = np.array(idx_train), np.array(idx_validate), np.array(idx_test)
        return self.split_idx(idx_train, idx_validate, idx_test)

    def default_transform(self):
        " Default implementation for '@dataset_transform' "
        self._debug(f"Default dataset transform not defined, skipping")
        return False

    def do(self, op):
        ''' Perform an abstract operation on a dataset '''
        self._debug(f"Dataset operation '{op}': Start")
        if op in self.operations_done:
            self._debug(f"Operation '{op}' has been done. Skipping")
            return True
        if op == DATASET_AUGMENT:
            ok = self.augment()
        elif op == DATASET_CREATE:
            ok = self.create()
        elif op == DATASET_PREPROCESS:
            ok = self.preprocess()
        elif op == DATASET_SPLIT:
            ok = self.split()
        elif op == DATASET_TRANSFORM:
            ok = self.transform()
        else:
            raise ValueError(f"Unknown dataset operation '{op}'")
        if ok:
            self.operations_done.add(op)
        self._debug(f"Dataset operation '{op}': End, ok={ok}")
        return ok

    def __getitem__(self, key):
        return self.dataset[key]

    def get_file_name(self, dataset_type=None, ext='pkl'):
        ''' Create a file name for dataset '''
        self._debug(f"dataset_type={dataset_type}, ext='{ext}'")
        return self._get_file_name(self.dataset_path, self.dataset_name, dataset_type, ext)

    def get_test(self):
        if self.dataset_test is None:
            self._info(f"Test dataset not found")
        return self.dataset_test

    def get_test_xy(self):
        if self.dataset_test_xy.x is None:
            self._info(f"Test dataset not found")
        return self.dataset_test_xy

    def get_train(self):
        if self.dataset_train is None:
            self._info(f"Training dataset not found, using whole dataset")
            return self.dataset
        return self.dataset_train

    def get_train_xy(self):
        if self.dataset_train_xy.x is None:
            self._info(f"Training dataset not found, using whole dataset")
            return self.dataset_xy
        return self.dataset_train_xy

    def get_validate(self):
        if self.dataset_validate is None:
            self._info(f"Validate dataset not found, using whole dataset")
            return self.dataset
        return self.dataset_validate

    def get_validate_xy(self):
        if self.dataset_validate_xy.x is None:
            self._info(f"Validate dataset not found, using whole dataset")
            return self.dataset_xy
        return self.dataset_validate_xy

    def in_out(self, ds, name):
        '''
        Split dataset inputs and outputs from dataset 'ds'
        Returns an InOut named tuple
        '''
        if ds is None:
            return InOut(None, None)
        self._debug(f"Get inputs & outputs from dataset '{name}'")
        (invoked, ret) = self.invoke_in_out(ds, name)
        if invoked:
            return ret
        # We provide a default implementation for 'in_out'
        if self.is_use_default_in_out:
            return self.default_in_out(ds, name)
        return InOut(None, None)

    def invoke_augment(self):
        " Invoke user defined function for '@dataset_augment' "
        args = [self.dataset]
        (invoked, ret) = self.config.invoke(DATASET_AUGMENT, 'Augment', args)
        if invoked:
            self.dataset = ret
        return invoked

    def invoke_create(self):
        " Invoke user defined function for '@dataset_create' "
        (invoked, ret) = self.config.invoke(DATASET_CREATE, 'Create dataset')
        if invoked:
            self.dataset = ret
        return invoked

    def invoke_in_out(self, ds, name):
        " Invoke user defined function for '@dataset_inout' "
        args = [ds]
        (invoked, ret) = self.config.invoke(DATASET_INOUT, f"InOut'{name}", args)
        if invoked:
            if ret is None or len(ret) != 2:
                self._fatal_error(f"User defined function '{DATASET_INOUT}' should return a tuple, but it returned '{ret}'")
            x, y  = ret
            return True, InOut(x, y)
        return False, InOut(None, None)

    def invoke_load(self):
        " Invoke user defined function fo '@dataset_load' "
        (invoked, ret) = self.config.invoke(DATASET_LOAD, 'Load dataset')
        if invoked:
            self.dataset = ret
        return invoked

    def invoke_preprocess(self):
        " Invoke user defined function for '@dataset_preprocess' "
        args = [self.dataset]
        (invoked, ret) = self.config.invoke(DATASET_PREPROCESS, 'Preprocess', args)
        if invoked:
            self.dataset = ret
        return invoked

    def invoke_save(self):
        " Invoke user defined function for '@dataset_save' "
        args = [self.dataset, self.dataset_train, self.dataset_test, self.dataset_validate]
        (invoked, ret) = self.config.invoke(DATASET_SAVE, 'Save dataset', args)
        return invoked

    def invoke_split(self):
        " Invoke user defined function for '@dataset_split' "
        args = [self.dataset]
        (invoked, ret) = self.config.invoke(DATASET_SPLIT, 'Split dataset', args)
        if invoked:
            # The returned dataset is a tuple, unpack it
            self.dataset_train, self.dataset_validate, self.dataset_test = ret
        return invoked

    def invoke_transform(self):
        " Invoke user defined function for '@dataset_transform' "
        args = [self.dataset]
        (invoked, ret) = self.config.invoke(DATASET_TRANSFORM, 'Transform dataset', args)
        if invoked:
            self.dataset = ret
        return invoked

    def __len__(self):
        return 0 if self.dataset is None else len(self.dataset)

    def load(self):
        ''' Try to load dataset, first from pickle otherwise from user defined function '''
        if self.default_load():
            self.should_save = False
            return True
        if self.invoke_load():
            self.should_save = True
            return True
        return False

    def preprocess(self):
        return self.invoke_preprocess()

    def reset(self):
        ''' Reset fields '''
        self.dataset = None
        self.dataset_test = None
        self.dataset_train = None
        self.dataset_validate = None
        self.operations_done = set()
        self.should_save = False

    def save(self):
        ''' Try to save dataset, first use user defined function otherwise save to pickle otherwise'''
        if self.do_not_save:
            return False
        return True if self.invoke_save() else self.default_save()

    def split(self):
        " Split dataset into train, test, validate "
        ret = self.invoke_split()
        if ret:
            return ret
        # We provide a default implementation for dataset_split
        if self.is_use_default_split:
            return self.default_split()
        return False

    def split_idx(self, idx_train, idx_validate, idx_test=None):
        ''' Split dataset using an index list / array '''
        len_test = len(idx_test) if idx_test is not None else 0
        self._debug(f"Split dataset by idx. Lengths, train: {len(idx_train)}, validate: {len(idx_validate)}, test:{len_test}")
        self.dataset_train = self[idx_train]
        self.dataset_validate = self[idx_validate]
        if len_test > 0:
            self.dataset_test = self[idx_test]
        return True

    def transform(self):
        " Transform dataset "
        ret = self.invoke_transform()
        if ret:
            return ret
        # We provide a default implementation
        if self.is_use_default_transform:
            return self.default_transform()
        return False
