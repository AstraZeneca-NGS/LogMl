import copy
import pandas as pd
import numpy as np

from sys import getsizeof

from ..datasets import Datasets
from ..datasets_base import InOut
from .df_augment import DfAugment
from .df_preprocess import DfPreprocess
from ...util.mem import bytes2human, memory


def _copy_df(df):
    """ Copy a DataFrame """
    return df.copy() if df is not None else None


def _copy_inout(inout):
    """ Copy an InOut object """
    return InOut(_copy_df(inout.x), _copy_df(inout.y)) if inout is not None else InOut(None, None)


def mem_type_shape(obj):
    mem, t, shape = memory(obj), type(obj).__name__, None
    if obj is None:
        mem = 0
    elif hasattr(obj, 'shape'):
        shape = obj.shape
    elif isinstance(obj, InOut):
        mem = bytes2human(getsizeof(obj.x) + getsizeof(obj.y))
        shape = f"(x: {shape_or_none(obj.x)}, y:{shape_or_none(obj.y)})"
    out = f"mem={mem} type={t}"
    if shape:
        out += f" shape={shape}"
    return out


def shape_or_none(df):
    return 'None' if df is None else df.shape


class DatasetsDf(Datasets):
    """
    A dataset based on a Pandas DataFrame
    (i.e. type(self.dataset) = DataFrame)
    """
    def __init__(self, config, model_type=None, set_config=True):
        super().__init__(config, set_config=False)
        self.count_na = dict()  # Count missing values for each field
        self.categories = dict()  # Convert these fields to categorical
        self.dataset_ori = None
        self.dataset_preprocess = None
        self.dates = list()  # Convert these fields to dates and expand to multiple columns
        self.model_type = model_type
        if set_config:
            self._set_from_config()

    def clone(self, deep=False):
        ds_clone = copy.copy(self)
        if deep:
            ds_clone.dataset = _copy_df(self.dataset)
            ds_clone.dataset_test = _copy_df(self.dataset_test)
            ds_clone.dataset_train = _copy_df(self.dataset_train)
            ds_clone.dataset_validate = _copy_df(self.dataset_validate)
            ds_clone.dataset_xy = _copy_inout(self.dataset_xy)
            ds_clone.dataset_test_xy = _copy_inout(self.dataset_test_xy)
            ds_clone.dataset_train_xy = _copy_inout(self.dataset_train_xy)
            ds_clone.dataset_validate_xy = _copy_inout(self.dataset_validate_xy)
        return ds_clone

    def _columns_na(self, df):
        """
        Get a columns that should b used for a new 'NA' dataframe (i.e. a dataframe
        of missing data indicators)
        """
        if df is None:
            return None
        # Keep output and 'na_columns'
        cols_na = set([c for c in df.columns if c in self.outputs or c in self.dataset_preprocess.na_columns])
        # Keep cathegorical if they have 'na'
        for c in df.columns:
            if c in self.dataset_preprocess.category_column:
                cat_col = self.dataset_preprocess.category_column[c]
                cat_isna = cat_col.isna().astype('int')
                if cat_isna.sum() > 0:
                    cols_na.add(c)
        return cols_na

    def create(self):
        """ Create dataset """
        ret = super().create()
        if ret:
            return True
        return self._create_from_csv()

    def _create_from_csv(self):
        """ Create from CSV: Load CSV dataframe """
        # Loading from CSV
        self._debug("Start")
        ret = self._load_from_csv()
        return ret

    def default_augment(self):
        """ Default implementation for '@dataset_augment' """
        self._debug(f"Dataset augment (default): Start")
        self.dataset_augment = DfAugment(self.dataset, self.config, self.outputs, self.model_type)
        self.dataset = self.dataset_augment()
        self._debug(f"Dataset augment: End")
        return True

    def default_in_out(self, df, name):
        """ Get inputs and outputs """
        self._debug(f"Default method, inputs & outputs from dataset '{name}'")
        outs = self.outputs
        if outs:
            # Split input and output variables
            self._debug(f"Default method, inputs & outputs from dataframe '{name}': Outputs {outs}")
            if len(outs) == 1:
                outs = outs[0]
            x, y = df.drop(outs, axis=1), df.loc[:, outs]
        else:
            self._debug(f"Default method, inputs & outputs from dataframe '{name}': No outputs defined")
            # Do not split: e.g. unsupervised learning
            x, y = df, None
        return InOut(x, y)

    def default_preprocess(self):
        """ Default implementation for '@dataset_preprocess' """
        self._debug(f"Using default dataset preprocess for dataset type 'DataFrame': Start")
        if self.dataset_ori is None:
            # Keep a copy.copy of the original dataset
            self.dataset_ori = self.dataset
        self.dataset_preprocess = DfPreprocess(self, self.config, self.outputs, self.model_type)
        self.dataset = self.dataset_preprocess()
        self._debug(f"Dataset preprocess: End")
        return True

    def default_save(self):
        """ Default implementation of '@dataset_save' """
        super().default_save()
        filename = self.get_file(ext='preproc_augment.csv')
        self._info(f"Saving dataframe to '{filename}'")
        return self._save_csv(filename, "Save as CSV", self.dataset, save_index=True)

    def get_input_names(self):
        """ Get dataset's input names """
        outs = self.get_output_names()
        return [c for c in self.dataset.columns if c not in outs]

    def _get_dfs(self, inputs=True, names=False):
        if inputs:
            dfs = [self.dataset, self.dataset_train, self.dataset_validate, self.dataset_test, self.dataset_xy.x, self.dataset_train_xy.x, self.dataset_validate_xy.x, self.dataset_test_xy.x]
            dfs_names = ['dataset', 'dataset_train', 'dataset_validate', 'dataset_test', 'dataset_xy.x', 'dataset_train_xy.x', 'dataset_validate_xy.x', 'dataset_test_xy.x']
        else:
            dfs = [self.dataset, self.dataset_train, self.dataset_validate, self.dataset_test]
            dfs_names = ['dataset', 'dataset_train', 'dataset_validate', 'dataset_test']
        if names:
            return [df for df in dfs if df is not None], [n for df, n in zip(dfs, dfs_names) if df is not None]
        return [df for df in dfs if df is not None]

    def _get_dataframe_na(self, df, cols_na):
        """ Get a new dataframe having only columns indicating missing data """
        if df is None:
            return None
        # Remove
        cols_to_remove = [c for c in df.columns if c not in cols_na]
        self._debug(f"Get dataframe NA: Removing columns: {cols_to_remove}")
        df_na = df.drop(columns=cols_to_remove)
        # Change categorical, only keep 'na' data
        for c in df.columns:
            if c in self.outputs:
                pass
            elif c in cols_na and c in self.dataset_preprocess.category_column:
                cat_col = self.dataset_preprocess.category_column[c]
                df_na[c] = cat_col.isnull().astype('float')
                self._debug(f"Get dataframe NA: Adding cathegorical column '{c}', number of NA: {df_na[c].sum()}")
        return df_na

    def get_datasets_na(self):
        """ Create a dataset of 'missing value indicators' """
        if self.dataset_preprocess is None:
            self._error("Cannot create 'missing' dataset")
            return None
        # Create a new datasets, with same parameters
        self._debug(f"Creating 'missing' dataset: Start")
        dsna = self.clone()
        dsna.reset()
        # Which columns should we keep?
        cols_na = self._columns_na(self.dataset)
        self._debug(f"Creating 'missing' dataset: Columns {cols_na}")
        if len(cols_na) <= len(self.outputs):
            self._debug(f"Creating 'missing' dataset: No input columns selected, returning None")
            return None
        # Copy all self.dataset*, removing non 'na' variables
        dsna.outputs = self.outputs
        dsna.dataset = self._get_dataframe_na(self.dataset, cols_na)
        dsna.dataset_test = self._get_dataframe_na(self.dataset_test, cols_na)
        dsna.dataset_train = self._get_dataframe_na(self.dataset_train, cols_na)
        dsna.dataset_validate = self._get_dataframe_na(self.dataset_validate, cols_na)
        # Split x,y for all datasets
        dsna.in_outs()
        self._debug(f"Creating 'missing' dataset: End. {dsna}")
        return dsna

    def __getitem__(self, key):
        # Make sure we create a copy of the DataFrame so that we don't run
        # into 'SettingWithCopyWarning' later
        return self.dataset.iloc[key].copy()

    def _load_from_csv(self):
        """ Load dataframe from CSV """
        csv_file = self.get_file(ext='csv')
        self._debug(f"Loading csv file '{csv_file}'")
        self.dataset = pd.read_csv(csv_file, low_memory=False, parse_dates=self.dates)
        return len(self.dataset) > 0

    def memory(self):
        """
        Return memory consumption as a string
        """
        out = f"Dataset {self.dataset_name} memory usage. total: {mem_type_shape(self)}, "
        out += f"dataset: {mem_type_shape(self.dataset)}, "
        out += f"dataset_test: {mem_type_shape(self.dataset_test)}, "
        out += f"dataset_train: {mem_type_shape(self.dataset_train)}, "
        out += f"dataset_validate: {mem_type_shape(self.dataset_validate)}, "
        out += f"dataset_xy: {mem_type_shape(self.dataset_xy)}, "
        out += f"dataset_test_xy: {mem_type_shape(self.dataset_test_xy)}, "
        out += f"dataset_train_xy: {mem_type_shape(self.dataset_train_xy)}, "
        out += f"dataset_validate_xy: {mem_type_shape(self.dataset_validate_xy)}, "
        return out

    def remove_inputs(self, names):
        [df.drop(columns=names, inplace=True) for df in self._get_dfs()]

    def remove_samples_if_missing(self, name):
        """
        Remove any sample from the dataset if 'name' (an input or output) has a missing value in that sample.
        E.g. In a dataframe, remove any row if column 'y' is NA
        Note: This invalidates any split, so it should be performed before any split operation
        """
        assert(self.dataset_train is None)
        assert(self.dataset_validate is None)
        assert(self.dataset_test is None)
        rows_to_remove = self.dataset[name].isna()
        if rows_to_remove.sum() > 0:
            self._debug(f"Remove samples with missing '{name}': {rows_to_remove.sum()} rows to remove")
            orishape = self.dataset.shape
            self.dataset = self.dataset.loc[~rows_to_remove].copy()
            self._info(f"Remove samples with missing '{name}': Removed {rows_to_remove.sum()} rows, dataFrame previous shape: {orishape}, new shape: {self.dataset.shape}")
        else:
            self._debug(f"Remove samples with missing '{name}': Nothing to remove")

    def set_column(self, col_name, values):
        self.dataset[col_name] = values

    def shuffle_input(self, name, restore=None, new_name=None):
        """
        Shuffle input variable 'name' (in all datasets)
        If 'restore' is assigned, use that data to restore the original values
        If 'new_name' is assigned, create a new column with that name instead of replacing the original column.
        Return: the original column values
        """
        if restore is None:
            self._debug(f"Shuffling column {name}, new_name={new_name}")
        else:
            self._debug(f"Restoring column {name}")
        dfs, dfs_names = self._get_dfs(names=True)
        if restore is None:
            restore = [None] * len(dfs)
        dfs = zip(dfs, restore, dfs_names)
        return [self._shuffle_input(df, name, res, new_name, df_name) for df, res, df_name in dfs]

    def _shuffle_input(self, df, col_name, restore, new_name, df_name):
        """
        Shuffle column 'name' from dataframe df and return the original values
        If 'restore' is assigned, use that data to restore the original values
        Arguments:
            df: DataFrame to shuffle
            name: Columna name to shuffle
            restore: Use this values, instead of shuffling
            new_name: If not None, create a new column with that name. If None, replace original column
            df_name: DataFrame name (used for debugging)
        """
        if df is None:
            return None
        if restore is not None:
            # Restore original data (un-shuffle)
            self._debug(f"Restoring column {col_name}, df_name={df_name}")
            df[col_name] = restore
            return restore
        else:
            # Shuffle column
            self._debug(f"Shuffling column {col_name}, new_name={new_name}, df_name={df_name}")
            x_col = df[col_name].copy()
            c = col_name if new_name is None else new_name
            df[c] = np.random.permutation(x_col)
            return x_col

    def __str__(self):
        return f"Datasets(df). Shapes: all={shape_or_none(self.dataset)}, train={shape_or_none(self.dataset_train)}, validate={shape_or_none(self.dataset_validate)}, test={shape_or_none(self.dataset_test)}"

    def zero_input(self, name, restore=None):
        """
        Zero input variable (i.e. make the column all zeros)
        Return: the original column values
        """
        dfs = [self.dataset, self.dataset_train, self.dataset_validate, self.dataset_test, self.dataset_xy.x, self.dataset_train_xy.x, self.dataset_validate_xy.x, self.dataset_test_xy.x]
        if restore is None:
            restore = [None] * len(dfs)
        dfs = zip(dfs, restore)
        return [self._zero_input(df, name, res) for df, res in dfs]

    def _zero_input(self, df, name, restore):
        """
        Zero column 'name' from dataframe df and return the original values
        If 'restore' is assigned, use that data to restore the original values
        """
        if df is None:
            return None
        if restore is not None:
            # Restore original data (un-shuffle)
            df[name] = restore
            return restore
        else:
            # Zero column
            x_col = df[name].copy()
            df[name] = np.zeros(x_col.shape)
            return x_col
