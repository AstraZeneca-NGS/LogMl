import copy
import pandas as pd
import numpy as np

from ..datasets import Datasets
from ..datasets_base import InOut
from .df_augment import DfAugment
from .df_preprocess import DfPreprocess
from .df_transform import DfTransform

from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype


def _copy_df(df):
    ''' Copy a DataFrame '''
    return df.copy() if df is not None else None


def _copy_inout(inout):
    ''' Copy an InOut object '''
    return InOut(_copy_df(inout.x), _copy_df(inout.y)) if inout is not None else InOut(None, None)


class DatasetsDf(Datasets):
    '''
    A dataset based on a Pandas DataFrame
    (i.e. Dataset.dataset must be a DataFrame)
    '''
    def __init__(self, config, model_type=None, set_config=True):
        super().__init__(config, set_config=False)
        self.count_na = dict()  # Count missing values for each field
        self.categories = dict()  # Convert these fields to categorical
        self.dataset_ori = None
        self.dataset_preprocess = None
        self.dataset_transform = None
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
        cols_na = set([c for c in df.columns if c in self.outputs or c in self.dataset_transform.na_columns])
        # Keep cathegorical if they have 'na'
        for c in df.columns:
            if c in self.dataset_transform.category_column:
                cat_col = self.dataset_transform.category_column[c]
                cat_isna = cat_col.isna().astype('int')
                if cat_isna.sum() > 0:
                    cols_na.add(c)
        return cols_na

    def create(self):
        ''' Create dataset '''
        ret = super().create()
        if ret:
            return True
        return self._create_from_csv()

    def _create_from_csv(self):
        ''' Create from CSV: Load CSV and transform dataframe '''
        # Loading from CSV
        self._debug("Start")
        ret = self._load_from_csv()
        return ret

    def default_augment(self):
        " Default implementation for '@dataset_augment' "
        self._debug(f"Dataset preprocess (default): Start")
        self.dataset_augment = DfAugment(self.dataset, self.config, self.outputs, self.model_type)
        self.dataset = self.dataset_augment()
        self._debug(f"Dataset preprocess: End")
        return True

    def default_in_out(self, df, name):
        ''' Get inputs and outputs '''
        self._debug(f"Default method, inputs & outputs from dataset '{name}'")
        outs = self.outputs
        if outs:
            # Split input and output variables
            self._debug(f"Default method, inputs & outputs from dataframe '{name}': Outputs {outs}")
            x, y = df.drop(outs, axis=1), df.loc[:, outs]
        else:
            self._debug(f"Default method, inputs & outputs from dataframe '{name}': No outputs defined")
            # Do not split: e.g. unsupervised learning
            x, y = df, None
        return InOut(x, y)

    def default_preprocess(self):
        " Default implementation for '@dataset_preprocess' "
        self._debug(f"Using default dataset preprocess for dataset type 'DataFrame': Start")
        if self.dataset_ori is None:
            # Keep a copy.copy of the original dataset
            self.dataset_ori = self.dataset
        self.dataset_preprocess = DfPreprocess(self.dataset, self.config, self.outputs, self.model_type)
        self.dataset = self.dataset_preprocess()
        self._debug(f"Dataset preprocess: End")
        return True

    def default_transform(self):
        " Default implementation for '@dataset_transform' "
        self._debug(f"Using default dataset transform for dataset type 'DataFrame'")
        if self.dataset_ori is None:
            # Keep a copy.copy of the original dataset
            self.dataset_ori = self.dataset
        self.dataset_transform = DfTransform(self.dataset, self.config, self.outputs)
        self.dataset = self.dataset_transform()
        self._debug(f"End: Columns after transform are {list(self.dataset.columns)}")
        return True

    def get_input_names(self):
        """ Get dataset's input names """
        return self.dataset.columns

    def _get_dataframe_na(self, df, cols_na):
        """ Get a new dataframe having only columns indicating missing data """
        if df is None:
            return None
        # Remove
        cols_to_remove = [c for c in df.columns if c not in cols_na]
        self._debug(f"Get dataframe NA: Removing columns: {cols_to_remove}")
        df_na = df.drop(columns=cols_to_remove)
        # Change cathegorical, only keep 'na' data
        for c in df.columns:
            if c in self.outputs:
                pass
            elif c in cols_na and c in self.dataset_transform.category_column:
                cat_col = self.dataset_transform.category_column[c]
                df_na[c] = cat_col.isna().astype('int')
                self._debug(f"Get dataframe NA: Adding cathegorical column '{c}', number of NA: {df_na[c].sum()}")
        return df_na

    def get_datasets_na(self):
        """ Create a dataset of 'missing value indicators' """
        if self.dataset_transform is None:
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
        self._debug(f"Creating 'missing' dataset: End")
        return dsna

    def __getitem__(self, key):
        # Make sure we create a copy of the DataFrame so that we don't run
        # into 'SettingWithCopyWarning' later
        return self.dataset.iloc[key].copy()

    def _load_from_csv(self):
        ''' Load dataframe from CSV '''
        csv_file = self.get_file_name(ext='csv')
        self._debug(f"Loading csv file '{csv_file}'")
        self.dataset = pd.read_csv(csv_file, low_memory=False, parse_dates=self.dates)
        return len(self.dataset) > 0

    def set_column(self, col_name, values):
        self.dataset[col_name] = values

    def shuffle_input(self, name, restore=None):
        """
        Shuffle input variable 'name' (in all datasets)
        If 'restore' is assigned, use that data to restore the original values
        Return: the original column values
        """
        dfs = [self.dataset, self.dataset_train, self.dataset_validate, self.dataset_test, self.dataset_xy.x, self.dataset_train_xy.x, self.dataset_validate_xy.x, self.dataset_test_xy.x]
        # Dataset names (used for debugging)
        dfs_names = ['dataset', 'dataset_train', 'dataset_validate', 'dataset_test', 'dataset_xy.x', 'dataset_train_xy.x', 'dataset_validate_xy.x', 'dataset_test_xy.x']
        if restore is None:
            restore = [None] * len(dfs)
        dfs = zip(dfs, restore, dfs_names)
        return [self._shuffle_input(df, name, res, df_name) for df, res, df_name in dfs]

    def _shuffle_input(self, df, col_name, restore, df_name=None):
        """
        Shuffle column 'name' from dataframe df and return the original values
        If 'restore' is assigned, use that data to restore the original values
        Arguments:
            df: DataFrame to shuffle
            name: Columna name to shuffle
            restore: Use this values, instead of shuffling
            df_name: DataFrame name (used for debugging)
        """
        if df is None:
            return None
        if restore is not None:
            # Restore original data (un-shuffle)
            df[col_name] = restore
            return restore
        else:
            # Shuffle column
            x_col = df[col_name].copy()
            df[col_name] = np.random.permutation(x_col)
            return x_col

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
        x_col = self.dataset[col_name]
        self.dataset[col_name] = np.random.permutation(x_col)
        return x_col

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
