#!/usr/bin/env python

import hashlib
import numpy as np
import pandas as pd
import re
import traceback

from .category import CategoriesPreprocess
from ...core import MODEL_TYPE_CLASSIFICATION
from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog
from .df_normalize import DfNormalize
from .df_impute import DfImpute
from ...util.sanitize import sanitize_name


def datetime_to_elapsed(dtimes):
    """
    Convert a numpy array of date_times into an 'epoch'
    If there are no missing values, an int64 is returned.
    Otherwise a float64 is used, so we can set missing to 'nan'
    """
    nanosec = 10 ** 9
    if dtimes.isnull().sum() > 0:
        # Use float, so we can use NaN in missing values
        elap = np.zeros(len(dtimes))
        is_null = dtimes.isnull()
        not_null = np.logical_not(is_null)
        elap[not_null] = dtimes[not_null].astype(np.int64) / nanosec
        elap[is_null] = np.nan
    else:
        # No missing values, we can safely use an int64
        elap = dtimes.astype(np.int64) // nanosec
    return elap


def bool_to_int_or_float(x):
    """
    Convert a numpy array of bool to a number
    If there are no missing values, an int8 is returned.
    Otherwise a float16 is used, so we can set missing to 'nan'
    """
    if x.isnull().sum() > 0:
        # Use float, so we can use NaN in missing values
        xf = np.zeros(len(x), dtype='float16')
        is_null = x.isnull()
        not_null = np.logical_not(is_null)
        xf[not_null] = x[not_null].astype(np.int8)
        xf[is_null] = np.nan
    else:
        # No missing values, we can safely use an int8
        return x.astype('int8')


class DfPreprocess(MlLog):
    """
    DataFrame preprocessing: convet categorical, one-hot encoding, impute
    missing data, normalize.

    How it works:
    Calculate some field (i.e. dataframe column) transformations
    and store the results in 'columns_to_add' and 'columns_to_remove'. Then
    apply these changes to the original dataframe to create a new dataframe.
    """

    def __init__(self, datasets, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS)
        self.datasets = datasets
        self.df = datasets.dataset
        self.balance = False
        self.categories = dict()  # Fields to be converted to categorical. Entries are list of categories
        self.category_column = dict()  # Store Pandas categorical definition
        self.columns_to_add = dict()
        self.columns_to_remove = set()
        self.dates = list()  # Convert these fields to dates and expand to multiple columns
        self.impute = dict()
        self.is_sanitize_column_names = True
        self.model_type = model_type
        self.na_columns = set()  # Columns added as 'missing data' indicators
        self.normalize = dict()
        self.normalize_df = None
        self.one_hot = list()  # Convert these fields to 'one hot encoding'
        self.one_hot_max_cardinality = 7
        self.outputs = outputs
        self.remove_missing_outputs = True
        self.remove_columns = list()
        self.remove_columns_after = list()
        self.remove_equal_inputs = True    # Remove columns having the exact same values
        self.skip_nas = set()  # Skip doing "missing data" on these columns (they have been covered somewhere else, e.g. one-hot)
        self.shuffle = False    # Shuffle samples
        self.std_threshold = 0.0  # Drop columns of stddev is less or equal than this threshold
        if set_config:
            self._set_from_config()

    def _add_datepart(self, field_name, prefix=None, time=True):
        """
        Helper function that creates a new dataframe with all columns relevant to
        a date in the column `field_name`.
        Source: fast.ai
        """
        has_na = self.df[field_name].isnull().sum() > 0
        self._info(f"Converting to date: field '{field_name}', dtype: {self.df[field_name].dtype}, has NA / NaT / Null: {has_na}")
        self._make_date(self.df, field_name)
        field = self.df[field_name]
        prefix = prefix if prefix else re.sub('[Dd]ate$', '', field_name)
        attr = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear', 'is_month_end', 'is_month_start',
                'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']
        if time:
            attr = attr + ['hour', 'minute', 'second']
        df = pd.DataFrame()
        for n in attr:
            dpname = f"{prefix}:{n}"
            xi = getattr(field.dt, n.lower())
            if pd.api.types.is_bool_dtype(xi):
                xi = bool_to_int_or_float(xi)
            df[dpname] = xi
            self._info(f"Adding date part '{dpname}', type {df[dpname].dtype}, for field '{field_name}'")
        dpname = f"{prefix}:elapsed"
        df[dpname] = datetime_to_elapsed(field)
        self._info(f"Adding date part '{dpname}', type {df[dpname].dtype}, for field '{field_name}'")
        # Add to replace and remove operations
        self.columns_to_add[field_name] = df
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def _balance(self):
        """ Balance an unbalanced dataset """
        if not self.balance:
            self._debug(f"Balance: Disabled, skipping")
            return True
        if self.model_type != MODEL_TYPE_CLASSIFICATION:
            self._debug(f"Balance: Cannot balance for model type '{self.model_type}', skipping")
            return False
        y = self.df[self.outputs]
        uniq, counts = np.unique(y, return_counts=True)
        self._debug(f"Balance: Start, dataframe shape: {self.df.shape}, percents: {counts / counts.sum()}, counts: {counts}, cathegories: {uniq}")
        counts_max = counts.max()
        for u, c in zip(uniq, counts):
            y = self.df[self.outputs]
            num = counts_max - c
            self._debug(f"Balance: Adding samples for outputs={u}, count={c}, to add: {num}, dataframe shape: {self.df.shape}")
            if num == 0:
                continue
            replace = num > c
            weights = (y == u).astype('float').to_numpy().flatten()
            rows_to_add = self.df.sample(n=num, replace=replace, weights=weights)
            self.df = pd.concat([self.df, rows_to_add])
        y = self.df[self.outputs]
        uniq, counts = np.unique(y, return_counts=True)
        self._debug(f"Balance: End, dataframe shape: {self.df.shape}, percents: {counts / counts.sum()}, counts: {counts}, cathegories: {uniq}")

    def __call__(self):
        """
        Preprocess dataframe columns
        Returns a new (preprocessed) dataset
        """
        if not self.enable:
            self._debug(f"Preprocessing dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', enable='{self.enable}'")
            return self.df
        self._debug(f"Preprocessing dataframe: Start. Shape: {self.df.shape}")
        self._sanitize_column_names()
        self._shuffle()
        self._remove_columns()
        self.convert_dates()
        self.create_categories()
        self.set_df(self.create())  # We have to create the new df before replacing nas, in case the dates have missing data
        self.set_df(self._remove_rows_with_missing_outputs())
        self.nas()
        self.set_df(self.create())
        self._remove_columns_after()
        self.drop_all_na()
        self.drop_zero_std()
        self.drop_equal_inputs()
        self.impute_df = DfImpute(self.df, self.config, self.outputs, self.model_type)
        self.df = self.impute_df()
        self.normalize_df = DfNormalize(self.df, self.config, self.outputs, self.model_type)
        self.df = self.normalize_df()
        self._balance()
        self._debug(f"Preprocessing dataframe: End. Shape: {self.df.shape}")
        return self.df

    def create(self):
        """ Create a new dataFrame based on the previously calculated conversions """
        self._debug(f"Creating transformed dataset: Start")
        # Create new dataFrame
        df_new = self.df.copy()
        # Drop old columns categorical columns
        df_new.drop(list(self.columns_to_remove), axis=1, inplace=True)
        # Join new columns
        if len(self.columns_to_add) > 0:
            dfs = list([df_new])
            dfs.extend(self.columns_to_add.values())
            df_new = pd.concat(dfs, axis=1)
        self.df_new = df_new
        # Reset columns
        self.columns_to_add = dict()
        self.columns_to_remove = set()
        self._debug(f"Creating transformed dataset: End. Shape: {df_new.shape}")
        return df_new

    def create_categories(self):
        """
        Create categories as defined in YAML file.
        This creates both number categories as well as one_hot encoding
        """
        create_cats = CategoriesPreprocess(self.df, self.categories, self.outputs, self.dates, self.one_hot, self.one_hot_max_cardinality)
        create_cats()
        self.columns_to_add.update(create_cats.columns_to_add)
        self.columns_to_remove.update(create_cats.columns_to_remove)
        self.category_column.update(create_cats.category_column)
        self.na_columns.update(create_cats.na_columns)
        self.skip_nas.update(create_cats.skip_nas)

    def convert_dates(self):
        """ Convert all dates """
        self._debug(f"Converting to 'date/time' values: fields {self.dates}")
        count = 0
        for field in self.dates:
            self._add_datepart(field)
        self._debug(f"Converting to 'date/time' values: End")

    def drop_all_na(self):
        " Drop features that have all 'na' values "
        self._debug(f"Dropping columns with all missing values: Start")
        to_remove = list()
        for c in self.df.columns:
            if self.df[c].isnull().all():
                self._info(f"Dropping column '{c}', all values are missing, values head: {list(self.df[c].head())}")
                to_remove.append(c)
        self.df.drop(to_remove, axis=1, inplace=True)
        self._debug(f"Dropping columns with all missing values: End")

    def drop_equal_inputs(self):
        " Remove input columns having the exact same values "
        if not self.remove_equal_inputs:
            self._debug(f"Dropping columns with exact same values disables, skipping")
            return
        self._debug(f"Dropping columns with exact same values: Start")
        column_hash = dict()
        to_remove = list()
        for c in self.df.columns:
            # Create a hash of column 'c'
            c_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df[c], index=True).values).hexdigest()
            if c_hash in column_hash:
                self._info(f"Dropping column '{c}', same values as column {column_hash[c_hash]}, hash {c_hash}")
                to_remove.append(c)
            else:
                column_hash[c_hash] = c
        self.df.drop(to_remove, axis=1, inplace=True)
        self._debug(f"Dropping columns with exact same values: End")

    def drop_zero_std(self):
        " Drop features that have standard deviation below a threshold "
        self._debug(f"Dropping columns with low standard deviation: Start, std_threshold: {self.std_threshold}")
        to_remove = list()
        for c in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[c]):
                continue
            stdev = self.df[c].std()
            if stdev <= self.std_threshold:
                self._info(f"Dropping column '{c}', standard deviation {stdev} <= {self.std_threshold}, values head: {list(self.df[c].head())}")
                to_remove.append(c)
        self.df.drop(to_remove, axis=1, inplace=True)
        self._debug(f"Dropping columns with low standard deviation: End")

    def _make_date(self, df, date_field):
        " Make sure `df[field_name]` is of the right date type. Source: fast.ai "
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
            self._debug(f"Converted field '{date_field}' to {df[date_field].dtype}")

    def nas(self):
        " Add 'missing value' indicators (i.e. '*_na' columns) "
        self._debug(f"Checking NA (missing values): Start")
        for field_name in self.df.columns:
            if field_name not in self.skip_nas:
                self.na(field_name)
        self._debug(f"Checking NA (missing values): End")

    def na(self, field_name):
        " Process 'na' columns (i.e missing data) "
        # Add '*_na' column
        xi = self.df[field_name].copy()
        xi_isna = xi.isna().astype('int8')
        count_na = sum(xi_isna)
        if count_na == 0:
            return False
        df_na = pd.DataFrame()
        name_na = f"{field_name}_na"
        self.na_columns.add(name_na)
        df_na[field_name] = xi
        df_na[name_na] = xi.isna().astype('int8')
        self._info(f"Field '{field_name}' has {count_na} missing values, added column '{name_na}'")
        # Add operations
        self.columns_to_add[field_name] = df_na
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)
        return True

    def _remove_columns(self):
        """ Remove columns (before transformations) """
        self.df_ori = self.df
        self._info(f"Removing columns (before): {self.remove_columns}")
        self.df.drop(self.remove_columns, inplace=True, axis=1)

    def _remove_columns_after(self):
        """ Remove columns (after transformations) """
        self.df_ori = self.df
        self._info(f"Removing columns (after): {self.remove_columns_after}")
        self.df.drop(self.remove_columns_after, inplace=True, axis=1)

    def _remove_rows_with_missing_outputs(self):
        """ Remove rows if output variable/s have missing values c"""
        if not self.remove_missing_outputs:
            self._debug("Remove missing outputs disabled, skipping")
            return
        self._debug(f"Remove samples with missing outputs: Start, outputs: {self.outputs}")
        for n in self.outputs:
            self.datasets.remove_samples_if_missing(n)
        return self.datasets.dataset
        self._debug(f"Remove samples with missing outputs: End")

    def _sanitize_column_names(self):
        """ Sanitize all column names """
        if not self.is_sanitize_column_names:
            self._debug("Sanitize column names disabled, skipping")
            return
        self._info("Sanitize column names")
        cols_ori = list(self.df.columns)
        cols_new = [sanitize_name(c) for c in self.df.columns]
        self.df.columns = cols_new
        for i in range(len(self.df.columns)):
            if cols_new[i] != cols_ori[i]:
                self._info(f"Sanitize column names: Changed '{cols_ori[i]}' to '{cols_new[i]}'")

    def set_df(self, df):
        self.df = df
        self.datasets.dataset = self.df
        return self.df

    def _shuffle(self):
        """ Shuffle samples in dataset """
        if not self.shuffle:
            self._debug(f"Shuffle samples: Disabled, skipping")
            return True
        self._debug(f"Shuffle samples: Start")
        # The following line samples 100 of the rows 'in place' without replacement
        # Reference: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        # Note: Resset with `drop=True` means not to create a new columns having the old index data
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self._debug(f"Shuffle samples: End")
        return True
