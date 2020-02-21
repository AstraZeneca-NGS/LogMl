#!/usr/bin/env python

import hashlib
import numpy as np
import pandas as pd
import re
import traceback

from collections import namedtuple
from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog
from .df_normalize import DfNormalize
from .df_impute import DfImpute
from ...util.sanitize import sanitize_name


class DfPreprocess(MlLog):
    '''
    DataFrame preprocessing: convet categorical, one-hot encoding, impute
    missing data, normalize.

    How it works:
    Calculate some field (i.e. dataframe column) transformations
    and store the results in 'columns_to_add' and 'columns_to_remove'. Then
    apply these changes to the original dataframe to create a new dataframe.
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS)
        self.df = df
        self.balance = False
        self.categories = dict()  # Fields to be converted to categorical. Entries are list of categories
        self.category_column = dict()  # Store Pandas categorical definition
        self.columns_to_add = dict()
        self.columns_to_remove = set()
        self.na_columns = set()     # Columns added as 'missing data' indicators
        self.dates = list()  # Convert these fields to dates and expand to multiple columns
        self.is_sanitize_column_names = True
        self.model_type = model_type
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
        '''
        Helper function that creates a new dataframe with all columns relevant to
        a date in the column `field_name`.
        Source: fast.ai
        '''
        self._info(f"Converting to date: field '{field_name}'")
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
                xi = xi.astype('int8')
            df[dpname] = xi
            self._info(f"Adding date part '{dpname}', type {df[dpname].dtype}, for field '{field_name}'")
        df[f'{prefix}:elapsed'] = field.astype(np.int64) // 10 ** 9
        # Add to replace and remove operations
        self.columns_to_add[field_name] = df
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def _balance(self):
        """ Balance an unbalanced dataset """
        if not self.balance:
            self._debug(f"Balance: Disabled, skipping")
            return True
        if self.model_type != 'classification':
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
        self._remove_rows_with_missing_outputs()
        self._remove_columns()
        self.convert_dates()
        self.create_categories()
        self.df = self.create()  # We have to create the new df before replacing nas, in case the dates have missing data
        self.nas()
        self.df = self.create()
        self._remove_columns_after()
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
        # for c in self.columns_to_add:
        #     self._debug(f"Creating transformed dataset: Adding columns '{c}'")
        #     df_new = df_new.join(self.columns_to_add[c])
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
        '''
        Create categories as defined in YAML file.
        This creates both number categories as well as one_hot encoding
        '''
        # Forced categories from YAML config
        self._debug(f"Converting to categorical: Start")
        self.category_re_expand()
        for field_name in self.df.columns:
            if not self.is_categorical_column(field_name):
                continue
            if field_name in self.categories:
                self._create_category(field_name)
            elif field_name in self.one_hot or self.should_be_one_hot(field_name):
                self._create_one_hot(field_name)
            else:
                self._create_category(field_name, self.categories.get(field_name))
        self._debug(f"Converting to categorical: End")

    def _create_category(self, field_name):
        " Convert field to category numbers "
        cat_values = self.categories[field_name]
        categories, na_as_zero = None, True
        if isinstance(cat_values, list):
            categories = cat_values
        elif isinstance(cat_values, dict):
            categories = cat_values.get('values')
            na_as_zero = cat_values.get('na_as_zero', True)
        self._debug(f"Converting to category: field '{field_name}', categories: {categories}")
        xi = self.df[field_name]
        xi_cat = xi.astype('category').cat.as_ordered()
        # Categories can be either 'None' or a list
        if categories:
            xi_cat.cat.set_categories(categories, ordered=True, inplace=True)
        self.category_column[field_name] = xi_cat
        df_cat = pd.DataFrame()
        codes = xi_cat.cat.codes
        add_to_codes = 0
        missing_values = codes < 0
        if np.any(missing_values):
            if field_name in self.outputs and self.remove_missing_outputs:
                # We need to remove these missing outputs as well. These outputs
                # might be created when we forced the cathegory values. For
                # instance the real cathegories are ['a', 'b', 'c'] and we forced them
                # to ['a', 'b'], all the input having values 'c' will now be 'NA'.
                # Since 'remove_missing_outputs' optione is active, we have to remove these new 'NA' rows
                self._remove_rows_with_missing_outputs(rows_to_remove=missing_values)
                add_to_codes = 0
            else:
                # Note: Add one so that "missing" is zero instead of "-1"
                add_to_codes = 1 if na_as_zero else 0
                self._debug(f"Converting to category: field '{field_name}': Missing values, there are {(codes < 0).sum()} codes < 0). Adding {add_to_codes} to convert missing values to '{0 if na_as_zero else -1}'")
        df_cat[field_name] = codes + add_to_codes
        # Add to replace and remove operations
        self.columns_to_add[field_name] = df_cat
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)
        self._info(f"Converted to category: field '{field_name}', categories: {list(xi_cat.cat.categories)}")

    def _create_one_hot(self, field_name):
        " Create a one hot encodig for 'field_name' "
        self._info(f"Converting to one-hot: field '{field_name}'")
        has_na = self.df[field_name].isna().sum() > 0
        self._debug(f"Converting to one-hot: field '{field_name}', has missing data: {has_na}")
        df_one_hot = pd.get_dummies(self.df[field_name], dummy_na=has_na)
        self.rename_category_cols(df_one_hot, f"{field_name}:")
        if has_na:
            self.na_columns.add(f"{field_name}:nan")
        # Add to transformations
        self.columns_to_add[field_name] = df_one_hot
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def convert_dates(self):
        ''' Convert all dates '''
        self._debug(f"Converting to 'date/time' values: fields {self.dates}")
        count = 0
        for field in self.dates:
            self._add_datepart(field)
        self._debug(f"Converting to 'date/time' values: End")

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

    def is_categorical_column(self, field_name):
        " Is column 'field_name' a categorical column in the dataFrame? "
        return (self.df[field_name].dtype == 'O') and (field_name not in self.dates)

    def _make_date(self, df, date_field):
        " Make sure `df[field_name]` is of the right date type. Source: fast.ai "
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

    def category_re_expand(self):
        """
        Find all matches for regular expressions and update self.categories with matched values
        """
        categories_add = dict()
        for regex in self.categories:
            if len(regex) < 1:
                self._debug(f"Bad entry for 'categories': {regex}")
                continue
            for fname in self.df.columns:
                try:
                    if re.match(regex, fname) is not None:
                        self._debug(f"Field name '{fname}' matches regular expression '{regex}': Using values {self.categories[regex]}")
                        categories_add[fname] = self.categories[regex]
                except Exception as e:
                    self._error(f"Category regex: Error trying to match regular expression: '{regex}'\nException: {e}\n{traceback.format_exc()}")
        # Update dictionary with regex matched values
        self.categories.update(categories_add)

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
        ''' Remove columns (before transformations) '''
        self.df_ori = self.df
        self._info(f"Removing columns (before): {self.remove_columns}")
        self.df.drop(self.remove_columns, inplace=True, axis=1)

    def _remove_columns_after(self):
        ''' Remove columns (after transformations) '''
        self.df_ori = self.df
        self._info(f"Removing columns (after): {self.remove_columns_after}")
        self.df.drop(self.remove_columns_after, inplace=True, axis=1)

    def _remove_rows_with_missing_outputs(self, rows_to_remove=None):
        ''' Remove rows if output variable/s are missing '''
        if not self.remove_missing_outputs:
            self._debug("Remove missing outputs disabled, skipping")
            return
        self._debug(f"Remove samples with missing outputs: Start, outputs: {self.outputs}")
        if rows_to_remove is None:
            rows_to_remove = self.df[self.outputs].isna().any(axis=1)
        if rows_to_remove.sum() > 0:
            self.df_ori = self.df
            self.df = self.df.loc[~rows_to_remove].copy()
            self._info(f"Remove samples with missing outputs: Removed {rows_to_remove.sum()} rows, dataFrame previous shape: {self.df_ori.shape}, new shape: {self.df.shape}")
        self._debug(f"Remove samples with missing outputs: End")

    def rename_category_cols(self, df, prepend):
        '''
        Rename dataFrame columns by prepending a string and sanitizing the name
        Used to rename columns of a 'one hot' encoding
        '''
        names = dict()
        for c in df.columns:
            name = f"{prepend}{sanitize_name(c)}"
            names[c] = name
        df.rename(columns=names, inplace=True)

    def _sanitize_column_names(self):
        ''' Sanitize all column names '''
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

    def should_be_one_hot(self, field_name):
        " Should we convert to 'one hot' encoding? "
        if field_name in self.outputs:
            return False
        xi = self.df[field_name]
        xi_cat = xi.astype('category')
        count_cats = len(xi_cat.cat.categories)
        # Note: If there are only two categories, it already is "one-hot"
        return count_cats > 2 and count_cats <= self.one_hot_max_cardinality

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
