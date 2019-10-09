#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from ...core.config import CONFIG_DATASET_TRANSFORM
from ...core.log import MlLog

sanitize_valid_chars = set('_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
sanitize_dict = {
    '+': '_plus_',
    '-': '_',
    '=': '_eq_',
    '<': '_less_than_',
    '>': '_more_than_',
    '/': '_slash_',
}


def sanitize(s):
    ''' Sanitize a string by only allowing "valid" characters '''
    return ''.join(c if c in sanitize_char else '_' for c in str(s))


def sanitize_name(s):
    ''' Sanitize a string to be used as a variable or column name '''
    return ''.join(sanitize_char(c) for c in str(s))


def sanitize_char(c):
    ''' Sanitize a string by only allowing "valid" characters '''
    if c in sanitize_valid_chars:
        return c
    if c in sanitize_dict:
        return sanitize_dict[c]
    return '_'


class DfTransform(MlLog):
    '''
    Data Frame transformations.
    Calculate some field (i.e. dataframe column) transformations
    and store the results in 'columns_to_add' and 'columns_to_remove'. Then apply these
    changes to the original dataframe to create a new dataframe.
    '''

    def __init__(self, df, config, set_config=True):
        super().__init__(config, CONFIG_DATASET_TRANSFORM)
        self.df = df
        self.categories = dict()  # Fields to be converted to categorical. Entries are list of categories
        self.category_column = dict()  # Store Pandas categorical definition
        self.columns_to_add = dict()
        self.columns_to_remove = set()
        self.dates = list()  # Convert these fields to dates and expand to multiple columns
        self.one_hot = list()  # Convert these fields to 'one hot encoding'
        self.one_hot_max_cardinality = 7
        self.outputs = list()
        self.skip_nas = set()  # Skip doing "missing data" transformation on this column (it has been covered somewhere else, e.g. one-hot)
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
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
                'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        df = pd.DataFrame()
        for n in attr:
            dpname = prefix + n
            xi = getattr(field.dt, n.lower())
            if pd.api.types.is_bool_dtype(xi):
                xi = xi.astype('int8')
            df[dpname] = xi
            self._info(f"Adding date part '{dpname}', type {df[dpname].dtype}, for field '{field_name}'")
        df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
        # Add to replace and remove operations
        self.columns_to_add[field_name] = df
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def __call__(self):
        """
        Preprocess and create a new dataframe from pre-processing transformations
        Returns a new (transformed) dataset
        """
        self.transform()
        self.df = self.create()
        self.drop_zero_std()
        return self.df

    def create(self):
        """ Create a new dataFrame based on the previously calculated conversions """
        # Create new dataFrame
        df_new = self.df.copy()
        # Drop old columns categorical columns
        df_new.drop(list(self.columns_to_remove), axis=1, inplace=True)
        # Join new columns
        for c in self.columns_to_add:
            df_new = df_new.join(self.columns_to_add[c])
        self.df_new = df_new
        return df_new

    def create_categories(self):
        '''
        Create categories as defined in YAML file.
        This creates both number categories as well as one_hot encoding
        '''
        # Forced categories from YAML config
        self._info(f"Converting to categorical: Start")
        for field_name in self.df.columns:
            if not self.is_categorical_column(field_name):
                continue
            if field_name in self.categories.keys():
                self._create_category(field_name, self.categories.get(field_name))
            elif field_name in self.one_hot or self.should_be_one_hot(field_name):
                self._create_one_hot(field_name)
            else:
                self._create_category(field_name, self.categories.get(field_name))
        self._info(f"Converting to categorical: End")

    def _create_category(self, field_name, categories=None):
        " Convert field to category numbers "
        self._info(f"Converting to category: field '{field_name}'")
        xi = self.df[field_name]
        xi_cat = xi.astype('category').cat.as_ordered()
        if categories:
            xi_cat.cat.set_categories(categories, ordered=True, inplace=True)
        self.category_column[field_name] = xi_cat
        df_cat = pd.DataFrame()
        df_cat[field_name] = xi_cat.cat.codes
        # Add to replace and remove operations
        self.columns_to_add[field_name] = df_cat
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def _create_one_hot(self, field_name):
        " Create a one hot encodig for 'field_name' "
        self._info(f"Converting to one-hot: field '{field_name}'")
        has_na = self.df[field_name].isna().sum() > 0
        df_one_hot = pd.get_dummies(self.df[field_name], dummy_na=has_na)
        self.rename_category_cols(df_one_hot, f"{field_name}_")
        # Add to transformations
        self.columns_to_add[field_name] = df_one_hot
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def convert_dates(self):
        ''' Convert all dates '''
        self._info(f"Converting to 'date/time' values: fields {self.dates}")
        count = 0
        for field in self.dates:
            self._add_datepart(field)
        self._info(f"Converting to 'date/time' values: End")

    def drop_zero_std(self):
        " Drop features that have standard deviation below a threshold "
        self._info(f"Dropping columns with low standard deviation: Start, std_threshold: {self.std_threshold}")
        to_remove = list()
        for c in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[c]):
                continue
            stdev = self.df[c].std()
            if stdev <= self.std_threshold:
                self._info(f"Dropping column '{c}', standard deviation {stdev} <= {self.std_threshold}, values head: {list(self.df[c].head())}")
                to_remove.append(c)
        self.df.drop(to_remove, axis=1, inplace=True)
        self._info(f"Dropping columns with low standard deviation: End")

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

    def nas(self):
        " Transform 'NA' columns (i.e. missing data) "
        self._info(f"Checking NA (missing values): Start")
        for field_name in self.df.columns:
            if field_name not in self.skip_nas:
                self.na(field_name)
        self._info(f"Checking NA (missing values): End")

    def na(self, field_name):
        " Process 'na' columns (i.e missing data) "
        # Add '*_na' column
        xi = self.df[field_name].copy()
        xi_isna = xi.isna().astype('int8')
        count_na = sum(xi_isna)
        if count_na == 0:
            return False
        df_na = pd.DataFrame()
        df_na[f"{field_name}_na"] = xi.isna().astype('int8')
        # Replace missing values by median
        # TODO: Add other strategies (e.g. mean).
        # TODO: Define on column by column basis
        replace_value = xi.median()
        xi[xi.isna()] = replace_value
        df_na[field_name] = xi
        self._info(f"Filling {count_na} NA values: field '{field_name}', value: '{replace_value}'")
        # Add operations
        self.columns_to_add[field_name] = df_na
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)
        return True

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

    def should_be_one_hot(self, field_name):
        " Should we convert to 'one hot' encoding? "
        if field_name in self.outputs:
            return False
        xi = self.df[field_name]
        xi_cat = xi.astype('category')
        count_cats = len(xi_cat.cat.categories)
        return count_cats <= self.one_hot_max_cardinality

    def transform(self):
        '''
        Perform data frame pre-processing steps:
            - Convert categorical data
            - Convert on-hot encoding
            - Convert dates into multiple columns
        '''
        self.convert_dates()
        self.create_categories()
        self.nas()
