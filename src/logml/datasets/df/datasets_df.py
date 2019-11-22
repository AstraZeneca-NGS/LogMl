#!/usr/bin/env python

import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from ..datasets import Datasets, InOut
from .df_preprocess import DfPreprocess
from .df_transform import DfTransform

from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype


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

    def _columns_na(self, df):
        """ Get a columns that should b used for a new dataframe having only columns indicating missing data """
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
        return False

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

    def _get_dataframe_na(self, df, cols_na):
        """ Get a new dataframe having only columns indicating missing data """
        if df is None:
            return None
        # Remove
        cols_to_remove = [c for c in df.columns if c not in cols_na]
        df_na = df.drop(columns=cols_to_remove)
        # Change cathegorical, only keep 'na' data
        for c in df.columns:
            if c in cols_na and c in self.dataset_transform.category_column:
                cat_col = self.dataset_transform.category_column[c]
                df_na[c] = cat_col.isna().astype('int')
        return df_na

    def get_datasets_na(self):
        if self.dataset_transform is None:
            self._error("Cannot create 'missing' dataset")
            return None
        # Create a new datasets, with same parameters
        self._debug(f"Creating 'missing' dataset: Start")
        dsna = copy.copy(self)
        dsna.reset()
        # Which columns should we keep?
        cols_na = self._columns_na(self.dataset)
        self._debug(f"Creating 'missing' dataset: Columns {cols_na}")
        if len(cols_na) <= 0:
            self._debug(f"Creating 'missing' dataset: No columns selected, returning None")
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
        return self.dataset.iloc[key]

    def _load_from_csv(self):
        ''' Load dataframe from CSV '''
        csv_file = self.get_file_name(ext='csv')
        self._debug(f"Loading csv file '{csv_file}'")
        self.dataset = pd.read_csv(csv_file, low_memory=False, parse_dates=self.dates)
        return len(self.dataset) > 0
