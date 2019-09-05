#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from .dataset import Dataset, InOut
from .df_transform import DfTransform

from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype


class DatasetDf(Dataset):
    '''
    A dataset based on a Pandas DataFrame
    (i.e. Dataset.dataset must be a DataFrame)
    '''
    def __init__(self, config, set_config=True):
        super().__init__(config, set_config=False)
        self.count_na = dict()  # Count missing values for each field
        self.categories = dict()  # Convert these fields to categorical
        self.dataset_ori = None
        self.dates = list()  # Convert these fields to dates and expand to multiple columns
        self.one_hot = list()  # Convert these fields to 'one hot encoding'
        self.one_hot_max_cardinality = None  # Convert to one hot encoding, all fields with cardinality <= 'one_hot_max_cardinality'
        self._set_from_config()
        self.missing_values = dict()  # Value used to replace missing values
        if set_config:
            self._set_from_config()

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

    def default_transform(self):
        " Default implementation for '@dataset_transform' "
        self._debug(f"Using default dataset transform for DataFrame")
        self.dataset_ori = self.dataset  # Keep a copy of the original dataset
        dft = DfTransform(self.dataset, self.config)
        self.dataset = dft()
        self._debug(f"End: Columns after transform are {list(self.dataset.columns)}")
        return True

    def _load_from_csv(self):
        ''' Load dataframe from CSV '''
        csv_file = self.get_file_name(ext='csv')
        self._debug(f"Loading csv file '{csv_file}'")
        self.dataset = pd.read_csv(csv_file, low_memory=False, parse_dates=self.dates)
        return len(self.dataset) > 0
