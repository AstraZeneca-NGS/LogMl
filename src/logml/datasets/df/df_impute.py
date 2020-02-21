#!/usr/bin/env python

import numpy as np
import pandas as pd
import re
from collections import namedtuple

from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog
from .methods_fields import MethodsFields

# Method names
IMPUTATION_METHODS = ['mean', 'median', 'minus_one', 'most_frequent', 'one', 'skip', 'zero']


class DfImpute(MethodsFields):
    '''
    DataFrame imputation of missing data
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS, 'impute', IMPUTATION_METHODS, df.columns, outputs)
        self.df = df
        self.model_type = model_type
        self.outputs = set(outputs)
        if set_config:
            self._set_from_config()
        self._initialize()

    def __call__(self):
        """
        Impute inputs
        Returns a new (imputed) dataset
        """
        if not self.enable:
            self._debug(f"Imputing dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', sub-section 'normalize', enable='{self.enable}'")
            return self.df
        self._debug("Imputing dataframe: Start")
        self._impute()
        self._debug("Imputing dataframe: End")
        return self.df

    def is_numeric(self, col_name):
        return np.issubdtype(self.df[col_name].dtype, np.number)

    def _impute(self):
        ''' impute variables '''
        self._debug("Imputing dataset (dataframe): Start")
        fields_to_impute = list(self.df.columns)
        for c in fields_to_impute:
            if not self.is_numeric(c):
                self._debug(f"Impute: Variable '{c}' is not numeric, skipping")
                continue
            if self.is_skip(c):
                self._debug(f"Impute: Variable '{c}' defined as 'skip', skipping")
                continue
            nm = self.find_method(c)
            xi = self.df[c].copy()
            count_na = sum(xi.isna().astype('int8'))
            if nm is None:
                self._debug(f"Impute: No method defined field '{c}', skipping")
            elif count_na > 0:
                replace_value = nm(self.df[c])
                if replace_value is not None:
                    xi[xi.isna()] = replace_value
                    self._info(f"Impute: Field '{c}' has {count_na} NA values, imputing with value '{replace_value}', method '{nm.__name__}'")
                    self.df[c] = xi
            else:
                self._debug(f"Impute: Field '{c}' has no 'NA' values, skipping")
        self._debug("Imputing dataset (dataframe): End")

    def _impute_mean(self, xi):
        ''' Impute using 'mean' method '''
        return xi[~np.isnan(xi)].mean()

    def _impute_median(self, xi):
        ''' Impute using 'median' method '''
        return np.median(xi[~np.isnan(xi)])

    def _impute_minus_one(self, xi):
        ''' Impute using 'minus_one' method '''
        return -1

    def _impute_most_frequent(self, xi):
        ''' Impute using 'most_frequent' method '''
        # Select the most frequent values
        count = dict()
        for n in xi[~np.isnan(xi)]:
            count[n] = 1 + count.get(n, 0)
        count_max = max(count.values())
        # If there is more than one 'most frequent value', use the median of them
        most_freqs = [k for k, c in count.items() if c == count_max]
        return np.median(np.array(most_freqs))

    def _impute_one(self, xi):
        ''' Impute using 'one' method '''
        return 1

    def _impute_skip(self, xi):
        return None

    def _impute_zero(self, xi):
        ''' Impute using 'zero' method '''
        return 0
