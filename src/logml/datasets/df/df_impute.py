#!/usr/bin/env python

import numpy as np
import pandas as pd
import re
from collections import namedtuple

from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog
from .methods_fields import MethodsFields

# Method names
IMPUTATION_METHODS = ['mean', 'median', 'most_frequent', 'one', 'skip', 'zero']


class DfImpute(MethodsFields):
    '''
    DataFrame imputation of missing data
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super()._init__(config, CONFIG_DATASET_PREPROCESS, 'normalize', IMPUTATIAON_METHODS)
        self.df = df
        self.model_type = model_type
        self.outputs = set(outputs)
        if set_config:
            self._set_from_config()
        self._initialize()

    def __call__(self):
        """
        Normalize inputs
        Returns a new (transformed) dataset
        """
        if not self.enable:
            self._debug(f"Normalizing dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', sub-section 'normalize', enable='{self.enable}'")
            return self.df
        self._debug("Normalizing dataframe: Start")
        self._normalize()
        self._debug("Normalizing dataframe: End")
        return self.df

    def is_numertic(self, col_name):
        return np.issubdtype(self.df[col_name].dtype, np.number)

    def is_range_01(self, col_name):
        '''
        Is this variable in [0,1] range?
        We say the varaible is in [0.0, 1.0] range if the min is
        in [0.0 , 0.1] and the max is in [0.9, 1.0]
        '''
        xi = self.df[col_name]
        self._debug(f"DTYPE: {col_name}\t{xi.dtype}")
        xi_min = xi.unique().min()
        xi_max = xi.unique().max()
        return 0.0 <= xi_min and xi_min <= 0.1 and 0.9 <= xi_max and xi_max <= 1.0

    def _normalize(self):
        ''' Normalize variables '''
        self._debug("Normalizing dataset (dataframe): Start")
        fields_to_normalize = list(self.df.columns)
        fields_normalized = set()
        is_classification = (self.model_type == 'classification')
        for c in fields_to_normalize:
            if not self.is_numertic(c):
                self._debug(f"Normalize variable '{c}' is not numeric, skipping")
                continue
            if self.is_range_01(c):
                self._debug(f"Normalize variable '{c}' is in [0, 1], skipping")
                continue
            if is_classification and c in self.outputs:
                self._debug(f"Normalize variable '{c}' is an output varaible for a classification model, skipping")
                continue
            nm = self.find_method(c)
            xi = self.df[c]
            self._debug(f"Before normalization '{c}': mean={np.nanmean(xi)}, std={np.nanstd(xi)}")
            if nm is None:
                self._debug(f"Normalize field '{c}': No method defined, skipping")
            else:
                self._info(f"Normalizing field '{c}', method {nm.__name__}")
                # Normalize and replace column
                xi = nm(xi)
                self._debug(f"After normalization '{c}': mean={np.nanmean(xi)}, std={np.nanstd(xi)}")
                self.df[c] = xi
        self._debug("Normalizing dataset (dataframe): End")

    def _impute_mean(self, xi):
        ''' Impute using 'mean' method '''
        return xi[~np.isnan(xi)].mean()

    def _impute_median(self, xi):
        ''' Impute using 'median' method '''
        return np.median(xi[~np.isnan(xi)])

    def _impute_most_frequent(self, xi):
        ''' Impute using 'most_frequent' method '''
        # Select the most frequent values
        count = dict()
        for n in x[~np.isnan(x)]:
            count[n] = 1 + count.get(n,0)
        count_max = max(count.values())
        # If there is more than one 'most frequent value', use the median of them
        most_freqs = [k for k, c in count.items() if c == count_max]
        return np.median(np.array(most_freqs))

    def _impute_one(self, xi):
        ''' Impute using 'one' method '''
        return 1

    def _impute_zero(self, xi):
        ''' Impute using 'zero' method '''
        return 0
