#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from collections import namedtuple
from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog


NormalizationMethod = namedtuple('NormalizationMethod', ['method', 'fields'])


class DfPreprocess(MlLog):
    '''
    DataFrame preprocessing: Normalize / re-scale inputs
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS)
        self.df = df
        self.model_type = model_type
        self.normalize = dict()
        self.outputs = set(outputs)
        if set_config:
            self._set_from_config()
        self._init_methods()
        self._set_normalize_method_default()

    def _init_methods(self):
        self.normalize_methods = dict()
        self.normalize_methods['log'] = NormalizationMethod(method=self._normalize_log, fields=self.get_normalize_list('log'))
        self.normalize_methods['log1p'] = NormalizationMethod(method=self._normalize_log1p, fields=self.get_normalize_list('log1p'))
        self.normalize_methods['maxabs'] = NormalizationMethod(method=self._normalize_maxabs, fields=self.get_normalize_list('maxabs'))
        self.normalize_methods['minmax'] = NormalizationMethod(method=self._normalize_minmax, fields=self.get_normalize_list('minmax'))
        self.normalize_methods['minmax_neg'] = NormalizationMethod(method=self._normalize_minmax_neg, fields=self.get_normalize_list('minmax_neg'))
        self.normalize_methods['quantile'] = NormalizationMethod(method=self._normalize_quantile, fields=self.get_normalize_list('quantile'))
        self.normalize_methods['standard'] = NormalizationMethod(method=self._normalize_standard, fields=self.get_normalize_list('standard'))
        self.normalize_skip = self.get_normalize_list('skip')

    def __call__(self):
        """
        Preprocess dataframe columns
        Returns a new (transformed) dataset
        """
        if not self.enable:
            self._debug(f"Preprocessing dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', enable='{self.enable}'")
            return self.df
        self._debug("Preprocessing dataframe: Start")
        self._normalize()
        self._debug("Preprocessing dataframe: End")
        return self.df

    def find_normalize_method(self, col_name):
        ''' Find a normalizetion method for this column '''
        if col_name in self.normalize_skip:
            self._debug(f"Normalize variable '{col_name}' in skip list: skipping")
            return
        for nm_name, nm in self.normalize_methods.items():
            if nm.fields is True:
                continue
            if col_name in nm.fields:
                return nm.method
        return self.normalize_method_default

    def get_normalize_list(self, name):
        li = self.normalize.get(name, list())
        return list() if li is None else li

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
            nm = self.find_normalize_method(c)
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

    def _normalize_log(self, xi):
        ''' Normalize using 'log' method '''
        return np.log(xi)

    def _normalize_log1p(self, xi):
        ''' Normalize using 'log1p' method '''
        return np.log1p(xi)

    def _normalize_maxabs(self, xi):
        ''' Normalize using 'maxabs' method '''
        return xi / np.max(np.abs(xi))

    def _normalize_minmax(self, xi):
        ''' Normalize using 'minmax' method. Convert to interval [0, 1] '''
        mi = np.min(xi)
        ma = np.max(xi)
        if (ma - mi) <= 0.0:
            return xi
        return (xi - mi) / (ma - mi)

    def _normalize_minmax_neg(self, xi):
        ''' Normalize using 'minmax' method. Convert to interval [-1, 1] '''
        mi = np.min(xi)
        ma = np.max(xi)
        if (ma - mi) <= 0.0:
            return xi
        return 2.0 * (xi - mi) / (ma - mi) - 1.0

    def _normalize_quantile(self, xi):
        ''' Normalize using 'quantile' method '''
        return None

    def _normalize_standard(self, xi):
        ''' Normalize using 'standard' method '''
        me = np.nanmean(xi)
        std = np.nanstd(xi)
        if std <= 0.0:
            return (xi - me)
        return (xi - me) / std

    def _set_normalize_method_default(self):
        ''' Set default normalization method '''
        meth = list()
        self.normalize_method_default = None
        if self.get_normalize_list('log') is True:
            self.normalize_method_default = self._normalize_log
            meth.append('log')
        if self.get_normalize_list('log1p') is True:
            self.normalize_method_default = self._normalize_log1p
            meth.append('log1p')
        if self.get_normalize_list('maxabs') is True:
            self.normalize_method_default = self._normalize_maxabs
            meth.append('maxabs')
        if self.get_normalize_list('minmax') is True:
            self.normalize_method_default = self._normalize_minmax
            meth.append('minmax')
        if self.get_normalize_list('minmax_neg') is True:
            self.normalize_method_default = self._normalize_minmax_neg
            meth.append('minmax_neg')
        if self.get_normalize_list('quantile') is True:
            self.normalize_method_default = self._normalize_quantile
            meth.append('quantile')
        if self.get_normalize_list('skip') is True:
            self.normalize_method_default = self._normalize_skip
            meth.append('skip')
        if self.get_normalize_list('standard') is True:
            self.normalize_method_default = self._normalize_standard
            meth.append('standard')
        if len(meth) > 1:
            self._fatal_error(f"Dataset (DataFrame) preprocessing: More than one default method ({meth}). Only one should be set to 'True'")
        self._debug(f"Default normalization method set to {meth}")
