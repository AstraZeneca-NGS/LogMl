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

    def __init__(self, df, config, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS)
        self.df = df
        self.log = list()
        self.logp1 = list()
        self.maxabs = list()
        self.minmax: list()
        self.quantile = list()
        self.skip = list()
        self.standard = list()
        if set_config:
            self._set_from_config()
        self._init_methods()
        self._set_default_method()

    def _init_methods(self):
        self.normalize_methods['log'] = NormalizationMethod(method=self._normalize_log, fields=self.log)
        self.normalize_methods['log1'] = NormalizationMethod(method=self._normalize_log1, fields=self.log1)
        self.normalize_methods['maxabs'] = NormalizationMethod(method=self._normalize_maxabs, fields=self.maxabs)
        self.normalize_methods['minmax'] = NormalizationMethod(method=self._normalize_minmax, fields=self.minmax)
        self.normalize_methods['quantile'] = NormalizationMethod(method=self._normalize_quantile, fields=self.quantile)
        self.normalize_methods['standard'] = NormalizationMethod(method=self._normalize_standard, fields=self.standard)

    def __call__(self):
        """
        Preprocess dataframe columns
        Returns a new (transformed) dataset
        """
        self._debug("Preprocessing dataframe: Start")
        self.sanity_check()
        self.normalize()
        self._debug("Preprocessing dataframe: End")
        return self.df

    def find_normalize_method(self, col_name):
        ''' Find a normalizetion method for this column '''
        for nm_name, nm in self.normalize_methods.items():
            if c in nm.fields:
                return nm.method
        return self.normalize_method_default

    def normalize(self):
        ''' Normalize variables '''
        self._debug("Normalizing dataset (dataframe): Start")
        fields_to_normalize = list(self.df.columns)
        fields_normalized = set()
        for c in fields_to_normalize:
            nm = self.find_normalize_method(c)
            if nm is None:
                self._debug()
            else:
                # Normalize and replace column
                xi = nm(self.df[c])
                self.df[c] = xi
        self._debug("Normalizing dataset (dataframe): End")

    def _normalize_log(self, xi):
        ''' Normalize using 'log' method '''
        
        return None

    def _normalize_log1(self, xi):
        ''' Normalize using 'log1' method '''
        return None

    def _normalize_maxabs(self, xi):
        ''' Normalize using 'maxabs' method '''
        return None

    def _normalize_minmax(self, xi):
        ''' Normalize using 'minmax' method '''
        return None

    def _normalize_quantile(self, xi):
        ''' Normalize using 'quantile' method '''
        return None

    def _normalize_standard(self, xi):
        ''' Normalize using 'standard' method '''
        return None


    def _set_normalize_method_default(self):
        ''' Set default normalization method '''
        meth = list()
        self.normalize_method_default = None
        if self.log is True:
            self.normalize_method_default = self.normalize_log
            meth.append('log')
        if self.logp1 is True:
            self.normalize_method_default = self.normalize_logp1
            meth.append('logp1')
        if self.maxabs is True:
            self.normalize_method_default = self.normalize_maxabs
            meth.append('maxabs')
        if self.minma is True:
            self.normalize_method_default = self.normalize_minma
            meth.append('minma')
        if self.quantile is True:
            self.normalize_method_default = self.normalize_quantile
            meth.append('quantile')
        if self.skip is True:
            self.normalize_method_default = self.normalize_skip
            meth.append('skip')
        if self.standard is True:
            self.normalize_method_default = self.normalize_standard
            meth.append('standard')
        if len(meth) > 1:
            self._fatal_error(f"Dataset (DataFrame) preprocessing: More than one default method ({meth}). Only one should be set to 'True'")
        self._debug(f"Default normalization method set to {meth}")
