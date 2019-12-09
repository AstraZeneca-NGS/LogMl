#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from collections import namedtuple
from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog


# Method and fields to apply to
MethodAndFields = namedtuple('MethodAndFields', ['method', 'fields'])

METHOD_SKIP_NAME = 'skip'

class MethodsFields(MlLog):
    ''' A mapping from methods to fields (e.g. normalization methods applied to fields)'''

    def __init__(self, config, section, subsection, method_names):
        super().__init__(config, section)
        self.section = section
        self.subsection = subsection
        self.method_names = method_names
        self.__dict__[self.subsection] = dict() # List of fields indexed by method (this is populated from config file)

    def find_method(self, name):
        '''
        Find a method for input variable 'name'
        Returns None if field 'name' is in 'skip' list
        '''
        if self.is_skip(name):
            self._debug(f"Method {self.subsection}, variable '{name}' in skip list")
            return None
        for mn_name, mns in self._methods.items():
            if mns.fields is True:
                continue
            if name in mns.fields:
                return mns.method
        return self.method_default

    def get_fields(self, name):
        """ Get fields for method 'name' """
        return self.__dict__[self.subsection].get(name, list())

    def get_method(self, name):
        """ Get the (callable) method by name. The method must be named '_{self.section}_{name}' """
        return getattr(self, f"_{self.subsection}_{name}")

    def _init_methods(self):
        """ Initialize methods and field names  mapping """
        self._methods = dict()
        for n in self.method_names:
            self._methods[n] = MethodAndFields(method=self.get_method(n), fields=self.get_fields(n))

    def _initialize(self):
        self._init_methods()
        self._set_default_method()

    def is_skip(self, name):
        ''' Is 'name' in the list of fields to skip? '''
        return name in self.get_fields(METHOD_SKIP_NAME)

    def _set_default_method(self):
        ''' Set default method (from configuration) '''
        default_method = list()
        self.method_default = None
        for n in self.method_names:
            if self.get_fields(n) is True:
                self.method_default = self.get_method(n)
                default_method.append('log')
        if len(default_method) > 1:
            self._fatal_error(f"Dataset (DataFrame) preprocessing: More than one default method ({default_method}). Only one should be set to 'True'")
        self._debug(f"Default method set to {default_method}")
