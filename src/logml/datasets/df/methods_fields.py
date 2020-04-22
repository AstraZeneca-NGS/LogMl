#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from collections import namedtuple
from ...core.config import CONFIG_DATASET_PREPROCESS, CONFIG_ENABLE
from ...core.log import MlLog


# Method and fields to apply to
MethodAndFields = namedtuple('MethodAndFields', ['method', 'fields'])

# Name, fields and parameters
NameFieldsParams = namedtuple('FieldsParams', ['name', 'fields', 'params'])

METHOD_SKIP_NAME = 'skip'


class MatchFields(MlLog):
    def __init__(self, config, section, subsection, field_names, outputs):
        super().__init__(config, section)
        self.field_names = field_names
        self.outputs = set(outputs)
        self.section = section
        self.subsection = subsection
        # List of fields / regex, indexed by method, populated from config file
        params = config.get_parameters(section).get(subsection, dict())
        self.__dict__[self.subsection] = params
        # Set 'enable' and remove from parameters
        self.enable = params.get(CONFIG_ENABLE, True)
        if CONFIG_ENABLE in params:
            del params[CONFIG_ENABLE]

    def match_fields(self, regex):
        """
        Find all fields matching a regex
        Note: We use 'fullmatch' instead of 'match' because the regex 'x1' should
        only match the field name 'x1' and not 'x10'
        """
        matched = [fname for fname in self.field_names if re.fullmatch(regex, fname) is not None]
        self._debug(f"Regex '{regex}' matched field names: {matched}")
        if not matched:
            self._fatal_error(f"Regex '{regex}' (config section '{self.section}', subsection '{self.subsection}') did not match any of the field names: {list(self.field_names)}")
        return matched

    def match_input_fields(self, regex):
        """ Match a regex only against input fields """
        matched = [f for f in self.match_fields(regex) if f not in self.outputs]
        if not matched:
            self._fatal_error(f"Regex '{regex}' (config section '{self.section}', subsection '{self.subsection}') did not match any input field names: {[f for f in self.field_names if f not in self.outputs]}")
        return matched


class MethodsFields(MatchFields):
    """ A mapping from methods to fields (e.g. normalization methods applied to fields)"""

    def __init__(self, config, section, subsection, method_names, field_names, outputs):
        super().__init__(config, section, subsection, field_names, outputs)
        self.method_names = method_names
        self.fields_by_method = dict()

    def find_method(self, name):
        """
        Find a method for input variable 'name'
        Returns None if field 'name' is in 'skip' list
        """
        if self.is_skip(name):
            self._debug(f"Method {self.subsection}, variable '{name}' in skip list")
            return None
        for mn_name, mns in self._methods.items():
            if mns.fields is True:
                continue
            if name in mns.fields:
                return mns.method
        return self.method_default

    def get_fields(self, method_name):
        """ Get fields for method 'method_name' """
        return self.fields_by_method.get(method_name, list())

    def get_method(self, name):
        """ Get the (callable) method by name. The method must be named '_{self.section}_{name}' """
        return getattr(self, f"_{self.subsection}_{name}")

    def _init_methods(self):
        """ Initialize methods and field names  mapping """
        self._methods = dict()
        for n in self.method_names:
            self._methods[n] = MethodAndFields(method=self.get_method(n), fields=self.get_fields(n))

    def _initialize(self):
        self._populate_fields_by_method()
        self._init_methods()
        self._set_default_method()

    def is_skip(self, name):
        """
        Is 'name' in the list of fields to skip?
        Return: True if skip is a list and 'name' is in the list, False otherwise (even if skip=True, i.e. defined as default method)
        """
        skip = self.get_fields(METHOD_SKIP_NAME)
        return False if skip is True else name in skip

    def _populate_fields_by_method(self):
        """ Set values in 'self.fields_by_method' from the data in config.
        The original values can include field names, 'True' or a regex, we need to match the fields
        """
        self.fields_by_method = dict()
        fields_by_method = self.__dict__[self.subsection]
        for method_name, fields in fields_by_method.items():
            if fields is True:
                pass    # 'True' means use as default method
            else:
                # Resolve each item in 'fields', then flatten list
                fields = [f for item in fields for f in self.match_fields(item)]
            self.fields_by_method[method_name] = fields
            self._debug(f"Method '{method_name}' for fields '{fields}'")

    def _set_default_method(self):
        """ Set default method (from configuration) """
        default_method = list()
        self.method_default = None
        for n in self.method_names:
            if self.get_fields(n) is True:
                self.method_default = self.get_method(n)
                default_method.append(n)
        if len(default_method) > 1:
            self._fatal_error(f"Dataset (DataFrame) preprocessing: More than one default method ({default_method}). Only one should be set to 'True'")
        self._debug(f"Default method set to {default_method}")


class FieldsParams(MatchFields):
    """A class to parse a list of configurations options
    Examples:
    ```
        pca:
            name_pca_1:
              num: 2
              fields: ['x.*']
            name_pca_2:
              num: 7
              fields: ['z.*']
        log_ratio:
            name_log_ratio:
              base: 'e'
              fields: ['z.*']
    ```
    """

    def __init__(self, df, config, section, subsection, field_names, outputs, params=None, madatory_params=None):
        super().__init__(config, section, subsection, field_names, outputs)
        self.df = df
        self.params = params
        self.madatory_params = madatory_params
        self.name_fields_params = dict()
        # self._set_from_config()
        self._initialize()

    def array_to_df(self, x, cols):
        return pd.DataFrame(x, columns=cols, index=self.df.index)

    def calc(self, namefieldparams, x):
        """Method to calculate
        Arguments:
            namefieldparams: A NameFieldsParams object
            x: Input variables
        Returns:
            A Pnadas DataFrame with new values (or None on failure)
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def __call__(self):
        """ Perform calculation and return a dataFrame with 'original + augmented' data """
        self._info(f"Augment dataset: Start, name: {self.subsection}")
        dfs = list()
        for nf in self.name_fields_params.values():
            ret = self.calc(nf, self.df[nf.fields])
            if ret is not None:
                dfs.append(ret)
                self._debug(f"DataFrame calculated has shape {ret.shape}")
        df = None
        if len(dfs) > 0:
            df = pd.concat(dfs, axis=1)
            self._debug(f"DataFrame joined shape {df.shape}")
        else:
            self._debug(f"No results")
        self._info(f"Augment dataset: End, name: {self.subsection}")
        return df

    def get_col_names_num(self, namefieldparams, x):
        " Create a list of column names as 'name_i' "
        return [f"{namefieldparams.name}_{i}" for i in range(x.shape[1])]

    def _get_params(self, name, vals):
        """
        Get a dictionary of parameters
        @Return A dictionary of paramters on success, None on failure to satisfy
        mandatory parameters
        """
        params = dict()
        if self.params is None:
            return params
        for param_name in self.params:
            value = vals.get(param_name)
            if value is None and self.madatory_params is not None and param_name in self.madatory_params:
                self._warning(f"Parameter '{param_name}' is not set, ignoring entry '{name}' in sub section '{self.subsection}'")
                return None
            else:
                params[param_name] = value
        return params

    def _initialize(self):
        for name, vals in self.__dict__[self.subsection].items():
            params = self._get_params(name, vals)
            if params is None:
                continue
            regex_list = vals.get('fields')
            if regex_list is None:
                self._warning(f"Fields not set, ignoring entry '{name}' in sub section '{self.subsection}'")
                continue
            fields = [f for regex in regex_list for f in self.match_input_fields(regex)]
            self._debug(f"FieldsParams: Name='{name}', params={params}, regex list={regex_list}, matched input fields={fields}")
            self.name_fields_params[name] = NameFieldsParams(name, fields, params)
