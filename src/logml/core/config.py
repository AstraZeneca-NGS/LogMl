
import argparse
import copy
import logging
import sys
import yaml

from .files import MlFiles
from .registry import MlRegistry


DEFAULT_YAML = "config.yaml"

CONFIG_CONFIG = ''
CONFIG_ANALYSIS = 'analysis'
CONFIG_CROSS_VALIDATION = 'cross_validation'
CONFIG_DATASET = 'dataset'
CONFIG_DATASET_AUGMENT = 'dataset_augment'
CONFIG_DATASET_EXPLORE = 'dataset_explore'
CONFIG_DATASET_FEATURE_IMPORTANCE = "dataset_feature_importance"
CONFIG_DATASET_PREPROCESS = 'dataset_preprocess'
CONFIG_ENABLE = 'enable'
CONFIG_FUNCTIONS = 'functions'
CONFIG_HYPER_PARAMETER_OPTMIMIZATION = 'hyper_parameter_optimization'
CONFIG_LOGGER = 'logger'
CONFIG_MODEL = 'model'
CONFIG_MODEL_SEARCH = 'model_search'
CONFIG_MODEL_ANALYSIS = 'model_analysis'


def update_dict(d, u):
    """ Recursively update dictionary 'd' using items in 'u' """
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, collections.Mapping):
            d[k] = v
        elif isinstance(v, collections.Mapping):
            d[k] = update_dict(dv, v)
        else:
            d[k] = v
    return d


class Config(MlFiles):
    """
    Parameters object: Reads a YAML file and stores configuration parameters
    """
    def __init__(self, config_file=None, parameters=None, argv=None, log_level=None):
        super().__init__(CONFIG_CONFIG)
        self.argv = argv if argv is not None else sys.argv
        self.config_file = config_file
        self.parameters = parameters if parameters is not None else dict()
        self.is_debug = False
        self.exit_on_fatal_error = True
        if log_level:
            self.set_log_level(log_level)

    def __call__(self):
        self.parse_cmd_line()
        return self.read_config_yaml()

    def _config_sanity_check(self):
        """
        Check configuration parameters
        Returns True on success, False on failure
        """
        return True

    def copy(self, disable_all=False):
        """ Create a copy of the config, disable all sections if 'disable_all' is true """
        conf = copy.deepcopy(self)
        if disable_all:
            for sec in [CONFIG_DATASET, CONFIG_DATASET_EXPLORE, CONFIG_DATASET_FEATURE_IMPORTANCE, CONFIG_HYPER_PARAMETER_OPTMIMIZATION, CONFIG_MODEL_ANALYSIS, CONFIG_MODEL_SEARCH]:
                conf.set_enable(sec, enable=False)
        return conf

    def __getitem__(self, key):
        return self.get_parameters(key)

    def get_parameters(self, section):
        """ Get 'section' parameters """
        if section in self.parameters:
            return self.parameters[section]
        self._debug(f"Config has not parameters for '{section}'.")
        return dict()

    def get_parameters_functions(self, fname):
        """ Get user-defined functions parameters """
        return self.get_parameters_section(CONFIG_FUNCTIONS, fname, default_value=dict())

    def get_parameters_section(self, section, param_name, default_value=None):
        """ Get parameters 'param_name' from section 'section' """
        fdict = self.get_parameters(section)
        if not fdict:
            self._debug(f"Config has no '{section}' section.")
            return default_value
        if param_name in fdict:
            return fdict[param_name]
        self._debug(f"Config has no parameters for '{param_name}' in '{section}' section, returning default value '{default_value}'.")
        return default_value

    def invoke(self, name, tag, args=None):
        return MlRegistry().invoke(name, tag, args, self.get_parameters_functions(name))

    def parse_cmd_line(self):
        """ Parse command line options """
        if not self.argv:
            return True
        if self.argv[0].endswith('ipykernel_launcher.py'):
            # Launched form Jupyter notebook, skip command line args parsing
            return True
        self._debug(f"args: {self.argv}")
        parser = argparse.ArgumentParser()

        parser.add_argument('-c', '--config', help=f"Path to config (YAML) file. Default: '{DEFAULT_YAML}'", metavar='config.yaml', default=DEFAULT_YAML)
        parser.add_argument('-d', '--debug', help=f"Debug mode", action='store_true')
        parser.add_argument('-v', '--verbose', help=f"Verbose mode", action='store_true')

        # Parse command line
        args = parser.parse_args(self.argv[1:])
        if self.config_file is None:
            self.config_file = args.config
        self.is_verbose = args.verbose
        if self.is_verbose:
            self.set_log_level(logging.INFO)
        self.is_debug = args.debug
        if self.is_debug:
            self.set_log_level(logging.DEBUG)
        return True

    def read_config_yaml(self):
        """
        Reads a configuration YAML file and checks parameters
        Returns True on success, False on failure
        """
        self._info(f"Reading yaml file '{self.config_file}'")
        self.parameters = self._load_yaml(self.config_file)
        self._debug(f"params: {self.parameters}")
        self._set_from_config()
        return self._config_sanity_check()

    def set_enable(self, section, enable=True):
        if section in self.parameters:
            self.parameters[section]['enable'] = enable

    def update(self, params):
        """
        Create a new config and update parameters from 'params' dictionary
        recursively
        """
        config_new = copy.deepcopy(self)
        update_dict(config_new.parameters, params)
        return config_new

    def update_section(self, section, params):
        """
        Create a new config and update parameters from 'params' dictionary.
        E.g.: These parameters have been set by hyper parameter optimization process.
        Return new config object with updated values.
        """
        config_new = copy.deepcopy(self)
        parameters = config_new.parameters[section] if section is not None else config_new.parameters
        if params:
            for param_type in params.keys():
                if (param_type not in parameters) or (parameters[param_type] is None):
                    parameters[param_type] = dict()
                parameters[param_type].update(params[param_type])
                self._debug(f"Updating {param_type}: {params[param_type]}")
        return config_new

    def __str__(self):
        return f"config_file '{self.config_file}', parameters: {self.parameters}"
