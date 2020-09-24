
import hyperopt
import hyperopt.hp
import numpy as np

from ..core.config import CONFIG_HYPER_PARAMETER_OPTMIMIZATION, CONFIG_FUNCTIONS, CONFIG_MODEL
from ..core.log import MlLog


HYPER_PARAM_CREATE_DATASET = ['augment', 'dataset_create', 'dataset_load', 'dataset_split', 'preprocess']
HYPER_PARAM_TYPES = list(HYPER_PARAM_CREATE_DATASET)
HYPER_PARAM_TYPES.extend(['model_train', 'model_create'])


class HyperOpt(MlLog):
    """ Hyper parameter optimization class """
    def __init__(self, logml):
        super().__init__(logml.config, CONFIG_HYPER_PARAMETER_OPTMIMIZATION)
        self.logml = logml
        self.algorithm = 'tpe'
        self.best = None
        self.best_fit = None
        self.best_params = None
        self.iteration = 0
        self.max_evals = 100
        self.random_seed = None
        self.show_progressbar = False
        self.space = dict()
        self._set_from_config()

    def __call__(self):
        if not self.enable:
            self._info(f"Hyper-parameter search disable (model 'enable={self.enable}'), skipping")
            return True
        return self.hyper_parameter_search()

    def _config_sanity_check(self):
        """
        Check parameters from config.
        Return True on success, False if there are errors
        """
        model_enable = self.config.get_parameters(CONFIG_MODEL).get('enable')
        if self.enable and not model_enable:
            self._fatal_error(f"Config file '{self.config.config_file}', section {CONFIG_HYPER_PARAMETER_OPTMIMIZATION} inconsistency: Hyper-parameter search is enabled, but model is disabled (section {CONFIG_MODEL}, enable:{model_enable})")
        return True

    def create_objective_function(self):
        """
        Create an objective function for hyper-parameter tunning
        Note, we just crete a closure and return a wrapper
        """
        def _objective_function_wrapper(params):
            return self.objective_function(params)

        self._debug(f"Create objective function")
        return _objective_function_wrapper

    def create_search_space(self):
        """ Create search space for all hyper parameters """
        self._debug(f"Create search space")
        space = dict()
        for param_type in HYPER_PARAM_TYPES:
            if param_type in self.space:
                space.update(self._create_search_space(self.space[param_type], param_type))
        return space

    def _create_search_space(self, space_def, param_type):
        """ Create a search space, given a set of parameters """
        space = dict()
        if space_def is None:
            return space
        for param_label, param_def in space_def.items():
            self._debug(f"param_label:{param_label}, param_def:{param_def}")
            space[f"{param_type}.{param_label}"] = self._search_space(param_label, param_def)
        return space

    def get_algorithm(self):
        """ Get algorithm type from YAML definition"""
        algo = self.algorithm
        if algo is None:
            raise ValueError(f"Missing 'algorithm' entry in 'hyper_parameter_optimization' section (YAML file)")
        if algo == 'tpe':
            return hyperopt.tpe.suggest
        if algo == 'random':
            return hyperopt.random.suggest
        raise ValueError(f"Unknown  'algorithm' type '{algo}' in 'hyper_parameter_optimization' section (YAML file)")

    def hyper_parameter_search(self):
        """ Perform hyper parameter search """
        self._debug(f"Start")
        self.trials = hyperopt.Trials()
        self.objective = self.create_objective_function()
        # Create a search space
        self.search_space = self.create_search_space()
        # Get max_evals from YAML file paramters
        self.max_evals = self.parameters['max_evals']
        # Select algorithm
        self.algorithm = self.get_algorithm()
        # Should we create a new dataset on each iteration? True if we have any hyper-parameter from HYPER_PARAM_CREATE_DATASET
        self.is_create_dataset = any([self.space[param_type] for param_type in HYPER_PARAM_CREATE_DATASET if param_type in self.space])
        # Start search
        self._info(f"Search: Create dataset={self.is_create_dataset}")
        if self.random_seed is not None:
            self._debug(f"Using random seed {self.random_seed}")
        rand_state = np.random.RandomState(seed=self.random_seed)
        self.best = hyperopt.fmin(fn=self.objective,
                                  space=self.search_space,
                                  algo=self.algorithm,
                                  max_evals=self.max_evals,
                                  trials=self.trials,
                                  rstate=rand_state,
                                  show_progressbar=self.show_progressbar)
        self._info(f"Hyper parameter search best fit:{self.best}, best parameters: {self.best_params}")
        self.save_results()
        self._debug(f"End")
        return True

    def _new_dataset(self, config):
        """
        When we are using hyper-paarmeters in dataset 'create', 'augment' or 'postprocessing', we
        must create datasets from scratch each time.
        We build a copy of the dataset object but set it so that:
          - It doesn't load from pickle
          - It doesn't save to pickle
          - Reset fields (dataset / train / test / validate)
        Note: We do a 'copy' in order to preserve parameters and class type
        """
        if not self.is_create_dataset:
            return None
        self._debug(f"Creating new dataset")
        datasets = self.logml.datasets.clone()
        datasets.config = config
        datasets.do_not_load_pickle = True
        datasets.do_not_save = True
        datasets.reset(soft=True)
        datasets()
        return datasets

    def objective_function(self, params):
        """
        Objective function invoked in hyper-parameter tunning
        It invokes training & test, then returns the test result metric to minimize
        """
        self.iteration += 1
        # Create a new config with updated parameters
        params_ml = self._space2params(params)
        self._debug(f"Iteration: {self.iteration}, parameters={params_ml}")
        config_new = self.config.update_section(CONFIG_FUNCTIONS, params_ml)
        # Create new dataset
        dataset = self._new_dataset(config_new)
        # Train and evaluate model
        ret_train = self.logml.model_train(config_new, dataset)     # Note: We train a single model (we don't use the scatter & gather method 'model_train_scatter()')
        self._debug(f"Model train returned: {ret_train}")
        ret_val = self.logml.get_model_eval_validate()
        if ret_val is None:
            self._warning(f"Model test evaluation is 'None'. Either the `@model_evaluate` function returned 'None' or it was not executed")
            return np.inf
        self._debug(f"Model validation returned: {ret_val}")
        # Update best fit
        if self.best_fit is None or ret_val < self.best_fit:
            self.best_fit = ret_val
            self.best_params = params_ml
        self._info(f"Hyper parameter optimization:\titeration: {self.iteration}\tfit: {ret_val}\tparameters: {params_ml}\tbest fit: {self.best_fit}\tbest parameters: {self.best_params}")
        return ret_val

    def save_results(self):
        """ Save hyper parameter search results to picle file """
        mltrain = self.logml.model
        file_name = mltrain.get_file('hyper_param_search')
        self._debug(f"Save hyper-parameter search results: Saving to pickle file '{file_name}'")
        results = {'best': self.best, 'trials': self.trials, 'max_evals': self.trials}
        self.logml._save_pickle(file_name, 'hyper_param_search', results)
        return True

    def _search_space(self, param_label, param_def):
        """ Create a search space from the definition """
        self._debug(f"param_label={param_label}, param_def={param_def}")
        distribution = param_def[0]
        args = list()
        args.append(param_label)
        args.extend(param_def[1:])
        to_eval = f"hyperopt.hp.{distribution}(*{args})"
        self._debug(f"eval: {to_eval}")
        sp = eval(to_eval)
        self._debug(f"eval returned: {sp}")
        return sp

    def _space2params(self, params):
        """ Repack parameters for LogMl.model_train """
        ps = dict()
        for key in params:
            param_type, param_name = key.split('.', 2)
            if param_type not in ps:
                ps[param_type] = dict()
            ps[param_type][param_name] = params[key]
        return ps
