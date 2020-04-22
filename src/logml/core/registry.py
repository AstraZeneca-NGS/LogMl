import functools
import time

from .log import MlLog


# Registry names
DATASET_AUGMENT = 'dataset_augment'
DATASET_CREATE = 'dataset_create'
DATASET_INOUT = 'dataset_inout'
DATASET_LOAD = 'dataset_load'
DATASET_PREPROCESS = 'dataset_preprocess'
DATASET_SAVE = 'dataset_save'
DATASET_SPLIT = 'dataset_split'
MODEL_CREATE = 'model_create'
MODEL_EVALUATE = 'model_evaluate'
MODEL_PREDICT = 'model_predict'
MODEL_SAVE = 'model_save'
MODEL_TRAIN = 'model_train'

ENABLE = 'enable'

REGISTRATION_KEYS = [DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT
                     , DATASET_LOAD, DATASET_PREPROCESS, DATASET_SAVE
                     , DATASET_SPLIT
                     , MODEL_CREATE, MODEL_EVALUATE, MODEL_PREDICT, MODEL_SAVE
                     , MODEL_TRAIN]


def singleton(cls):
    """ Singleton decorator """
    @functools.wraps(cls)
    def wrapper():
        if not wrapper.instance:
            wrapper.instance = cls()
        return wrapper.instance
    wrapper.instance = None
    return wrapper


@singleton
class MlRegistry(MlLog):
    """ Singleton class used to register functions """
    def __init__(self, model_name=None, parameters=None):
        super().__init__()
        self._functions = dict()     # Functions registered for each stage

    def get_function(self, function_name):
        return self._functions.get(function_name)

    def has_function(self, function_name):
        return function_name in self._functions

    def invoke(self, function_name, name=None, args=None, kwargs=None):
        """
        Invoke a function by name, populate parameters
        returns: A tuple with first element 'True' if the function was invoked
                 and second element is the function's return value.
                 A tuple with first element 'False' if the function was NOT
                 invoked, second element is None
        """
        # Initialize
        if not name:
            name = function_name
        if args is None:
            args = list()
        kwargs = dict() if kwargs is None else dict(kwargs)
        # Is there a function registered?
        if not self.has_function(function_name):
            self._debug(f"{name}: No function registered as '{function_name}', skipping")
            return False, None
        # Is the funciton disabled?
        if ENABLE in kwargs:
            if not kwargs[ENABLE]:
                self._debug(f"{name}: Is disabled in config file ({function_name}.{ENABLE} = {kwargs[ENABLE]}), skipping")
                return False, None
            del kwargs[ENABLE]  # We don't want to pass this as an argument
        # Populate parameters
        f = self.get_function(function_name)
        # Invoke function
        self._debug(f"Invoking '{name}' function '{f.__name__}': Start, args={args}, kwargs={kwargs}")
        time_start = time.process_time()
        retval = f(*args, **kwargs)
        time_end = time.process_time()
        if retval is None:
            self._warning(f"Invoking '{name}' function '{f.__name__}': Returned 'None'")
        self._debug(f"Invoked '{name}' function '{f.__name__}': Elapsed {(time_end - time_start):.4f} seconds")
        return True, retval

    def register(self, name, f):
        """ Register a function """
        self._debug(f"Register: key '{name}', function '{f.__name__}'")
        if name not in REGISTRATION_KEYS:
            raise Exception(f"Invalid registration key '{name}', valid keys are {REGISTRATION_KEYS}")
        self._functions[name] = f

    def reset(self):
        """ Reset registry """
        self._functions = dict()


# Decorators register the functions to LogMl singleton
def register(name, f):
    MlRegistry().register(name, f)
    return f


def dataset_augment(f):
    return register(DATASET_AUGMENT, f)


def dataset_create(f):
    return register(DATASET_CREATE, f)


def dataset_inout(f):
    return register(DATASET_INOUT, f)


def dataset_load(f):
    return register(DATASET_LOAD, f)


def dataset_preprocess(f):
    return register(DATASET_PREPROCESS, f)


def dataset_save(f):
    return register(DATASET_SAVE, f)


def dataset_split(f):
    return register(DATASET_SPLIT, f)


def model_create(f):
    return register(MODEL_CREATE, f)


def model_evaluate(f):
    return register(MODEL_EVALUATE, f)


def model_predict(f):
    return register(MODEL_PREDICT, f)


def model_save(f):
    return register(MODEL_SAVE, f)


def model_train(f):
    return register(MODEL_TRAIN, f)
