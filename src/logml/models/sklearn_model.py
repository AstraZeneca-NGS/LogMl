import inspect
import sklearn
import sklearn.dummy
import sklearn.naive_bayes
import traceback

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsIC

from .model_cv import ModelCv
from ..util.etc import camel_to_snake


class SkLearnModel(ModelCv):
    ''' Create a wrapper for a SkLearn model '''
    def __init__(self, config, datasets, class_name, params, set_config=True):
        super().__init__(config, datasets, set_config)
        self.is_save_params = False
        # Set model specific paramters
        self.class_name = class_name
        model_name = camel_to_snake(class_name)
        self._debug(f"Class name: '{class_name}', model name: '{model_name}'")
        if set_config:
            self._set_from_config()
        # Set parameters
        if params:
            for n in params:
                self.__dict__[n] = params[n]
                self._debug(f"Setting {n} = {params[n]}")

    def clone(self):
        """ Clone the model """
        model_clone = super().clone()
        if model_clone.model is not None:
            model_clone.model = clone(model_clone.model)
        return model_clone

    def default_model_create(self, x, y):
        """ Create real model from SciKit learn """
        self._info(f"Creating model based on class '{self.class_name}'")
        class_reference = eval(self.class_name)
        args_spec = inspect.getargspec(class_reference.__init__)
        args_init = args_spec.args
        self._debug(f"Class '{self.class_name}' has constructor arguments: {args_init}")
        kwargs = dict()
        for arg in args_init:
            if arg in self.__dict__:
                val = self.__dict__.get(arg)
                kwargs[arg] = val
        self._debug(f"Invoking constructor '{self.class_name}', with arguments: {kwargs}")
        self.model = eval(f"{self.class_name}(**kwargs)")
        return True

    def default_model_predict(self, x):
        """ Default implementation for '@model_predict' """
        try:
            self._debug(f"Model predict ({self.class_name}): Start")
            y_hat = self.model.predict(x)
            self._debug(f"Model predict ({self.class_name}): End")
            return y_hat
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return None

    def default_model_train(self, x, y):
        """ Fit the model using training data """
        try:
            self._debug(f"Model train ({self.class_name}): Start")
            self.train_results = self.model.fit(x, y)
            self._debug(f"Model train ({self.class_name}): End")
            return True
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return False

    def loss_(self, x, y):
        """ Return the loss """
        ret = super().loss_(x, y)
        if ret is not None:
            return ret
        # Use sklearn model's 'score'
        return 1.0 - self.model.score(x, y)


class ModelSkExtraTreesRegressor(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100):
        super().__init__(config, datasets, class_name='ModelSkExtraTreesRegressor', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = ExtraTreesRegressor(n_jobs=self.n_jobs, n_estimators=self.n_estimators)
        return True


class ModelSkExtraTreesClassifier(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100):
        super().__init__(config, datasets, class_name='ModelSkExtraTreesClassifier', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = ExtraTreesClassifier(n_jobs=self.n_jobs, n_estimators=self.n_estimators)
        return True


class ModelSkGradientBoostingRegressor(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkGradientBoostingRegressor', params=None, set_config=False)
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = GradientBoostingRegressor()
        return True


class ModelSkGradientBoostingClassifier(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkGradientBoostingClassifier', params=None, set_config=False)
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = GradientBoostingClassifier()
        return True


class ModelSkLassoCV(SkLearnModel):
    def __init__(self, config, datasets, cv):
        super().__init__(config, datasets, class_name='ModelSkLassoCV', params=None, set_config=False)
        self.cv = cv
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = LassoCV(cv=self.cv)
        return True


class ModelSkLassoLarsAIC(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkLassoLarsAIC', params=None, set_config=False)
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = LassoLarsIC(criterion='aic')
        return True


class ModelSkLassoLarsBIC(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkLassoLarsBIC', params=None, set_config=False)
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = LassoLarsIC(criterion='bic')
        return True


class ModelSkRandomForestRegressor(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100, max_depth=None, bootstrap=True):
        super().__init__(config, datasets, class_name='ModelSkRandomForestRegressor', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = RandomForestRegressor(n_jobs=self.n_jobs, n_estimators=self.n_estimators, max_depth=self.max_depth, bootstrap=self.bootstrap)
        return True


class ModelSkRandomForestClassifier(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100, max_depth=None, class_weight='balanced', bootstrap=True):
        super().__init__(config, datasets, class_name='ModelSkRandomForestClassifier', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = RandomForestClassifier(n_jobs=self.n_jobs, n_estimators=self.n_estimators, max_depth=self.None, class_weight=self.class_weight, bootstrap=self.bootstrap)
        return True


class ModelSkRidgeCV(SkLearnModel):
    def __init__(self, config, datasets, cv):
        super().__init__(config, datasets, class_name='ModelSkRidgeCV', params=None, set_config=False)
        self.cv = cv
        self.default_model_create(None, None)

    def default_model_create(self, x, y):
        self.model = RidgeCV(cv=self.cv)
        return True
