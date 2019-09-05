#!/usr/bin/env python


class ModelConfidence:
    ''' Calculate mean and variance of the estimation base on multiple modes '''

    def __init__(self, models, max_workers=None):
        '''
        Models can be either a is a list or an Ensembl estimator
        from SciKitLearn (e.g. a ranfdom forest)
        '''
        self.models = models.estimators_ if 'estimators_' in models else models
        self.max_workers = max_workers
        self.predictions = None
        self.figsize = (16, 10)

    def plot(self, bins=100):
        plt.figure(figsize=self.figsize)
        pred = self.predictions.mean(axis=0)
        plt.hist(pred, bins=bins)
        plt.figure(figsize=self.figsize)
        plt.hist(self.std(), bins=bins)

    def predict(self, x):
        ''' Predict using all models, return mean prediction '''
        self.predictions = np.stack([m.predict(x) for m in self.models])
        return self.predictions.mean(axis=0)

    def predict_single_sample(self, x, sample_number):
        ''' Predict using all models, return mean prediction for sample number '''
        xi = x_val.iloc[sample_number]
        xi = np.array(xi).reshape(1, -1)
        return self.predict(x)

    def std(self):
        " Standard deviation of previous predictions "
        return self.predictions.std(axis=0)

    def __repr__(self):
        return f"number of models: {len(self.models)}, mean : {np.mean(self.predictions)}, std: {np.std(self.predictions)}, predictions: {self.predictions}"
