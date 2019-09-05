

from scipy.cluster import hierarchy as hc
from sklearn.tree import export_graphviz
from IPython import display


class FeatureImportance:
    '''
    Estimate feature importance.
    How it works: Suffle a column and analyze how model performance is
    degraded. Most important features will make the model perform much
    worse when shuffled, unimportant features will not affect performance
    '''

    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y
        self.performance = dict()
        self.importance = None
        self.figsize = (16, 10)

    def analyze(self, verbose=False):
        # Base performance
        x_copy = self.x.copy()
        pred = self.model.predict(x_copy)
        # Shuffle each solumn
        for c in self.x:
            # Shuffle column 'c'
            x_copy = self.x.copy()
            xi = np.random.permutation(x_copy[c])
            x_copy[c] = xi
            # How did it perform
            pred_c = self.model.predict(x_copy)
            perf_c = self.rmse(pred, pred_c)
            self.performance[c] = perf_c
            if verbose:
                print(f"{c}: {perf_c}")
        # List of items sorted by importance (most important first)
        self.importance = sorted(self.performance.items(), key=lambda kv: kv[1], reverse=True)

    def tree_graph(self, max_depth=3, file_dot='tree.dot', file_png='tree.png'):
        # Train a single tree with all the samples
        m = RandomForestRegressor(n_estimators=1, max_depth=max_depth, bootstrap=False)
        m.fit(x_train, y_train)
        # Export the tree to a graphviz 'dot' format
        str_tree = export_graphviz(m.estimators_[0],
                                   out_file='tree.dot',
                                   feature_names=x_train.columns,
                                   filled=True,
                                   rounded=True)
        print(f"Created dot file: '{file_dot}'")
        # Convert 'dot' to 'png'
        args = ['dot', '-Tpng', file_dot, '-o', 'tree.png']
        subprocess.run(args)
        print(f"Created image: '{file_png}'")

    def most_important(self, importance_threshold=None, ratio_to_most_important=100, df=None):
        """
        Select features to keep either using an absolute value or
        a ratio to most important feature
        """
        if ratio_to_most_important is not None:
            most_important = fi.importance[0]
            importance_threshold = most_important[1] / ratio_to_most_important

        important_features = [f[0] for f in self.importance if f[1] > importance_threshold]
        unimportant_features = [f[0] for f in self.importance if f[0] not in important_features]
        return important_features, unimportant_features

    def plot(self, x=None):
        " Plot importance distributions "
        imp_x = np.array([f[0] for f in self.importance])
        imp_y = np.array([f[1] for f in self.importance])
        # Show bar plot
        plt.figure(figsize=self.figsize)
        plt.barh(imp_x, imp_y)
        # Show line plot
        plt.figure(figsize=self.figsize)
        plt.plot(imp_x, imp_y)

    def rmse(self, x, y):
        return math.sqrt(((x - y)**2).mean())

    def __repr__(self):
        return "\n".join([f"{f[0]} : {f[1]}" for f in fi.importance])
