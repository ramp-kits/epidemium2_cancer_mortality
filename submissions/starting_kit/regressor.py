from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = make_pipeline(
            SimpleImputer(strategy='median'),
            ExtraTreesRegressor(
                n_estimators=10, max_leaf_nodes=10, random_state=61))

    def fit(self, X, y):
        return self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
