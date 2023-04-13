from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

import numpy as np
import pandas as pd

from skbio.stats.composition import clr, alr, ilr, multiplicative_replacement
from typing import List, Dict
from .utils import _apply_transform


class CompositionalKMeans(BaseEstimator, ClusterMixin):
    """
    KMeans with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for KMeans
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        if transform_type == "clr":
            self.transform_f = clr
        elif transform_type == "alr":
            self.transform_f = alr
        elif transform_type == "ilr":
            self.transform_f = ilr
        else:
            raise TypeError("Unsupported transform type")
        
        self.clusterer = KMeans(**kwargs)
        
        if isinstance(composition_columns, list) and all(isinstance(i, int) for i in composition_columns):
            self.composition_columns = np.array(composition_columns)
        else:
            self.composition_columns = composition_columns
        self._mpr = multiplicative_replacement
        self._delta = 1e-6
        
    def fit(self, X, y=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.clusterer.fit(X_transformed, y, sample_weight)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.fit_predict(X_transformed, y, sample_weight)

    def fit_transform(self, X, y=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.fit_transform(X_transformed, y, sample_weight)

    def get_feature_names_out(self, input_features=None):
        return self.clusterer.get_feature_names_out(input_features)

    def get_params(self, deep=True):
        return self.clusterer.get_params(deep)

    def predict(self, X, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.predict(X_transformed, sample_weight)

    def score(self, X, y=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.score(X_transformed, y, sample_weight)

    def set_output(self, transform=None):
        self.clusterer.set_output(transform)
        return self

    def set_params(self, **params):
        self.clusterer.set_params(**params)
        return self

    def transform(self, X):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.transform(X_transformed)
    

class CompositionalDBSCAN(BaseEstimator, ClusterMixin):
    """
    DBSCAN with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for DBSCAN
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.clusterer = DBSCAN(**kwargs)
        if transform_type == "clr":
            self.transform_f = clr
        elif transform_type == "alr":
            self.transform_f = alr
        elif transform_type == "ilr":
            self.transform_f = ilr
        else:
            raise TypeError("Unsupported transform type")
        
        if isinstance(composition_columns, list) and all(isinstance(i, int) for i in composition_columns):
            self.composition_columns = np.array(composition_columns)
        else:
            self.composition_columns = composition_columns
        self._mpr = multiplicative_replacement
        self._delta = 1e-6
    
    def fit(self, X, y=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.clusterer.fit(X_transformed, y, sample_weight)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.fit_predict(X_transformed, y, sample_weight)

    def get_params(self, deep=True):
        return self.clusterer.get_params(deep)

    def set_params(self, **params):
        self.clusterer.set_params(**params)
        return self


class CompositionalAgglomerativeClustering(BaseEstimator, ClusterMixin):
    """
    AgglomerativeClustering with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for AgglomerativeClustering
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.clusterer = AgglomerativeClustering(**kwargs)
        if transform_type == "clr":
            self.transform_f = clr
        elif transform_type == "alr":
            self.transform_f = alr
        elif transform_type == "ilr":
            self.transform_f = ilr
        else:
            raise TypeError("Unsupported transform type")
        
        if isinstance(composition_columns, list) and all(isinstance(i, int) for i in composition_columns):
            self.composition_columns = np.array(composition_columns)
        else:
            self.composition_columns = composition_columns
        self._mpr = multiplicative_replacement
        self._delta = 1e-6
    
    def fit(self, X, y=None):
        X_transformed = _apply_transform(self, X)
        self.clusterer.fit(X_transformed, y)
        return self

    def fit_predict(self, X, y=None):
        X_transformed = _apply_transform(self, X)
        return self.clusterer.fit_predict(X_transformed, y)

    def get_params(self, deep=True):
        return self.clusterer.get_params(deep)

    def set_params(self, **params):
        self.clusterer.set_params(**params)
        return self

