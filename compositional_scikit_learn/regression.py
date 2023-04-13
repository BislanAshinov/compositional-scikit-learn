from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np
import pandas as pd

from skbio.stats.composition import clr, alr, ilr, multiplicative_replacement
from .utils import _apply_transform


class CompositionalLinearRegression(BaseEstimator, RegressorMixin):
    """
    LinearRegression with pre-transform compositional data in input data.
    
    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for LinearRegression
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = LinearRegression(**kwargs)
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
    
    def fit(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight)
        return self
    
    def get_params(self, deep=True):
        return self.regressor.get_params(deep)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalRidge(BaseEstimator, RegressorMixin):
    """
    Ridge with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for Ridge
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = Ridge(**kwargs)
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
    
    def fit(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight)
        return self
    
    def get_params(self, deep=True):
        return self.regressor.get_params(deep)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalLasso(BaseEstimator, RegressorMixin):
    """
    Lasso with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for Lasso
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = Lasso(**kwargs)
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
    
    def fit(self, X, y, sample_weight=None, check_input=True):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight, check_input)
        return self
    
    def get_params(self, deep=True):
        return self.regressor.get_params(deep)
    
    def path(self, X, y, *args, **kwargs):
        X_transformed = _apply_transform(self, X)
        return self.regressor.path(X_transformed, y, *args, **kwargs)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalElasticNet(BaseEstimator, RegressorMixin):
    """
    ElasticNet with pre-transform compositional data in input data.
    
    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for ElasticNet
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = ElasticNet(**kwargs)
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
    
    def fit(self, X, y, sample_weight=None, check_input=True):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight, check_input)
        return self
    
    def get_params(self, deep=True):
        return self.regressor.get_params(deep)
    
    def path(self, X, y, *args, **kwargs):
        X_transformed = _apply_transform(self, X)
        return self.regressor.path(X_transformed, y, *args, **kwargs)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalSVR(BaseEstimator, RegressorMixin):
    """
    SVR with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for SVR
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = SVR(**kwargs)
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
    
    def fit(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight)
        return self
    
    def get_params(self, deep=True):
        return self.regressor.get_params(deep)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """
    DecisionTreeRegressor with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for DecisionTreeRegressor
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = DecisionTreeRegressor(**kwargs)
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

    def apply(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.regressor.apply(X_transformed, check_input)

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.cost_complexity_pruning_path(X_transformed, y, sample_weight)

    def decision_path(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.regressor.decision_path(X_transformed, check_input)

    def fit(self, X, y, sample_weight=None, check_input=True):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight, check_input)
        return self

    def get_depth(self):
        return self.regressor.get_depth()

    def get_n_leaves(self):
        return self.regressor.get_n_leaves()

    def get_params(self, deep=True):
        return self.regressor.get_params(deep)

    def predict(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed, check_input)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)

    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    RandomForestRegressor with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for RandomForestRegressor
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = RandomForestRegressor(**kwargs)
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

    def apply(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.apply(X_transformed)

    def decision_path(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.decision_path(X_transformed)

    def fit(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight)
        return self

    def get_params(self, deep=True):
        return self.regressor.get_params(deep)

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)

    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self
    

class CompositionalKNeighborsRegressor(BaseEstimator, RegressorMixin):
    """
    KNeighborsRegressor with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for KNeighborsRegressor
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = KNeighborsRegressor(**kwargs)
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

    def fit(self, X, y):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y)
        return self

    def get_params(self, deep=True):
        return self.regressor.get_params(deep)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        X_transformed = _apply_transform(self, X)
        return self.regressor.kneighbors(X_transformed, n_neighbors, return_distance)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        X_transformed = _apply_transform(self, X)
        return self.regressor.kneighbors_graph(X_transformed, n_neighbors, mode)

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)

    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self


class CompositionalGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    GradienBoostingRegressor with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for GradienBoostingRegressor
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.regressor = GradientBoostingRegressor(**kwargs)
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

    def apply(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.apply(X_transformed)

    def fit(self, X, y, sample_weight=None, monitor=None):
        X_transformed = _apply_transform(self, X)
        self.regressor.fit(X_transformed, y, sample_weight, monitor)
        return self

    def get_params(self, deep=True):
        return self.regressor.get_params(deep)

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.regressor.score(X_transformed, y, sample_weight)

    def set_params(self, **params):
        self.regressor.set_params(**params)
        return self

    def staged_predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.regressor.staged_predict(X_transformed)

