from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
import pandas as pd

from skbio.stats.composition import clr, alr, ilr, multiplicative_replacement
from typing import List, Dict
from .utils import _apply_transform


class CompositionalLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    LogisticRegression with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for LogisticRegression
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
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
        
        self.classifier = LogisticRegression(**kwargs)
        
        if isinstance(composition_columns, list) and all(isinstance(i, int) for i in composition_columns):
            self.composition_columns = np.array(composition_columns)
        else:
            self.composition_columns = composition_columns
        self._mpr = multiplicative_replacement
        self._delta = 1e-6

    def fit(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.classifier.fit(X_transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight=sample_weight)

    def decision_function(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.decision_function(X_transformed)

    def densify(self):
        self.classifier.densify()
        return self

    def get_params(self, deep=True):
        return self.classifier.get_params(deep=deep)

    def predict_log_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_log_proba(X_transformed)

    def predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed)

    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self

    def sparsify(self):
        self.classifier.sparsify()
        return self


class CompositionalSVC(BaseEstimator, ClassifierMixin):
    """
    SVC with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for SVC
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.classifier = SVC(**kwargs)
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
        self.classifier.fit(X_transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight=sample_weight)

    def decision_function(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.decision_function(X_transformed)

    def get_params(self, deep=True):
        return self.classifier.get_params(deep=deep)

    def predict_log_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_log_proba(X_transformed)
    
    def predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed)

    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self


class CompositionalSGDClassifier(BaseEstimator, ClassifierMixin):
    """
    SGDClassifier with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for SGDClassifier
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
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
        
        self.classifier = SGDClassifier(**kwargs)
        
        if isinstance(composition_columns, list) and all(isinstance(i, int) for i in composition_columns):
            self.composition_columns = np.array(composition_columns)
        else:
            self.composition_columns = composition_columns
        self._mpr = multiplicative_replacement
        self._delta = 1e-6

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.classifier.fit(X_transformed, y, coef_init=coef_init, intercept_init=intercept_init, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight=sample_weight)

    def decision_function(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.decision_function(X_transformed)

    def densify(self):
        self.classifier.densify()
        return self

    def get_params(self, deep=True):
        return self.classifier.get_params(deep=deep)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        self.classifier.partial_fit(X_transformed, y, classes=classes, sample_weight=sample_weight)
        return self

    def predict_log_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_log_proba(X_transformed)

    def predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed)

    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self

    def sparsify(self):
        self.classifier.sparsify()
        return self

    
class CompositionalDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    DecisionTreeClassifier with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for DecisionTreeClassifier
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.classifier = DecisionTreeClassifier(**kwargs)
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
        self.classifier.fit(X_transformed, y, sample_weight=sample_weight, check_input=check_input)
        return self
    
    def predict(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed, check_input=check_input)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight=sample_weight)

    def apply(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.classifier.apply(X_transformed, check_input=check_input)

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.cost_complexity_pruning_path(X_transformed, y, sample_weight=sample_weight)

    def decision_path(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.classifier.decision_path(X_transformed, check_input=check_input)

    def get_depth(self):
        return self.classifier.get_depth()
    
    def get_n_leaves(self):
        return self.classifier.get_n_leaves()

    def get_params(self, deep=True):
        return self.classifier.get_params(deep=deep)

    def predict_log_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_log_proba(X_transformed)

    def predict_proba(self, X, check_input=True):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed, check_input=check_input)

    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self


class CompositionalRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    RandomForestClassifier with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for RandomForestClassifier
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.classifier = RandomForestClassifier(**kwargs)
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
        self.classifier.fit(X_transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed)

    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight=sample_weight)

    def apply(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.apply(X_transformed)

    def decision_path(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.decision_path(X_transformed)

    def get_params(self, deep=True):
        return self.classifier.get_params(deep=deep)

    def predict_log_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_log_proba(X_transformed)

    def predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed)

    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self


class CompositionalKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """
    KNeighborsClassifier with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for KNeighborsClassifier
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.classifier = KNeighborsClassifier(**kwargs)
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
        self.classifier.fit(X_transformed, y)
        return self
    
    def get_params(self, deep=True):
        return self.classifier.get_params(deep)
    
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        X_transformed = _apply_transform(self, X)
        return self.classifier.kneighbors(X_transformed, n_neighbors, return_distance)
    
    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        X_transformed = _apply_transform(self, X)
        return self.classifier.kneighbors_graph(X_transformed, n_neighbors, mode)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed)
    
    def predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self


class CompositionalGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    GradientBoostingClassifier with pre-transform compositional data in input data.

    Params:
        transform_type:
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        composition_columns:
            Names or indexes of input container, which contains compositional vector. 
            For pd.DataFrame must be list of strings. For np.ndarray must be list of int or np.ndarray
        **kwargs:
            Params for GradientBoostingClassifier
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 composition_columns=None, 
                 **kwargs) -> None:
        super().__init__()
        self.classifier = GradientBoostingClassifier(**kwargs)
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
        return self.classifier.apply(X_transformed)
    
    def decision_function(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.decision_function(X_transformed)
    
    def fit(self, X, y, sample_weight=None, monitor=None):
        X_transformed = _apply_transform(self, X)
        self.classifier.fit(X_transformed, y, sample_weight, monitor)
        return self
    
    def get_params(self, deep=True):
        return self.classifier.get_params(deep)
    
    def predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict(X_transformed)
    
    def predict_log_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_log_proba(X_transformed)
    
    def predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.predict_proba(X_transformed)
    
    def score(self, X, y, sample_weight=None):
        X_transformed = _apply_transform(self, X)
        return self.classifier.score(X_transformed, y, sample_weight)
    
    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self
    
    def staged_decision_function(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.staged_decision_function(X_transformed)
    
    def staged_predict(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.staged_predict(X_transformed)
    
    def staged_predict_proba(self, X):
        X_transformed = _apply_transform(self, X)
        return self.classifier.staged_predict_proba(X_transformed)


