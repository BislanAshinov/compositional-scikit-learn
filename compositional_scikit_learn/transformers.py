import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from skbio.stats.composition import clr, alr, ilr, multiplicative_replacement
from .utils import _apply_transform


class LRTransorm(BaseEstimator, TransformerMixin):
    """
    Log-ratio transforms for compositional data columns.
    
    Params:
        transform_name: 
            Type of log-ratio transform: "clr", "alr" or "ilr".
            https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations 
        d: 
            Dimension of compositional data vector
        composition_columns:
            Names or indexes of compositional data columns. Names for pandas.DataFrame, indexes for np.ndarray.
        delta:
            A small number to be used to replace zeros for multiplicative replacement 
            More info: 
            J. A. Martin-Fernandez. `Dealing With Zeros and Missing Values in Compositional Data Sets Using Nonparametric Imputation`
            http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.multiplicative_replacement.html
    """
    def __init__(self, 
                 transform_type: str = "clr", 
                 d = None,
                 composition_columns=None,
                 delta=1e-6) -> None:
        super().__init__()

        if transform_type == "clr":
            self.transform_f = clr
        elif transform_type == "alr":
            self.transform_f = alr
        elif transform_type == "ilr":
            self.transform_f = ilr
        else:
            raise TypeError("Unsupported transform type")
        
        self._d = d
        self.composition_columns = composition_columns
        self._mpr = multiplicative_replacement
        self._delta = delta
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data.
        return self

    def transform(self, X):
        return _apply_transform(self, X)
        
            