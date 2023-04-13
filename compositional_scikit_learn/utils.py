import numpy as np
import pandas as pd
from skbio.stats.composition import alr, ilr, clr, multiplicative_replacement


def _apply_transform(estimator, X):
    if estimator.composition_columns is not None:
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            X_composition = X[estimator.composition_columns].values
        else:
            X_transformed = np.copy(X)
            X_composition = X[:, estimator.composition_columns]
    else:
        X_transformed = X
        X_composition = X

    if (X_composition == 0).any():
        X_composition = estimator._mpr(X_composition, estimator._delta)

    X_composition_transformed = estimator.transform_f(X_composition)

    if estimator.transform_f in (ilr, alr):
        reduced_columns = estimator.composition_columns[:-1]
    else:
        reduced_columns = estimator.composition_columns

    if estimator.composition_columns is not None:
        if isinstance(X, pd.DataFrame):
            if X_composition_transformed.ndim == 1:
                X_transformed.loc[:, reduced_columns] = X_composition_transformed[:, np.newaxis]
            else:
                X_transformed.loc[:, reduced_columns] = X_composition_transformed
            if estimator.transform_f in (ilr, alr):
                X_transformed.drop(columns=estimator.composition_columns[-1], inplace=True)
        else:
            if X_composition_transformed.ndim == 1:
                X_composition_transformed = X_composition_transformed[:, np.newaxis]
            else:
                X_transformed[:, reduced_columns] = X_composition_transformed
            if estimator.transform_f in (ilr, alr):
                X_transformed = np.delete(X_transformed, estimator.composition_columns[-1], axis=1)
    else:
        X_transformed = X_composition_transformed

    return X_transformed


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
