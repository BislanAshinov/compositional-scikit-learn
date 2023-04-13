import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from compositional_scikit_learn import LRTransorm
from compositional_scikit_learn import _apply_transform, softmax


@pytest.fixture
def test_data():
    X, y = make_regression(n_samples=1000, n_features=7, random_state=42)
    X[:, :4] = softmax(X[:, :4])
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    return X, X_df, y


@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_lr_transform_dataframe(test_data, transform_type):
    X, X_df, y = test_data
    transformer = LRTransorm(transform_type=transform_type, composition_columns=['a', 'b', 'c', 'd'])
    X_transformed = transformer.fit_transform(X_df)

    assert isinstance(X_transformed, pd.DataFrame)
    if transform_type in ('ilr', 'alr'):
        assert X_transformed.shape == (X_df.shape[0], X_df.shape[1] - 1)
    else:
        assert X_transformed.shape == X_df.shape


@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_lr_transform_ndarray(test_data, transform_type):
    X, X_df, y = test_data
    transformer = LRTransorm(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, np.ndarray)
    if transform_type in ('ilr', 'alr'):
        assert X_transformed.shape == (X.shape[0], X.shape[1] - 1)
    else:
        assert X_transformed.shape == X.shape


@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_lr_transform_pipeline(test_data, transform_type):
    X, X_df, y = test_data
    transformer = LRTransorm(transform_type=transform_type, composition_columns=['a', 'b', 'c', 'd'])
    pipeline = Pipeline([
        ('lr_transform', transformer),
        ('linear_regression', LinearRegression())
    ])
    pipeline.fit(X_df, y)
    y_pred = pipeline.predict(X_df)

    assert r2_score(y, y_pred) > 0.7


