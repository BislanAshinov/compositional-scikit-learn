import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from compositional_scikit_learn.regression import _apply_transform, CompositionalLinearRegression, CompositionalRidge, CompositionalLasso
from compositional_scikit_learn.regression import CompositionalElasticNet, CompositionalSVR, CompositionalKNeighborsRegressor
from compositional_scikit_learn.regression import CompositionalRandomForestRegressor, CompositionalGradientBoostingRegressor, CompositionalDecisionTreeRegressor
from compositional_scikit_learn import _apply_transform, softmax


@pytest.fixture
def test_data():
    X, y = make_regression(n_samples=1000, n_features=7, random_state=42)
    X[:, :4] = softmax(X[:, :4])
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    return X, X_df, y


@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_apply_transform_dataframe(test_data, transform_type):
    X, X_df, y = test_data
    estimator = CompositionalLinearRegression(transform_type=transform_type, composition_columns=['a', 'b', 'c', 'd'])
    X_transformed = _apply_transform(estimator, X_df)

    assert isinstance(X_transformed, pd.DataFrame)
    if transform_type in ('ilr', 'alr'):
        assert X_transformed.shape == (X_df.shape[0], X_df.shape[1] - 1)
    else:
        assert X_transformed.shape == X_df.shape


@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_apply_transform_ndarray(test_data, transform_type):
    X, X_df, y = test_data
    estimator = CompositionalLinearRegression(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    X_transformed = _apply_transform(estimator, X)
    
    print(type(X_transformed))

    assert isinstance(X_transformed, np.ndarray)
    if transform_type in ('ilr', 'alr'):
        assert X_transformed.shape == (X_df.shape[0], X_df.shape[1] - 1)
    else:
        assert X_transformed.shape == X.shape


@pytest.mark.parametrize("regressor_class", [CompositionalLinearRegression, CompositionalRidge, CompositionalLasso, CompositionalElasticNet])
@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_compositional_regressions_1(test_data, transform_type, regressor_class):
    X, X_df, y = test_data
    comp_lr = regressor_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    comp_lr.fit(X, y)
    y_pred = comp_lr.predict(X)

    assert r2_score(y, y_pred) > 0.6


@pytest.mark.parametrize("regressor_class", [CompositionalKNeighborsRegressor, CompositionalRandomForestRegressor])
@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_compositional_regressions_2(test_data, transform_type, regressor_class):
    X, X_df, y = test_data
    comp_lr = regressor_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    comp_lr.fit(X, y)
    y_pred = comp_lr.predict(X)

    assert r2_score(y, y_pred) > 0.7
    

@pytest.mark.parametrize("regressor_class", [CompositionalGradientBoostingRegressor, CompositionalDecisionTreeRegressor])
@pytest.mark.parametrize('transform_type', ['clr', 'alr', 'ilr'])
def test_compositional_regressions_3(test_data, transform_type, regressor_class):
    X, X_df, y = test_data
    comp_lr = regressor_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    comp_lr.fit(X, y)
    y_pred = comp_lr.predict(X)

    assert r2_score(y, y_pred) > 0.8
