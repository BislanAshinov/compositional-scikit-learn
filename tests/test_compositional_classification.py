import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from compositional_scikit_learn.classification import CompositionalLogisticRegression, CompositionalSGDClassifier, CompositionalSVC
from compositional_scikit_learn.classification import CompositionalKNeighborsClassifier, CompositionalDecisionTreeClassifier
from compositional_scikit_learn.classification import CompositionalRandomForestClassifier, CompositionalGradientBoostingClassifier
from compositional_scikit_learn import _apply_transform, softmax


@pytest.fixture
def test_data():
    X, y = make_classification(n_samples=1000, n_features=7, random_state=42)
    X[:, :4] = softmax(X[:, :4])
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    return X, X_df, y


@pytest.mark.parametrize("classifier_class", [CompositionalLogisticRegression, CompositionalSVC, CompositionalSGDClassifier])
@pytest.mark.parametrize("transform_type", ["clr", "alr", "ilr"])
def test_compositional_classifiers_1(test_data, classifier_class, transform_type):
    X, X_df, y = test_data
    comp_classifier = classifier_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    comp_classifier.fit(X, y)
    y_pred = comp_classifier.predict(X)

    assert accuracy_score(y, y_pred) > 0.8


@pytest.mark.parametrize("classifier_class", [CompositionalKNeighborsClassifier, CompositionalDecisionTreeClassifier])
@pytest.mark.parametrize("transform_type", ["clr", "alr", "ilr"])
def test_compositional_classifiers_2(test_data, classifier_class, transform_type):
    X, X_df, y = test_data
    comp_classifier = classifier_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    comp_classifier.fit(X, y)
    y_pred = comp_classifier.predict(X)

    assert accuracy_score(y, y_pred) > 0.8


@pytest.mark.parametrize("classifier_class", [CompositionalRandomForestClassifier, CompositionalGradientBoostingClassifier])
@pytest.mark.parametrize("transform_type", ["clr", "alr", "ilr"])
def test_compositional_classifiers_3(test_data, classifier_class, transform_type):
    X, X_df, y = test_data
    comp_classifier = classifier_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    comp_classifier.fit(X, y)
    y_pred = comp_classifier.predict(X)

    assert accuracy_score(y, y_pred) > 0.8