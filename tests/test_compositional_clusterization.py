import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from compositional_scikit_learn.clusterization import CompositionalKMeans, CompositionalDBSCAN, CompositionalAgglomerativeClustering
from compositional_scikit_learn import _apply_transform, softmax


@pytest.fixture
def test_data():
    X, _ = make_blobs(n_samples=1000, centers=10, n_features=7, random_state=42)
    X[:, :4] = softmax(X[:, :4])
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    return X, X_df


@pytest.mark.parametrize("clustering_class", [CompositionalAgglomerativeClustering, CompositionalDBSCAN, CompositionalKMeans])
@pytest.mark.parametrize("transform_type", ["clr", "alr", "ilr"])
def test_compositional_clustering(test_data, clustering_class, transform_type):
    X, X_df = test_data
    comp_clustering = clustering_class(transform_type=transform_type, composition_columns=[0, 1, 2, 3])
    labels = comp_clustering.fit_predict(X)

    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        assert silhouette > 0.2
    else:
        assert set(labels) == {-1} or set(labels) == {0}
