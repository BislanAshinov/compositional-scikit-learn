import numpy as np 
from skbio.stats.composition import clr, multiplicative_replacement
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def _preprocess_compositional_data(X, delta=1e-6):
    X = np.array(X)
    if (X == 0).any():
        X = multiplicative_replacement(X, delta)
    return clr(X)

def compositional_euclidean_distance(X_true, X_pred):
    X_true_transformed = _preprocess_compositional_data(X_true)
    X_pred_transformed = _preprocess_compositional_data(X_pred)
    return euclidean_distances(X_true_transformed, X_pred_transformed)

def compositional_cosine_similarity(X_true, X_pred):
    X_true_transformed = _preprocess_compositional_data(X_true)
    X_pred_transformed = _preprocess_compositional_data(X_pred)
    return cosine_similarity(X_true_transformed, X_pred_transformed)

def compositional_Aitchison_distance(X_true, X_pred):
    return compositional_euclidean_distance(X_true, X_pred)

def kullback_leibler_divergence_between_composes(X_true, X_pred): 
    return np.sum(X_true * (np.log(X_true) - np.log(X_pred)), axis=0)

def jensen_shannon_divergence_between_composes(X_true, X_pred):
    m = 0.5 * (X_true + X_pred)
    return 0.5 * (kullback_leibler_divergence_between_composes(X_true, m) + kullback_leibler_divergence_between_composes(X_pred, m))

def hellinger2_distance(X_true, X_pred):
    return euclidean_distances(np.sqrt(X_true), np.sqrt(X_pred)) / np.sqrt(2)
