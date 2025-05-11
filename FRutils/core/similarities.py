# frutil/similarities.py
"""
Module for calculating similarities and similarity matrices.
"""
import numpy as np


def linear_similarity(v1: float, v2: float) -> float:
    """
    computes max(0, 1 - |v1 - v2|)
    
    :param v1: float in range [0, 1]
    :param v2: float in range [0, 1]
    :return: float in range [0, 1]
    """
    sim = max(0.0, 1.0 - abs(v1 - v2))
    if not ((0.0 <= v1 <= 1.0) and (0.0 <= v2 <= 1.0)):
        raise ValueError("inputs must be in [0.0, 1.0].")
    return sim

def compute_feature_similarities(x1: np.ndarray, x2: np.ndarray, sim_func) -> np.ndarray:
    """
    Compute the similarity between two feature vectors using the specified similarity function.
    output is a numpy.ndarray of length n, where n is the number of features in the input vectors."""
    fs = np.vectorize(sim_func)(x1, x2)
    return fs

def aggregate_similarities(similarities: np.ndarray, agg_func) -> float:
    if not ((0.0 <= similarities).all() and (similarities <= 1.0).all()):
        raise ValueError("All similarities must be in the range [0.0, 1.0].")
    agg = agg_func(similarities)
    return agg

def compute_similarity_matrix(X: np.ndarray, sim_func, agg_func) -> np.ndarray:
    n = X.shape[0]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sims = np.array([
            aggregate_similarities(
                compute_feature_similarities(X[i], X[j], sim_func),
                agg_func
            ) for j in range(n)
        ])
        sim_matrix[i, :] = sims
    return sim_matrix

def compute_instance_similarities(instance: np.ndarray, X: np.ndarray, sim_func, agg_func) -> np.ndarray:
    return np.array([
        aggregate_similarities(compute_feature_similarities(instance, other, sim_func), agg_func)
        for other in X
    ])