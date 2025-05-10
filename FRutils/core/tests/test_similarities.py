import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import similarities
import tnorms as tn

def test_linear_similarity():
    assert similarities.linear_similarity(1.0, 1.0) == 1.0
    assert similarities.linear_similarity(0.0, 1.0) == 0.0
    assert similarities.linear_similarity(0.3, 0.5) == 0.8

def test_compute_feature_similarities():
    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.8])
    sim = similarities.compute_feature_similarities(x1, x2, sim_func=similarities.linear_similarity)
    assert np.allclose(sim, [0.8, 0.4])
    assert sim.shape == x1.shape
    assert (0.0 <= sim).all() and (sim <= 1.0).all()

def test_aggregate_similarities():
    sims = np.array([0.8, 0.9, 0.56])
    agg = similarities.aggregate_similarities(sims, agg_func=tn.tn_minimum)
    assert agg == 0.56
    assert 0.0 <= agg <= 1.0
    
    agg = similarities.aggregate_similarities(sims, agg_func=tn.tn_product)
    assert np.isclose(agg, 0.4032)
    assert 0.0 <= agg <= 1.0

def test_compute_similarity_matrix():
    X = np.array([[0.1, 0.5], [0.5, 1.0],[0.7, 0.2],[0.1, 0.9]])
    sim_matrix = similarities.compute_similarity_matrix(X, sim_func=similarities.linear_similarity, agg_func=tn.tn_product)
    assert sim_matrix.shape == (4, 4)
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all()


def test_compute_instance_similarities_basic():
    X = np.array([
        [0.0, 0.5],
        [0.5, 0.5],
        [1.0, 0.5]
    ])
    instance = np.array([0.25, 0.7])
    sims = similarities.compute_instance_similarities(instance, X, sim_func=similarities.linear_similarity, agg_func=tn.tn_minimum)
    expected = np.array([
        min(similarities.linear_similarity(0.25, 0.0), similarities.linear_similarity(0.7, 0.5)),
        min(similarities.linear_similarity(0.25, 0.5), similarities.linear_similarity(0.7, 0.5)),
        min(similarities.linear_similarity(0.25, 1.0), similarities.linear_similarity(0.7, 0.5))
    ])
    np.testing.assert_allclose(sims, expected, rtol=1e-5)

def test_compute_instance_similarities_output_range():
    X = np.array([
        [0.1, 0.9],
        [0.4, 0.4],
        [0.9, 0.1]
    ])
    instance = np.array([0.5, 0.5])
    sims = similarities.compute_instance_similarities(instance, X, sim_func=similarities.linear_similarity, agg_func=tn.tn_minimum)
    assert np.all((0.0 <= sims) & (sims <= 1.0)), "All similarity values should be in range [0.0, 1.0]"

def test_compute_instance_similarities_shape():
    X = np.random.rand(10, 5)
    instance = X[0]
    sims = similarities.compute_instance_similarities(instance, X, sim_func=similarities.linear_similarity, agg_func=tn.tn_minimum)
    assert sims.shape == (10,), "Output should have shape (n_samples,)"
