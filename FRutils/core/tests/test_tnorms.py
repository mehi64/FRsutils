import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tnorms
# Data used for running tests

data = np.array([0.5, 0.7, 0.34, 0.98, 1.2])

def test_tn_minimum():
    assert tnorms.tn_minimum(data) == 0.34

def test_tn_product():
    assert np.isclose(tnorms.tn_product(np.array([0.5, 0.5])), 0.25)
    assert np.isclose(tnorms.tn_product(data), 0.139944)
    
# def test_tn_lukasiewicz():
#     assert np.isclose(tnorms.tn_lukasiewicz(np.array([0.9, 0.3])), 0.2)

# def test_tn_drastic():
#     assert tnorms.tn_drastic(np.array([1.0, 0.7])) == 0.7
#     assert tnorms.tn_drastic(np.array([1.0, 1.0])) == 1.0
#     assert tnorms.tn_drastic(np.array([1.0, 0.0])) == 0.0
    
#     assert tnorms.tn_drastic(np.array([0.9, 0.7])) == 0.0
