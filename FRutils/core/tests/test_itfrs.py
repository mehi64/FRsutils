import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from models.itfrs import ITFRS
import tnorms as tn
import implicators as imp

def test_itfrs_approximations():
    # sim_matrix = np.array([
    #     [1.0, 0.8, 0.4],
    #     [0.8, 1.0, 0.5],
    #     [0.4, 0.5, 1.0]
    # ])
    # labels = np.array([1, 1, 0])
    sim_matrix = np.array([
        [1.0, 0.8, 0.4, 0.5, 0.8],
        [0.8, 1.0, 0.5, 0.4, 0.9],
        [0.4, 0.5, 1.0, 0.9, 0.4],
        [0.5, 0.4, 0.9, 1.0, 0.5],
        [0.8, 0.9, 0.4, 0.5, 1.0]
    ])
    labels = np.array([1, 1, 0, 0, 1])
    model = ITFRS(sim_matrix, labels, tnorm=tn.tn_minimum, implicator=imp.imp_kleene_dienes)
    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))
