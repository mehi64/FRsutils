import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import similarities
import tnorms
# import itfrs

import test_similarities as ts
import test_itfrs as ti
import test_tnorms as tt

tt.test_tn_minimum()

# similarities
# ts. test_linear_similarity()
# ts.test_compute_feature_similarities_linear()
# ts.test_aggregate_similarities()

ts.test_compute_similarity_matrix()

ts.test_compute_instance_similarities_basic()
ts.test_compute_instance_similarities_output_range()

values = np.array([
        [1.0, 0.2, 0.4, 0.5, 0.8],
        [0.8, 0.1, 0.5, 0.4, 0.9],
        [0.4, 0.3, 1.0, 0.9, 0.4],
        [0.5, 0.8, 0.02, 1.0, 0.7]
    ])
# values = np.array([1.0, 0.2, 0.4, 0.5, 0.8])
a = np.min(values, axis=0)
print(a)

ti.test_itfrs_approximations()


