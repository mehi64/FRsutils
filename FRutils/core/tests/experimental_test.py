import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import similarities

import test_similarities as ts

ts.test_compute_feature_similarities()
ts.test_aggregate_similarities()

ts.test_compute_similarity_matrix()

ts.test_compute_instance_similarities_basic()
ts.test_compute_instance_similarities_output_range()


