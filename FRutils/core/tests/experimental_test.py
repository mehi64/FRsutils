import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))


# import implicators
import tnorms
# import similarities
# import itfrs


# import test_implicators as timp
import test_tnorms as tt
# import test_similarities as ts
# import test_itfrs as ti

# # implicators

# timp.test_goedel_implicator_outputs()
# timp.test_gaines_implicator_outputs()
# timp.test_luk_implicator_outputs()
# timp.test_kd_implicator_outputs()
# timp.test_reichenbach_implicator_outputs()

#######################################################################
# t-norms
tt.test_tn_minimum_scalar_values()
tt.test_tn_minimum_nxnx2_map_values()




