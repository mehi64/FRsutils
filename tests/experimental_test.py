import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRsutils/core')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRsutils/core/models')))


#######################################################################
# t-norms
import tnorms
import test_tnorms as tt

tt.test_tn_minimum_scalar_values()
tt.test_tn_minimum_nxnx2_map_values()
tt.test_tn_product_scalar_values()
tt.test_tn_product_nxnx2_map_values()

######################################################################
# implicators
import implicators
import test_implicators as timp

timp.test_goedel_implicator_outputs()
timp.test_gaines_implicator_outputs()
timp.test_luk_implicator_outputs()
timp.test_kd_implicator_outputs()
timp.test_reichenbach_implicator_outputs()
######################################################################
# similarities
import similarities
import test_similarities as ts
ts.test_compute_similarity_matrix_with_linear_similarity_product_tnorm()
ts.test_compute_similarity_matrix_with_linear_similarity_minimum_tnorm()

import itfrs
import test_itfrs as ti
ti.test_itfrs_approximations_reichenbach_imp_product_tnorm()
# ti.test_itfrs_approximations_KD_imp_product_tnorm()
# ti.test_itfrs_approximations_Luk_imp_product_tnorm()
# ti.test_itfrs_approximations_Goedel_imp_product_tnorm()
# ti.test_itfrs_approximations_Gaines_imp_product_tnorm()








