import numpy as np
from FRsutils.core.similarities import Similarity, calculate_similarity_matrix
from FRsutils.core.tnorms import TNorm
from FRsutils.core.implicators import Implicator
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.utils.logger.logger_util import get_logger
import tests.syntetic_data_for_tests as sdf


data_synthteic = sdf.syntetic_dataset_factory()

TSTITFRS = data_synthteic.ITFRS_testing_dataset()
sim_matrix = TSTITFRS['sim_matrix']
y = TSTITFRS['y']

# X = np.array([
#     [0.10, 0.32, 0.48],
#     [0.20, 0.78, 0.93],
#     [0.73, 0.18, 0.28],
#     [0.91, 0.48, 0.73],
#     [1.00, 0.28, 0.47]
# ])
# labels = np.array([1, 1, 0, 1, 0])

# tnrm=TNorm.create("minimum")

# # Create Gaussian similarity with minimum tnorm
# similarity_func = SimilarityFunction.create("gaussian", tnrm, sigma=0.3)

# sim_matrix2 = calculate_similarity_matrix(X, similarity_func, tnrm)




# Create ITFRS model with product tnorm and gaines implicator
tnorm = TNorm.create("product")
implicator = Implicator.create("kleene")

logger = get_logger()

model = ITFRS(similarity_matrix=sim_matrix,
              labels=y,
              tnorm=tnorm,
              implicator=implicator,
              logger=logger)

upper = model.upper_approximation()
lower = model.lower_approximation()


print("tnorm:", tnorm.name)
print("implicator:", implicator.name)
print("\nLower Approximation:", lower)
print("Upper Approximation:", upper)

print("Done")
