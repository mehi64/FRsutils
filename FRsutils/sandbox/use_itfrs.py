import numpy as np
from FRsutils.core.similarities import SimilarityFunction
from FRsutils.core.tnorms import TNorm
from FRsutils.core.implicators import Implicator
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.utils.logger.logger_util import get_logger

# Example input data (5 samples, 3 features)
X = np.array([
    [0.10, 0.32, 0.48],
    [0.20, 0.78, 0.93],
    [0.73, 0.18, 0.28],
    [0.91, 0.48, 0.73],
    [1.00, 0.28, 0.47]
])
labels = np.array([1, 1, 0, 1, 0])

# Create Gaussian similarity with minimum tnorm
# similarity_func = SimilarityFunction.create("gaussian", tnorm=TNorm.create("minimum"), sigma=0.3)
# sim_matrix = similarity_func(X)

# # Create ITFRS model with product tnorm and gaines implicator
# tnorm = TNorm.create("product")
# implicator = Implicator.create("gaines")

# logger = get_logger()

# model = ITFRS(similarity_matrix=sim_matrix,
#               labels=labels,
#               tnorm=tnorm,
#               implicator=implicator,
#               logger=logger)

# lower = model.lower_approximation()
# upper = model.upper_approximation()

# print("Similarity matrix (Gaussian + Min):\n", sim_matrix)
# print("\nLower Approximation:", lower)
# print("Upper Approximation:", upper)

print("Done")
