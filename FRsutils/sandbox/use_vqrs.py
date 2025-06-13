import numpy as np
from FRsutils.core.similarities import Similarity, calculate_similarity_matrix
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.models.vqrs import VQRS
from FRsutils.utils.logger.logger_util import get_logger
import tests.syntetic_data_for_tests as sdf
from FRsutils.core.fuzzy_rough_model import FuzzyRoughModel as FRMODEL


data_synthteic = sdf.syntetic_dataset_factory()

TSTVQRS = data_synthteic.VQRS_testing_dataset()
sim_matrix = TSTVQRS['sim_matrix']
y = TSTVQRS['y']

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


Q_l = FuzzyQuantifier.create("quadratic", alpha=.1, beta=.6)
Q_u = FuzzyQuantifier.create("quadratic", alpha=.2, beta=1.0)



logger = get_logger()

model = VQRS(similarity_matrix=sim_matrix,
            labels=y,
            fuzzy_quantifier_lower=Q_l,
            fuzzy_quantifier_upper=Q_u,
            logger=logger)

upper = model.upper_approximation()
lower = model.lower_approximation()

args_ = {'similarity_matrix': sim_matrix, 
        'labels': y, 
        'fuzzy_quantifier_lower':Q_l,
        'fuzzy_quantifier_upper':Q_u}

frmodel = FRMODEL.create(name='vqrs', strict=False, **args_)

model_cls = FRMODEL.get_class("vqrs")
vqrs_model = model_cls.from_config(similarity_matrix=sim_matrix,
                                   labels=y,
                                    alpha_lower=0.1, beta_lower=0.6,
                                    alpha_upper=0.2, beta_upper=1.0,
                                    fuzzy_quantifier="quadratic")

upper1 = frmodel.upper_approximation()
lower1 = frmodel.lower_approximation()

upper2 = vqrs_model.upper_approximation()
lower2 = vqrs_model.lower_approximation()

# print("tnorm:", tnorm.name)
# print("implicator:", implicator.name)
print("Lower Approximation:", lower2)
print("Lower Approximation:", lower1)
print("Upper Approximation:", upper2)
print("Upper Approximation:", upper1)

print("Done")
