"""
VQRS implementation.
"""

from FRsutils.core.approximations import FuzzyRoughModel
import FRsutils.core.tnorms as tn
import numpy as np
import FRsutils.core.fuzzy_quantifiers as fq


class VQRS(FuzzyRoughModel):
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 alpha_lower: float,
                 betha_lower: float,
                 alpha_upper: float,
                 betha_upper: float):
        super().__init__(similarity_matrix, labels)
        self.alpha_lower = alpha_lower
        self.betha_lower = betha_lower
        self.alpha_upper = alpha_upper
        self.betha_upper = betha_upper

        self.tnorm = tn.MinTNorm()

    def _interim_calculations(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)

        # Since for the calculations of lower and upper approximation in VQRS, 
        # we use a sum operator, to exclude the same instance from calculations we set to 0.0 which
        # is ignored by sum operator. To be sure all is correct,
        # inside code, we set main diagonal to 0.0
        np.fill_diagonal(tnorm_vals, 0.0)
        nominator =  np.sum(tnorm_vals, axis=1)

        temp_similarity_matrix = np.copy(self.similarity_matrix)
        np.fill_diagonal(temp_similarity_matrix, 0.0)
        denominator =  np.sum(temp_similarity_matrix, axis=1)

        result = nominator / denominator
        return result

    def lower_approximation(self):
        result = self._interim_calculations()
        result = fq.fuzzy_quantifier_quad(result,
                                          self.alpha_lower,
                                          self.betha_lower)

        return result

    def upper_approximation(self):
        result = self._interim_calculations()
        result = fq.fuzzy_quantifier_quad(result,
                                          self.alpha_upper,
                                          self.betha_upper)

        return result

        