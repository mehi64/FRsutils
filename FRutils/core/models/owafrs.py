# Example fuzzy rough model (simplified): frutil/models/owafrs.py
"""
OWAFRS implementation.
"""
from ..approximations import FuzzyRoughModel
import numpy as np

class OWAFRS(FuzzyRoughModel):
    def lower_approximation(self):
        return np.min(self.similarity_matrix, axis=1)

    def upper_approximation(self):
        return np.max(self.similarity_matrix, axis=1)