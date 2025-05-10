# frutil/models/itfrs.py
"""
ITFRS implementation.
"""
from ..approximations import FuzzyRoughModel
from ..tnorms import tn_product
from ..implicators import imp_goedel
import numpy as np

class ITFRS(FuzzyRoughModel):
    def __init__(self, similarity_matrix: np.ndarray, labels: np.ndarray, tnorm, implicator):
        super().__init__(similarity_matrix, labels)
        self.tnorm = tnorm
        self.implicator = np.vectorize(implicator)

    def lower_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        return np.min(implication_vals, axis=1)

    def upper_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(np.stack([self.similarity_matrix, label_mask], axis=-1))
        return np.max(tnorm_vals, axis=1)