# frutil/models/owafrs.py
"""
OWAFRS implementation.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import owa_weights as owa_weights
from approximations import FuzzyRoughModel
import numpy as np

class OWAFRS(FuzzyRoughModel):
    def __init__(self, similarity_matrix: np.ndarray,
                 labels: np.ndarray,
                 tnorm,
                 implicator,
                 lower_app_weights_method : str,
                 upper_app_weights_method : str):
        super().__init__(similarity_matrix, labels)
        self.tnorm = tnorm
        self.implicator = np.vectorize(implicator)

        # sum_vals = np.sum(lower_approximation_weights) + np.sum(upper_approximation_weights)
        # assert np.isclose(sum_vals, 2.0), "lower & upper weights, each must sum up to 1.0"
        # assert not np.all((lower_approximation_weights >= 0) & (lower_approximation_weights <= 1))
        # assert not np.all((upper_approximation_weights >= 0) & (upper_approximation_weights <= 1))
        # assert not len(lower_approximation_weights) == len(upper_approximation_weights)
        
        if upper_app_weights_method not in ['sup_weights_linear']:
            raise ValueError(f"Unsupported weight type: {upper_app_weights_method}")
        if lower_app_weights_method not in ['inf_weights_linear']:
            raise ValueError(f"Unsupported weight type: {lower_app_weights_method}")
        
        n = len(labels)
        # We generate one less element regarding weights, because in calculations of
        # OWA, the same instance will be excluded from calculations. Therefore, 
        # lower and upper approximations have slightly higher values which are 
        # more realistic
        if(upper_app_weights_method == 'sup_weights_linear'):
            self.upper_approximation_weights = owa_weights._owa_suprimum_weights_linear(n - 1)
        if(lower_app_weights_method == 'inf_weights_linear'):
            self.lower_approximation_weights = owa_weights._owa_infimum_weights_linear(n - 1)

        

    def lower_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        if (__debug__):
            print(label_mask)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        
        # Since for the calculations of lower approximation, 
        # we use OWA operator which is a essentailly a product, 
        # to exclude the same instance from calculations we set
        # the diagonal to 0.0 which
        # is ignored by the product operator.

        np.fill_diagonal(implication_vals, 0.0)
        sorted_matrix = np.sort(implication_vals, axis=1)
        sorted_matrix = sorted_matrix[:, ::-1]
        sorted_matrix = sorted_matrix[:, :-1]

        result = np.matmul(sorted_matrix, self.lower_approximation_weights)
        return result

    def upper_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        if (__debug__):
            print(label_mask)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        
        # Since for the calculations of upper approximation, 
        # we use OWA operator which is a essentailly a product, 
        # to exclude the same instance from calculations we set
        # the diagonal to 0.0 which
        # is ignored by the product operator.

        np.fill_diagonal(implication_vals, 0.0)
        sorted_matrix = np.sort(implication_vals, axis=1)
        sorted_matrix = sorted_matrix[:, ::-1]
        sorted_matrix = sorted_matrix[:, :-1]

        result = np.matmul(sorted_matrix, self.upper_approximation_weights)
        return result