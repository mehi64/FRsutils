"""
@file owafrs.py
@brief OWAFRS (Ordered Weighted Averaging Fuzzy Rough Set) model implementation.

Extends fuzzy rough approximation using OWA weights for more flexible decision regions.

##############################################
# ✅ Quick Summary of Features
# - OWA-based lower and upper approximations
# - Linear weighting strategy for aggregation
# - Support for vectorized similarity and label matrices
# - Pluggable architecture for T-norms and implicators

# ✅ Design Patterns & Principles Used
# - Strategy: Delegates to provided T-norm and Implicator strategies
# - Template Method: Extends abstract base for fuzzy rough models
# - Adapter: `to_dict()` / `from_dict()` for serialization
# - Clean Code: SRP, fail-fast checks, structured docs, LSP
##############################################
"""

import FRsutils.core.tnorms as tn
import FRsutils.core.owa_weights as owa_weights
from FRsutils.core.approximations import BaseFuzzyRoughModel
import FRsutils.core.implicators as imp
from FRsutils.utils.logger.logger_util import get_logger
import numpy as np


@BaseFuzzyRoughModel.register("owafrs")
class OWAFRS(BaseFuzzyRoughModel):
    """
    @brief Ordered Weighted Averaging Fuzzy Rough Sets (OWAFRS) approximation model.
    """
    def __init__(self,
                 similarity_matrix: np.ndarray,
                 labels: np.ndarray,
                 tnorm: tn.TNorm,
                 implicator: imp.Implicator,
                 lower_app_weights_method: str,
                 upper_app_weights_method: str, logger=None):
        super().__init__(similarity_matrix, labels)
        self.logger = logger or get_logger()
        self.logger.debug(f"{self.__class__.__name__} initialized.")
        self.tnorm = tnorm
        self.implicator = implicator

        n = len(labels)
        if upper_app_weights_method not in ['sup_weights_linear']:
            raise ValueError(f"Unsupported upper weight method: {upper_app_weights_method}")
        if lower_app_weights_method not in ['inf_weights_linear']:
            raise ValueError(f"Unsupported lower weight method: {lower_app_weights_method}")

        self.upper_approximation_weights = owa_weights.owa_suprimum_weights_linear(n - 1)
        self.lower_approximation_weights = owa_weights.owa_infimum_weights_linear(n - 1)

    def lower_approximation(self) -> np.ndarray:
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        np.fill_diagonal(implication_vals, 0.0)
        sorted_matrix = np.sort(implication_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.lower_approximation_weights)

    def upper_approximation(self) -> np.ndarray:
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        sorted_matrix = np.sort(tnorm_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.upper_approximation_weights)

    def to_dict(self) -> dict:
        return {
            "type": "owafrs",
            "tnorm": self.tnorm.to_dict(),
            "implicator": self.implicator.to_dict(),
            "lower_app_weights_method": "inf_weights_linear",
            "upper_app_weights_method": "sup_weights_linear"
        }

    @classmethod
    def from_dict(cls, similarity_matrix, labels, data: dict) -> 'OWAFRS':
        tnorm = tn.TNorm.from_dict(data["tnorm"])
        implicator = imp.Implicator.from_dict(data["implicator"])
        return cls(similarity_matrix, labels, tnorm, implicator,
                   lower_app_weights_method=data["lower_app_weights_method"],
                   upper_app_weights_method=data["upper_app_weights_method"])

    def describe_params_detailed(self) -> dict:
        return {
            "tnorm": self.tnorm.describe_params_detailed(),
            "implicator": self.implicator.describe_params_detailed(),
            "lower_app_weights_method": {"type": "str", "value": "inf_weights_linear"},
            "upper_app_weights_method": {"type": "str", "value": "sup_weights_linear"}
        }

    @classmethod
    def validate_params(cls, **kwargs):
        # Placeholder for future strict validation
        pass
