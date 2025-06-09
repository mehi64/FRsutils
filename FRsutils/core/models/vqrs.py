"""
@file vqrs.py
@brief VQRS (Variable-precision Quantitative Rough Set) model implementation.

Uses fuzzy quantifiers to define soft decision regions based on similarity matrix.

##############################################
# ✅ Quick Summary of Features
# - VQRS approximation model using fuzzy quantifier bounds
# - Parameterized with alpha and beta for flexibility
# - Fixed T-norm (Min) for all calculations
# - Compatible with synthetic testing datasets

# ✅ Design Patterns & Principles Used
# - Strategy: Uses fuzzy quantifier function for Q
# - Template Method: Inherits contract from BaseFuzzyRoughModel
# - Adapter: `to_dict()` and `from_dict()` supported
# - Clean Code: Validation, structured documentation, introspection
##############################################
"""

from FRsutils.core.approximations import BaseFuzzyRoughModel
import FRsutils.core.tnorms as tn
import FRsutils.core.fuzzy_quantifiers as fq
from FRsutils.utils.logger.logger_util import get_logger
import numpy as np


@BaseFuzzyRoughModel.register("vqrs")
class VQRS(BaseFuzzyRoughModel):
    """
    @brief VQRS model for fuzzy rough approximation using fuzzy quantifiers.

    @param similarity_matrix: Square matrix with pairwise similarities in [0, 1]
    @param labels: Label array of same length as similarity matrix
    @param alpha_lower, beta_lower: Parameters for lower approximation quantifier
    @param alpha_upper, beta_upper: Parameters for upper approximation quantifier
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 alpha_lower: float,
                 beta_lower: float,
                 alpha_upper: float,
                 beta_upper: float, logger=None):
        super().__init__(similarity_matrix, labels)
        self.logger = logger or get_logger()
        self.logger.debug(f"{self.__class__.__name__} initialized.")
        
        if alpha_lower > beta_lower or alpha_upper > beta_upper:
            raise ValueError("Alpha must be less than or equal to beta for both bounds.")

        self.alpha_lower = alpha_lower
        self.beta_lower = beta_lower
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper

        self.tnorm = tn.MinTNorm()

    def _interim_calculations(self) -> np.ndarray:
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        numerator = np.sum(tnorm_vals, axis=1)
        denominator = np.sum(self.similarity_matrix, axis=1) - 1.0
        return numerator / denominator

    def lower_approximation(self) -> np.ndarray:
        return fq.fuzzy_quantifier_quadratic(self._interim_calculations(),
                                             self.alpha_lower, self.beta_lower)

    def upper_approximation(self) -> np.ndarray:
        return fq.fuzzy_quantifier_quadratic(self._interim_calculations(),
                                             self.alpha_upper, self.beta_upper)

    def to_dict(self) -> dict:
        return {
            "type": "vqrs",
            "alpha_lower": self.alpha_lower,
            "beta_lower": self.beta_lower,
            "alpha_upper": self.alpha_upper,
            "beta_upper": self.beta_upper
        }

    @classmethod
    def from_dict(cls, similarity_matrix, labels, data: dict) -> 'VQRS':
        return cls(similarity_matrix, labels,
                   data["alpha_lower"], data["beta_lower"],
                   data["alpha_upper"], data["beta_upper"])

    def describe_params_detailed(self) -> dict:
        return {
            "alpha_lower": {"type": "float", "value": self.alpha_lower},
            "beta_lower": {"type": "float", "value": self.beta_lower},
            "alpha_upper": {"type": "float", "value": self.alpha_upper},
            "beta_upper": {"type": "float", "value": self.beta_upper}
        }

    @classmethod
    def validate_params(cls, **kwargs):
        for key in ["alpha_lower", "beta_lower", "alpha_upper", "beta_upper"]:
            val = kwargs.get(key)
            if val is None or not isinstance(val, (float, int)):
                raise ValueError(f"Missing or invalid parameter: {key}")
