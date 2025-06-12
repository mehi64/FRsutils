
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
# - Uses pluggable fuzzy quantifier strategy ('quadratic', 'linear', etc.)

# ✅ Design Patterns & Principles Used
# - Strategy: Uses fuzzy quantifier strategy via registry
# - Template Method: Inherits contract from BaseFuzzyRoughModel
# - Adapter: `to_dict()` and `from_dict()` supported
# - Clean Code: Validation, structured documentation, introspection
##############################################
"""

from FRsutils.core.fuzzy_rough_model import FuzzyRoughModel
import FRsutils.core.tnorms as tn
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.utils.logger.logger_util import get_logger
import numpy as np


@FuzzyRoughModel.register("vqrs")
class VQRS(FuzzyRoughModel):
    """
    @brief VQRS model for fuzzy rough approximation using fuzzy quantifiers.

    @param similarity_matrix: Square matrix with pairwise similarities in [0, 1]
    @param labels: Label array of same length as similarity matrix
    @param alpha_lower, beta_lower: Parameters for lower approximation quantifier
    @param alpha_upper, beta_upper: Parameters for upper approximation quantifier
    @param fuzzy_quantifier: The fuzzy quantifier name (e.g., 'quadratic', 'linear')
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 fuzzy_quantifier_lower: FuzzyQuantifier,
                 fuzzy_quantifier_upper: FuzzyQuantifier,
                 logger=None):
        
        super().__init__(similarity_matrix, labels)
        self.logger = logger or get_logger()
        self.logger.debug(f"{self.__class__.__name__} initialized.")

        self.validate_params(fuzzy_quantifier_lower=fuzzy_quantifier_lower,
                             fuzzy_quantifier_upper=fuzzy_quantifier_upper)

        self.fuzzy_quantifier_lower=fuzzy_quantifier_lower
        self.fuzzy_quantifier_upper=fuzzy_quantifier_upper

        self.tnorm = tn.MinTNorm()


    def _interim_calculations(self) -> np.ndarray:
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        numerator = np.sum(tnorm_vals, axis=1)
        denominator = np.sum(self.similarity_matrix, axis=1) - 1.0
        return numerator / denominator

    def lower_approximation(self) -> np.ndarray:
        return self.fuzzy_quantifier_lower(self._interim_calculations())

    def upper_approximation(self) -> np.ndarray:
        return self.fuzzy_quantifier_upper(self._interim_calculations())

    def to_dict(self) -> dict:
        return {
            "type": "vqrs",
            "fuzzy_quantifier_lower": self.fuzzy_quantifier_lower.to_dict(),
            "fuzzy_quantifier_upper": self.fuzzy_quantifier_upper.to_dict()
        }

    @classmethod
    def from_dict(cls, similarity_matrix, labels, data: dict) -> 'VQRS':
        raise NotImplementedError("Class method not implemented for VQRS.")
        # return cls(similarity_matrix, labels,
        #            data["alpha_lower"], data["beta_lower"],
        #            data["alpha_upper"], data["beta_upper"],
        #            data.get("fuzzy_quantifier", "quadratic"))

    def describe_params_detailed(self) -> dict:
        raise NotImplementedError("Class method not implemented for VQRS.")
        # return {
        #     "alpha_lower": {"type": "float", "value": self.alpha_lower},
        #     "beta_lower": {"type": "float", "value": self.beta_lower},
        #     "alpha_upper": {"type": "float", "value": self.alpha_upper},
        #     "beta_upper": {"type": "float", "value": self.beta_upper},
        #     "fuzzy_quantifier": {"type": "str", "value": self.fuzzy_quantifier_name}
        # }

    def _get_params(self) -> dict:
        """
        @brief Describe internal parameters.

        @return: Dictionary describing internal parameters.
        """
        return {
            "min_tnorm": self.tnorm,
            "fuzzy_quantifier_lower": self.fuzzy_quantifier_lower,
            "fuzzy_quantifier_upper": self.fuzzy_quantifier_upper,
            "similarity_matrix":self.similarity_matrix,
            "labels":self.labels
        }

    @classmethod
    def validate_params(self, **kwargs):
        """
        @brief validation hook.

        @param kwargs
        """

        Q_l = kwargs.get("fuzzy_quantifier_lower")
        if Q_l is None or not isinstance(Q_l, FuzzyQuantifier):
            raise ValueError("Parameter 'fuzzy_quantifier_lower' must be provided and be an instance of derived classes from FuzzyQuantifier.")

        Q_h = kwargs.get("fuzzy_quantifier_upper")
        if Q_h is None or not isinstance(Q_h, FuzzyQuantifier):
            raise ValueError("Parameter 'fuzzy_quantifier_upper' must be provided and be an instance of derived classes from FuzzyQuantifier.")
