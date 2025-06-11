
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

from FRsutils.core.base_fuzzy_rough_model import BaseFuzzyRoughModel
import FRsutils.core.tnorms as tn
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
import numpy as np


@BaseFuzzyRoughModel.register("vqrs")
class VQRS(BaseFuzzyRoughModel):
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
                 alpha_lower: float,
                 beta_lower: float,
                 alpha_upper: float,
                 beta_upper: float,
                 fuzzy_quantifier: str = "quadratic"):
        super().__init__(similarity_matrix, labels)
        
        if alpha_lower > beta_lower or alpha_upper > beta_upper:
            raise ValueError("Alpha must be less than or equal to beta for both bounds.")

        self.alpha_lower = alpha_lower
        self.beta_lower = beta_lower
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper
        self.fuzzy_quantifier_name = fuzzy_quantifier.lower()

        self.tnorm = tn.MinTNorm()
        self.fq_lower = FuzzyQuantifier.create(self.fuzzy_quantifier_name, alpha=alpha_lower, beta=beta_lower)
        self.fq_upper = FuzzyQuantifier.create(self.fuzzy_quantifier_name, alpha=alpha_upper, beta=beta_upper)

    def _interim_calculations(self) -> np.ndarray:
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        numerator = np.sum(tnorm_vals, axis=1)
        denominator = np.sum(self.similarity_matrix, axis=1) - 1.0
        return numerator / denominator

    def lower_approximation(self) -> np.ndarray:
        return self.fq_lower(self._interim_calculations())

    def upper_approximation(self) -> np.ndarray:
        return self.fq_upper(self._interim_calculations())

    def to_dict(self) -> dict:
        return {
            "type": "vqrs",
            "alpha_lower": self.alpha_lower,
            "beta_lower": self.beta_lower,
            "alpha_upper": self.alpha_upper,
            "beta_upper": self.beta_upper,
            "fuzzy_quantifier": self.fuzzy_quantifier_name
        }

    @classmethod
    def from_dict(cls, similarity_matrix, labels, data: dict) -> 'VQRS':
        return cls(similarity_matrix, labels,
                   data["alpha_lower"], data["beta_lower"],
                   data["alpha_upper"], data["beta_upper"],
                   data.get("fuzzy_quantifier", "quadratic"))

    def describe_params_detailed(self) -> dict:
        return {
            "alpha_lower": {"type": "float", "value": self.alpha_lower},
            "beta_lower": {"type": "float", "value": self.beta_lower},
            "alpha_upper": {"type": "float", "value": self.alpha_upper},
            "beta_upper": {"type": "float", "value": self.beta_upper},
            "fuzzy_quantifier": {"type": "str", "value": self.fuzzy_quantifier_name}
        }

    @classmethod
    def validate_params(cls, **kwargs):
        for key in ["alpha_lower", "beta_lower", "alpha_upper", "beta_upper"]:
            val = kwargs.get(key)
            if val is None or not isinstance(val, (float, int)):
                raise ValueError(f"Missing or invalid parameter: {key}")
        fq_name = kwargs.get("fuzzy_quantifier", "quadratic")
        FuzzyQuantifier.validate_params(alpha=kwargs["alpha_lower"], beta=kwargs["beta_lower"])
        FuzzyQuantifier.validate_params(alpha=kwargs["alpha_upper"], beta=kwargs["beta_upper"])
