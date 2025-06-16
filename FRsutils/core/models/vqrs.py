"""
@file vqrs.py
@brief VQRS (Variable-precision Quantitative Rough Set) model implementation.

Supports both direct construction and lazy instantiation via config.
"""

import numpy as np
import FRsutils.core.tnorms as tn
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.utils.logger.logger_util import get_logger
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel


@FuzzyRoughModel.register("vqrs")
class VQRS(FuzzyRoughModel):
    """
    @brief VQRS model for fuzzy rough approximation using fuzzy quantifiers.

    @param similarity_matrix: Pairwise similarity matrix (n x n)
    @param labels: Corresponding label vector (n,)
    @param fuzzy_quantifier_lower: FuzzyQuantifier instance for lower approx
    @param fuzzy_quantifier_upper: FuzzyQuantifier instance for upper approx
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

        self.validate_params(
            fuzzy_quantifier_lower=fuzzy_quantifier_lower,
            fuzzy_quantifier_upper=fuzzy_quantifier_upper
        )

        self.fuzzy_quantifier_lower = fuzzy_quantifier_lower
        self.fuzzy_quantifier_upper = fuzzy_quantifier_upper
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

    def _get_params(self) -> dict:
        return {
            "min_tnorm": self.tnorm,
            "fuzzy_quantifier_lower": self.fuzzy_quantifier_lower,
            "fuzzy_quantifier_upper": self.fuzzy_quantifier_upper,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels
        }

    @classmethod
    def from_config(cls,
                    similarity_matrix,
                    labels,
                    alpha_lower: float,
                    beta_lower: float,
                    alpha_upper: float,
                    beta_upper: float,
                    fuzzy_quantifier: str = "quadratic",
                    logger=None,
                    **kwargs):
        """
        @brief Creates a VQRS model from parameterized config.

        @param similarity_matrix: n x n similarity matrix
        @param labels: length-n class vector
        @param alpha_lower, beta_lower: Lower approx quantifier params
        @param alpha_upper, beta_upper: Upper approx quantifier params
        @param fuzzy_quantifier: 'quadratic', 'linear', etc.
        @return: VQRS instance
        """
        fq_cls = FuzzyQuantifier.get_class(fuzzy_quantifier)
        fq_lower = fq_cls(alpha=alpha_lower, beta=beta_lower)
        fq_upper = fq_cls(alpha=alpha_upper, beta=beta_upper)
        return cls(similarity_matrix, labels, fq_lower, fq_upper, logger=logger)

    @classmethod
    def validate_params(cls, **kwargs):
        fq_l = kwargs.get("fuzzy_quantifier_lower")
        fq_u = kwargs.get("fuzzy_quantifier_upper")
        if fq_l is None or not isinstance(fq_l, FuzzyQuantifier):
            raise ValueError("fuzzy_quantifier_lower must be a valid FuzzyQuantifier instance.")
        if fq_u is None or not isinstance(fq_u, FuzzyQuantifier):
            raise ValueError("fuzzy_quantifier_upper must be a valid FuzzyQuantifier instance.")

    def describe_params_detailed(self) -> dict:
        return {
            "fuzzy_quantifier_lower": self.fuzzy_quantifier_lower.describe_params_detailed(),
            "fuzzy_quantifier_upper": self.fuzzy_quantifier_upper.describe_params_detailed()
        }
