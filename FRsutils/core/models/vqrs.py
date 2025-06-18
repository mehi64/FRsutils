"""
@file vqrs.py
@brief VQRS (Variable-precision Quantitative Rough Set) model implementation.

Supports both direct construction and lazy instantiation via config or dictionary.

##############################################
# ✅ Summary of Clean Code and Design Patterns
# - Strategy Pattern: Fuzzy quantifier is configurable
# - Adapter Pattern: Implements to_dict/from_dict serialization
# - Template Method: Inherits from FuzzyRoughModel abstract interface
# - Logger Injection: Supports injected or default logger
# - Fail-Fast Validation: Ensures correct quantifier types
##############################################
"""

import numpy as np
import FRsutils.core.tnorms as tn
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
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
        super().__init__(similarity_matrix, 
                         labels, 
                         logger=logger)
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

    def to_dict(self, include_data: bool = False) -> dict:
        data = {
            "type": "vqrs",
            "fuzzy_quantifier_lower": self.fuzzy_quantifier_lower.to_dict(),
            "fuzzy_quantifier_upper": self.fuzzy_quantifier_upper.to_dict()
        }
        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "VQRS":
        fq_lower = FuzzyQuantifier.from_dict(data["fuzzy_quantifier_lower"])
        fq_upper = FuzzyQuantifier.from_dict(data["fuzzy_quantifier_upper"])

        sim = similarity_matrix if similarity_matrix is not None else (np.array(data["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(data["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either as arguments or in the data dictionary.")

        return cls(sim, lbl, fq_lower, fq_upper, logger=logger)

    def _get_params(self) -> dict:
        return {
            "min_tnorm": self.tnorm,
            "fuzzy_quantifier_lower": self.fuzzy_quantifier_lower,
            "fuzzy_quantifier_upper": self.fuzzy_quantifier_upper,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels
        }

    @classmethod
    def from_config(cls, config: dict, similarity_matrix=None, labels=None, logger=None) -> "VQRS":
        fq_type = config.get("fuzzy_quantifier_lower")["type"]
        fq_cls = FuzzyQuantifier.get_class(fq_type)
        fq_lower = fq_cls(alpha=config["fuzzy_quantifier_lower"]["alpha"], beta=config["fuzzy_quantifier_lower"]["beta"])
        fq_upper = fq_cls(alpha=config["fuzzy_quantifier_upper"]["alpha"], beta=config["fuzzy_quantifier_upper"]["beta"])

        # Handle matrix and labels
        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in config or as arguments.")

        return cls(sim, lbl, fq_lower, fq_upper, logger=logger)
    
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
