"""
@file itfrs.py
@brief Implementation of the IT2 Fuzzy Rough Set (ITFRS) approximation model.

Provides a concrete implementation of the lower and upper approximations
using a fuzzy implicator and a T-norm operator over a similarity matrix.

##############################################
# ✅ Quick Summary of Features
# - ITFRS model for fuzzy rough approximation
# - Pluggable architecture for T-norm and Implicator
# - Lower and upper approximation computation
# - Class introspection and serialization support

# ✅ Summary Table of Design Principles
# - Strategy Pattern: Uses user-defined T-norm and Implicator strategies
# - Template Method: Inherits abstract methods from BaseFuzzyRoughModel
# - Adapter Pattern: Provides to_dict/from_dict for serialization
# - Clean Code: SRP, DRY, LSP, docstring documentation, fail-fast checks
##############################################
"""

from FRsutils.utils.constructor_utils.lazy_builder_from_config_mixin import LazyBuildableFromConfigMixin
from FRsutils.core.fuzzy_rough_model import FuzzyRoughModel
import FRsutils.core.tnorms as tn
import FRsutils.core.implicators as imp
from FRsutils.utils.logger.logger_util import get_logger
import numpy as np


@FuzzyRoughModel.register("itfrs")
class ITFRS(FuzzyRoughModel, LazyBuildableFromConfigMixin):
    """
    @brief Interval Type-2 Fuzzy Rough Set approximation model.

    @param similarity_matrix: Precomputed similarity matrix (n x n)
    @param labels: Array of class labels for each instance
    @param tnorm: T-norm operator (object from TNorm)
    @param implicator: Fuzzy implicator operator (object from Implicator)
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 tnorm: tn.TNorm, 
                 implicator: imp.Implicator,
                 logger=None):
        super().__init__(similarity_matrix, labels)
        self.logger = logger or get_logger()
        self.logger.debug(f"{self.__class__.__name__} initialized.")

        self.validate_params(tnorm=tnorm, 
                             implicator=implicator)

        self.tnorm = tnorm
        self.implicator = implicator

    def lower_approximation(self) -> np.ndarray:
        """
        @brief Compute the lower approximation using the implicator.

        @return: Lower approximation array (n,)
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        np.fill_diagonal(implication_vals, 1.0)
        return np.min(implication_vals, axis=1)

    def upper_approximation(self) -> np.ndarray:
        """
        @brief Compute the upper approximation using the T-norm.

        @return: Upper approximation array (n,)
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        return np.max(tnorm_vals, axis=1)

    def to_dict(self) -> dict:
        """
        @brief Serialize the ITFRS instance to dictionary.

        @return: Serializable dict including type and operator info.
        """
        return {
            "type": "itfrs",
            "tnorm": self.tnorm.to_dict(),
            "implicator": self.implicator.to_dict()
        }

    @classmethod
    def from_dict(cls, similarity_matrix, labels, data: dict) -> 'ITFRS':
        """
        @brief Reconstruct ITFRS model from serialized dictionary.

        @param similarity_matrix: Matrix used for similarity
        @param labels: Class label vector
        @param data: Serialized dictionary
        @return: ITFRS instance
        """
        tnorm = tn.TNorm.from_dict(data["tnorm"])
        implicator = imp.Implicator.from_dict(data["implicator"])
        return cls(similarity_matrix, labels, tnorm, implicator)

    def describe_params_detailed(self) -> dict:
        """
        @brief Describe internal T-norm and implicator parameters.

        @return: Dictionary describing parameters of components.
        """
        return {
            "tnorm": self.tnorm.describe_params_detailed(),
            "implicator": self.implicator.get_params_detailed()
        }
    
    def _get_params(self) -> dict:
        """
        @brief Describe internal T-norm and implicator parameters.

        @return: Dictionary containing T-norm and implicator used in itfrs.
        """
        return {
            "tnorm": self.tnorm,
            "implicator": self.implicator,
            "similarity_matrix":self.similarity_matrix,
            "labels":self.labels
        }

    @classmethod
    def validate_params(self, **kwargs):
        """
        @brief validation hook.

        @param kwargs
        """
        
        tnrm = kwargs.get("tnorm")
        if tnrm is None or not isinstance(tnrm, tn.TNorm):
            raise ValueError("Parameter 'tnorm' must be provided and be an instance of derived classes from TNorm.")

        impli = kwargs.get("implicator")
        if impli is None or not isinstance(impli, imp.Implicator):
            raise ValueError("Parameter 'implicator' must be provided and be an instance of derived classes from Implicator.")

    @classmethod
    def from_config(cls, similarity_matrix, labels, tnorm_name: str, implicator_name: str, 
                    tnorm_params=None, implicator_params=None, logger=None):
        """
        @brief Alternate constructor that creates an ITFRS model from registry names.

        @param similarity_matrix: Similarity matrix
        @param labels: Labels
        @param tnorm_name: Registered T-norm name
        @param implicator_name: Registered implicator name
        @param tnorm_params: Optional dict of T-norm parameters
        @param implicator_params: Optional dict of implicator parameters
        @param logger: Optional logger
        @return: ITFRS instance
        """
        tnorm = tn.TNorm.create(tnorm_name, **(tnorm_params or {}))
        implicator = imp.Implicator.create(implicator_name, **(implicator_params or {}))
        return cls(similarity_matrix, labels, tnorm, implicator, logger=logger)