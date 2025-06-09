"""
@file approximations.py
@brief Base class for fuzzy rough set approximation models.

Defines the abstract contract that all fuzzy rough models must implement,
including lower and upper approximations, boundary and positive regions.

##############################################
# ✅ Summary of Clean Code and Design Patterns
# - Template Method Pattern: abstract lower_approximation() and upper_approximation()
# - SRP: Handles only shape/format validation and contract definition
# - LSP: All subclasses can safely extend this without changing client behavior
# - Fail-Fast Validation: Input validation at init
##############################################
"""

from abc import ABC, abstractmethod
import numpy as np

from FRsutils.utils.constructor_utils.fuzzy_rough_lazy_buildable_mixin import FuzzyRoughModelRegistryMixin

class BaseFuzzyRoughModel(ABC, FuzzyRoughModelRegistryMixin):
    """
    @brief Abstract base class for fuzzy-rough approximation models.

    @param similarity_matrix: Symmetric (n x n) similarity matrix in [0, 1]
    @param labels: Label vector of length n
    """
    
    def __init__(self, similarity_matrix: np.ndarray, labels: np.ndarray):
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2:
            raise ValueError("similarity_matrix must be a 2D NumPy array.")
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square.")
        if not ((0.0 <= similarity_matrix).all() and (similarity_matrix <= 1.0).all()):
            raise ValueError("All similarity values must be in the range [0.0, 1.0].")
        if len(labels) != similarity_matrix.shape[0]:
            raise ValueError("Length of labels must match similarity_matrix size.")

        self.similarity_matrix = similarity_matrix
        self.labels = labels

    @abstractmethod
    def lower_approximation(self) -> np.ndarray:
        """
        @brief Abstract method to compute lower approximation.

        @return: Array of lower approximation values.
        """
        pass

    @abstractmethod
    def upper_approximation(self) -> np.ndarray:
        """
        @brief Abstract method to compute upper approximation.

        @return: Array of upper approximation values.
        """
        pass

    def boundary_region(self) -> np.ndarray:
        """
        @brief Compute the boundary region (upper - lower).

        @return: Difference of upper and lower approximation arrays.
        """
        return self.upper_approximation() - self.lower_approximation()

    def to_dict(self) -> dict:
        """
        @brief Placeholder for serialization logic.

        @return: Raise if not implemented.
        """
        raise NotImplementedError("Subclasses must implement to_dict().")

def __str__(self) -> str:
    return f"{self.__class__.__name__}(n={len(self.labels)})"

def __repr__(self) -> str:
    return self.__str__()

def positive_region(self) -> np.ndarray:
    """
    @brief Return the positive region (same as lower approx).

    @return: Lower approximation values.
    """
    return self.lower_approximation()
