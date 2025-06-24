"""
@file owa_weights.py
@brief OWA Weighting Strategies Framework

Provides a pluggable, serializable, and validated system for defining OWA weight generation strategies.
Supports dynamic registration, factory-based instantiation, and separate implementations for lower/upper weights.

##############################################
# ✅ Summary of Clean Code and Design Patterns
# - Registry Pattern: Subclass registration via @register(...)
# - Factory Method: Strategy.create(name, **kwargs)
# - Adapter Pattern: Serialization via to_dict / from_dict
# - Strategy Pattern: Linear, Harmonic, Exponential, etc. each define a behavior
# - Template Method: Abstract interface for raw weight generation
# - Fail-Fast Validation: Param checks in validate_params
##############################################

@example
>>> w = OWAWeightStrategy.create("linear")
>>> w.lower_weights(5)
array([0.06666667, 0.13333333, 0.2, 0.26666667, 0.33333333])
"""

import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin


class OWAWeightStrategy(RegistryFactoryMixin):
    """
    @brief Abstract base class for OWA weight strategies.

    Subclasses must implement `_raw_weights(n)` that generates unnormalized weights.
    This class handles normalization and sorting.
    """

    def lower_weights(self, n: int) -> np.ndarray:
        """
        @brief Returns ascending (infimum) OWA weights.

        @param n: Number of weights
        @return: Normalized weight vector in ascending order
        """
        self._validate_n(n)
        return self._normalize(self._raw_weights(n), order='asc')

    def upper_weights(self, n: int) -> np.ndarray:
        """
        @brief Returns descending (supremum) OWA weights.

        @param n: Number of weights
        @return: Normalized weight vector in descending order
        """
        self._validate_n(n)
        return self._normalize(self._raw_weights(n), order='desc')

    def weights(self, n: int, order: str = 'asc') -> np.ndarray:
        """
        @brief Unified method to retrieve OWA weights in specified order.

        @param n: Number of weights to compute.
        @param order: 'asc' for increasing weights, 'desc' for decreasing weights.
        @return: Normalized OWA weight vector.
        """
        if order == 'asc':
            return self.lower_weights(n)
        elif order == 'desc':
            return self.upper_weights(n)
        else:
            raise ValueError("order must be 'asc' or 'desc'")

    @abstractmethod
    def _raw_weights(self, n: int) -> np.ndarray:
        """
        @brief Generate unnormalized, unsorted weight values.

        @param n: Number of weights
        @return: Raw weights (to be normalized and sorted)
        """
        pass

    def _validate_n(self, n: int):
        """
        @brief Validates the number of weights.

        @param n: Number of weights
        @raise ValueError: If n is not a positive integer
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

    def _normalize(self, weights: np.ndarray, order='asc') -> np.ndarray:
        """
        @brief Normalizes weights and sorts them in the requested order.

        @param weights: Unnormalized weights
        @param order: 'asc' or 'desc'
        @return: Normalized and ordered weights
        @raise ValueError: If weights are not valid or order is invalid
        """
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        if order == 'asc':
            norm = np.sort(norm)
        elif order == 'desc':
            norm = np.sort(norm)[::-1]
        else:
            raise ValueError("The key `order` must be `asc` or `desc`")
        assert np.isclose(norm.sum(), 1.0)
        return norm


@OWAWeightStrategy.register("linear")
class LinearOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Linear OWA weighting strategy.
    Generates linearly increasing weights.
    """
    def _raw_weights(self, n: int) -> np.ndarray:
        return np.arange(1, n + 1)

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__, "name": "linear", "params": self._get_params()}


@OWAWeightStrategy.register("exponential", "exp")
class ExponentialOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Exponential OWA weighting strategy.
    Generates exponentially increasing weights controlled by a base.
    """
    def __init__(self, base: float = 2.0):
        self.validate_params(base=base)
        self.base = base

    def _raw_weights(self, n: int) -> np.ndarray:
        return self.base ** np.arange(1, n + 1)

    def _get_params(self) -> dict:
        return {"base": self.base}

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__, "name": "exponential", "params": self._get_params()}

    @classmethod
    def validate_params(cls, **kwargs):
        base = kwargs.get("base")
        if base is None or not isinstance(base, (int, float)) or base <= 1:
            raise ValueError("Parameter 'base' must be > 1")


@OWAWeightStrategy.register("harmonic", "harm")
class HarmonicOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Harmonic OWA weighting strategy.
    Generates weights inversely proportional to index (1/i).
    """
    def _raw_weights(self, n: int) -> np.ndarray:
        return 1.0 / np.arange(1, n + 1)

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__, "name": "harmonic", "params": self._get_params()}


@OWAWeightStrategy.register("logarithmic", "log")
class LogarithmicOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Logarithmic OWA weighting strategy.
    Uses log(i + 1) as raw weights.
    """
    def _raw_weights(self, n: int) -> np.ndarray:
        return np.log(np.arange(1, n + 1) + 1.0)

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__, "name": "logarithmic", "params": self._get_params()}
