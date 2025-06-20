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
# - Template Method: Abstract interface for lower_weights / upper_weights
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

    Subclasses must implement `lower_weights(n)` and `upper_weights(n)`.
    """

    @abstractmethod
    def lower_weights(self, n: int) -> np.ndarray:
        """
        @brief Computes weights in ascending order for infimum OWA.

        @param n: Number of weights to generate.
        @return: Normalized weight vector (ascending).
        """
        pass

    @abstractmethod
    def upper_weights(self, n: int) -> np.ndarray:
        """
        @brief Computes weights in descending order for supremum OWA.

        @param n: Number of weights to generate.
        @return: Normalized weight vector (descending).
        """
        pass

    def weights(self, n: int, descending: bool = False) -> np.ndarray:
        """
        @brief Unified method to retrieve OWA weights.

        Calls `lower_weights` when `descending=False` (default),
        and `upper_weights` when `descending=True`.

        @param n: Number of weights to compute.
        @param descending: If True, returns upper_weights; otherwise lower_weights.
        @return: Normalized OWA weight vector.
        
        @example
        >>> w = OWAWeightStrategy.create("linear")
        >>> w.weights(4)  # equivalent to w.lower_weights(4)
        >>> w.weights(4, descending=True)  # equivalent to w.upper_weights(4)
        """
        return self.upper_weights(n) if descending else self.lower_weights(n)



@OWAWeightStrategy.register("linear")
class LinearOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Linear OWA weighting:
    - Lower: wᵢ = 2i / (n(n+1))
    - Upper: wᵢ = 2(n−i+1) / (n(n+1))
    """

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(np.arange(1, n + 1))

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(np.arange(n, 0, -1))

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0)
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass


@OWAWeightStrategy.register("exponential", "exp")
class ExponentialOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Exponential OWA weighting:
    - Param: base > 1
    - Lower: wᵢ ∝ base^i
    - Upper: wᵢ ∝ base^(n−i+1)
    """

    def __init__(self, base: float = 2.0):
        self.validate_params(base=base)
        self.base = base

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(self.base ** np.arange(1, n + 1))

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(self.base ** np.arange(n, 0, -1))

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0)
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

    def _get_params(self) -> dict:
        return {"base": self.base}

    @classmethod
    def validate_params(cls, **kwargs):
        base = kwargs.get("base")
        if base is None or not isinstance(base, (int, float)) or base <= 1:
            raise ValueError("Parameter 'base' must be > 1")


@OWAWeightStrategy.register("harmonic", "harm")
class HarmonicOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Harmonic OWA weighting:
    - Lower: wᵢ ∝ 1 / i
    - Upper: wᵢ ∝ 1 / (n − i + 1)
    """

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(1.0 / np.arange(1, n + 1))

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(1.0 / np.arange(n, 0, -1))

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0)
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass


@OWAWeightStrategy.register("logarithmic", "log")
class LogarithmicOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Logarithmic OWA weighting:
    - Lower: wᵢ ∝ log(i + 1)
    - Upper: wᵢ ∝ log(n − i + 2)
    """

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(np.log(np.arange(1, n + 1) + 1.0))

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        return self._normalize(np.log(np.arange(n, 0, -1) + 1.0))

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0)
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass
