"""
@file owa_weights.py
@brief Framework for generating OWA weights for fuzzy infimum and suprimum smoothing.

Supports alias-based instantiation, serialization, and custom strategy definitions with
separate lower and upper weight generation methods.
"""

import numpy as np
from abc import abstractmethod
from typing import Type, Dict, List
import inspect


def _filter_args(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class OWAWeightStrategy():
    """
    @brief Abstract base class for OWA weight strategies.

    Each subclass must define lower_weights() and upper_weights().
    """

    _registry: Dict[str, Type['OWAWeightStrategy']] = {}
    _aliases: Dict[Type['OWAWeightStrategy'], List[str]] = {}

    @classmethod
    def register(cls, *names: str):
        def decorator(subclass: Type['OWAWeightStrategy']):
            if not names:
                raise ValueError("Must provide at least one name for registration.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"OWAWeightStrategy alias '{key}' already registered.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, strict: bool = False, **kwargs) -> 'OWAWeightStrategy':
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown OWA strategy alias: {name}")
        strat_cls = cls._registry[name]
        strat_cls.validate_params(**kwargs)
        ctor_args = _filter_args(strat_cls, kwargs)
        if strict:
            unused = set(kwargs) - set(ctor_args)
            if unused:
                raise ValueError(f"Unused parameters: {unused}")
        return strat_cls(**ctor_args)

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        return {names[0]: names for names in cls._aliases.values()}

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__.replace("OWAWeightStrategy", "").lower(),
            **self._get_params()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OWAWeightStrategy':
        data = data.copy()
        name = data.pop("type")
        return cls.create(name, **data)

    def _get_params(self) -> dict:
        return {}

    def help(self) -> str:
        return self.__class__.__doc__.strip() if self.__class__.__doc__ else "No documentation available."

    @abstractmethod
    def lower_weights(self, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def upper_weights(self, n: int) -> np.ndarray:
        pass


@OWAWeightStrategy.register("linear")
class LinearOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Linear OWA weighting strategy:
    - Lower weights: wᵢ = 2i / (n(n+1)) (infimum)
    - Upper weights: wᵢ = 2(n−i+1) / (n(n+1)) (suprimum)
    """

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        weights = np.arange(1, n + 1)
        return self._normalize(weights)

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        weights = np.arange(n, 0, -1)
        return self._normalize(weights)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0), "Weights must sum to 1"
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n <= 0:
            raise ValueError("n must be an integer >= 1")


@OWAWeightStrategy.register("exponential", "exp")
class ExponentialOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Exponential OWA weighting strategy.

    @param base: Exponential base > 1.
    - Lower weights: wᵢ ∝ base^i (i=1 to n)
    - Upper weights: wᵢ ∝ base^(n−i+1)
    """

    def __init__(self, base: float = 2.0):
        self.base = base

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        powers = np.arange(1, n + 1)
        weights = self.base ** powers
        return self._normalize(weights)

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        powers = np.arange(n, 0, -1)
        weights = self.base ** powers
        return self._normalize(weights)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0), "Weights must sum to 1"
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n <= 0:
            raise ValueError("n must be an integer >= 1")

    @classmethod
    def validate_params(cls, **kwargs):
        base = kwargs.get("base")
        if base is None:
            raise ValueError("Missing required parameter: base")
        if not isinstance(base, (int, float)) or base <= 1:
            raise ValueError("Parameter 'base' must be a number > 1")

    def _get_params(self) -> dict:
        return {"base": self.base}

@OWAWeightStrategy.register("harmonic", "harm")
class HarmonicOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Harmonic OWA weighting strategy:
    - Lower weights: wᵢ ∝ 1 / i
    - Upper weights: wᵢ ∝ 1 / (n − i + 1)
    """

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        weights = 1.0 / np.arange(1, n + 1)
        return self._normalize(weights)

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        weights = 1.0 / np.arange(n, 0, -1)
        return self._normalize(weights)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0), "Weights must sum to 1"
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")


@OWAWeightStrategy.register("logarithmic", "log")
class LogarithmicOWAWeightStrategy(OWAWeightStrategy):
    """
    @brief Logarithmic OWA weighting strategy:
    - Lower weights: wᵢ ∝ log(i + 1)
    - Upper weights: wᵢ ∝ log(n − i + 2)
    """

    def lower_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        weights = np.log(np.arange(1, n + 1) + 1.0)
        return self._normalize(weights)

    def upper_weights(self, n: int) -> np.ndarray:
        self._validate_n(n)
        weights = np.log(np.arange(n, 0, -1) + 1.0)
        return self._normalize(weights)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        assert np.isclose(norm.sum(), 1.0), "Weights must sum to 1"
        return norm

    def _validate_n(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
