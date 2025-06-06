"""
Fuzzy Implicators Framework

This module provides an extensible, class-based framework for computing fuzzy logic
implicators. It supports serialization, deserialization, and dynamic creation of
implicator instances by alias name. Parameterized implicators such as Yager, Weber,
Frank, and Sugeno-Weber are also included.

@file fuzzy_implicators.py
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Dict, List
import inspect
import json

def _filter_args(cls, kwargs: dict) -> dict:
    """
    Filters keyword arguments to match the constructor of a given class.

    @param cls: The target class whose constructor will be inspected.
    @param kwargs: Keyword arguments to filter.
    @return: Filtered dictionary of constructor-compatible arguments.
    """
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

class Implicator(ABC):
    """
    Abstract base class for fuzzy implicators.

    Provides a registration mechanism, dynamic creation, and interface
definitions for all fuzzy implicator implementations.
    """

    _registry: Dict[str, Type['Implicator']] = {}
    _aliases: Dict[Type['Implicator'], List[str]] = {}

    @classmethod
    def register(cls, *names: str):
        """
        Registers a fuzzy implicator class with one or more aliases.

        @param names: List of string aliases for the implicator.
        """
        def decorator(subclass: Type['Implicator']):
            if not names:
                raise ValueError("At least one name must be provided for registration.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"Implicator alias '{key}' is already registered.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """
        Lists all registered implicators along with their aliases.

        @return: Dictionary mapping primary name to list of aliases.
        """
        return {names[0]: names for subclass, names in cls._aliases.items()}

    @classmethod
    def create(cls, name: str, strict: bool = False, **kwargs) -> 'Implicator':
        """
        Creates an implicator instance by name.

        @param name: Alias of the implicator.
        @param strict: Whether to raise an error on unused parameters.
        @param kwargs: Parameters for the constructor.
        @return: An instance of the requested implicator.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown implicator alias: {name}")
        implicator_cls = cls._registry[name]
        implicator_cls.validate_params(**kwargs)
        ctor_args = _filter_args(implicator_cls, kwargs)
        if strict:
            unused = set(kwargs) - set(ctor_args)
            if unused:
                raise ValueError(f"Unused parameters: {unused}")
        return implicator_cls(**ctor_args)

    @classmethod
    def validate_params(cls, **kwargs):
        """
        Validates constructor parameters.

        @param kwargs: Parameters to validate.
        """
        pass

    @abstractmethod
    def __call__(self, a: float, b: float) -> float:
        """
        Computes the implicator result for two fuzzy values.

        @param a: Antecedent value [0.0, 1.0].
        @param b: Consequent value [0.0, 1.0].
        @return: Resulting fuzzy truth value.
        """
        pass

    def to_dict(self) -> dict:
        """
        Serializes the implicator instance to a dictionary.

        @return: Serializable representation.
        """
        return {
            "type": self.__class__.__name__,
            "params": {
                k: getattr(self, k)
                for k in inspect.signature(self.__init__).parameters
                if k != "self" and hasattr(self, k)
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Implicator':
        """
        Deserializes an implicator from a dictionary.

        @param data: Serialized representation.
        @return: Implicator instance.
        """
        return cls.create(data["type"], **data.get("params", {}))

    def help(self) -> str:
        """
        Returns the docstring of the implicator class.

        @return: String documentation.
        """
        return inspect.getdoc(self.__class__) or "No documentation available."

# Non-parameterized implicators
@Implicator.register("gaines")
class GainesImplicator(Implicator):
    """
    Gaines fuzzy implicator:
    - If a <= b: returns 1
    - If a > b and a > 0: returns b / a
    - If a == 0: returns 0
    """
    def __call__(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        if a <= b:
            return 1.0
        elif a > 0:
            return b / a
        else:
            return 0.0

@Implicator.register("goedel")
class GoedelImplicator(Implicator):
    """
    Gödel fuzzy implicator:
    - If a <= b: returns 1
    - Else: returns b
    """
    def __call__(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return 1.0 if a <= b else b

@Implicator.register("kleene", "kleene-dienes")
class KleeneDienesImplicator(Implicator):
    """
    Kleene-Dienes fuzzy implicator:
    - Computes max(1 - a, b)
    """
    def __call__(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return max(1.0 - a, b)

@Implicator.register("reichenbach")
class ReichenbachImplicator(Implicator):
    """
    Reichenbach fuzzy implicator:
    - Computes 1 - a + a * b
    """
    def __call__(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return 1.0 - a + a * b

@Implicator.register("lukasiewicz","luk")
class LukasiewiczImplicator(Implicator):
    """
    Łukasiewicz fuzzy implicator:
    - Computes min(1, 1 - a + b)
    """
    def __call__(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return min(1.0, 1.0 - a + b)

# Parameterized implicators
@Implicator.register("yager")
class YagerImplicator(Implicator):
    """
    Yager fuzzy implicator:
    - Computes min(1, (1 - a)^p + b^p)^(1/p)

    @param p: Exponent parameter > 0 (default 2)
    """
    def __init__(self, p: float = 2.0):
        self.p = p

    def __call__(self, a: float, b: float) -> float:
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("Inputs must be in range [0, 1].")
        return min(1.0, ((1 - a) ** self.p + b ** self.p) ** (1 / self.p))

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get("p")
        if p is None:
            raise ValueError("Missing required parameter: p")
        if not isinstance(p, (int, float)) or p <= 0:
            raise ValueError("Parameter 'p' must be a positive number.")

@Implicator.register("weber")
class WeberImplicator(Implicator):
    """
    Weber fuzzy implicator:
    - Computes min(1, (b^p) / (a^p + (1 - a)^p))

    @param p: Exponent parameter > 0 (default 2)
    """
    def __init__(self, p: float = 2.0):
        self.p = p

    def __call__(self, a: float, b: float) -> float:
        denom = (a ** self.p + (1 - a) ** self.p)
        return min(1.0, b ** self.p / denom if denom != 0 else 1.0)

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get("p")
        if p is None:
            raise ValueError("Missing required parameter: p")
        if not isinstance(p, (int, float)) or p <= 0:
            raise ValueError("Parameter 'p' must be a positive number.")

@Implicator.register("frank")
class FrankImplicator(Implicator):
    """
    Frank fuzzy implicator:
    - Computes log_b(1 + ((b^p - 1)(1 - a^p)) / (p - 1))

    @param p: Base parameter, p > 0 and p != 1 (default: 2.0)
    """
    def __init__(self, p: float = 2.0):
        self.p = p

    def __call__(self, a: float, b: float) -> float:
        if self.p == 1:
            return 1.0 - a + a * b
        num = (self.p ** b - 1) * (1 - self.p ** a)
        denom = self.p - 1
        result = 1 + num / denom
        return np.clip(np.log(result) / np.log(self.p), 0, 1)

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get("p")
        if p is None or not isinstance(p, (int, float)) or p <= 0 or p == 1:
            raise ValueError("Parameter 'p' must be > 0 and != 1")

@Implicator.register("sugeno-weber", "sw")
class SugenoWeberImplicator(Implicator):
    """
    Sugeno-Weber fuzzy implicator:
    - Computes min(1, (b - a + a * b) / (1 + p * (1 - a) * b))

    @param p: Interaction parameter (default: 1.0)
    """
    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, a: float, b: float) -> float:
        denom = 1 + self.p * (1 - a) * b
        return min(1.0, (b - a + a * b) / denom if denom != 0 else 1.0)

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get("p")
        if p is None or not isinstance(p, (int, float)):
            raise ValueError("Parameter 'p' must be a number")
