

# ✅ Quick Summary of Features
# Feature	Description
# register(*names)	Register implicator with aliases
# create(name, **kwargs)	Instantiate implicator from registry
# list_available()	Returns registered implicators
# to_dict() / from_dict()	Serialization / deserialization
# help()	Returns class-level documentation
# validate_params()	Validates constructor parameters
# describe_params_detailed()	Returns parameter types and values
# apply_pairwise_matrix()	Applies implicator to matrix pairs
# __call__()	Smart dispatcher for scalar, vector, or matrix


# ✅ Summary Table of Design Principles
# Category	Name	Usage & Where Applied
#################################################################################
# Design Pattern	Factory Method	                Implicator.create(name, **kwargs) dynamically creates Implicator objects from registered classes based on name/alias.
# Design Pattern	Registry Pattern	            Implicator._registry and register() decorator maintain a central registry of all available Implicator subclasses and aliases.
# Design Pattern	Template Method	                Abstract base class Implicator defines method signatures like __call__(), which are implemented differently by subclasses.
# Design Pattern	Decorator (Class Registration)	@Implicator.register('name', ...) dynamically registers subclasses with aliases using a class decorator.
# Design Pattern	Strategy Pattern	            Each Implicator subclass (e.g., Goedel, Yager) implements its own strategy for computing the fuzzy implication.
# Design Pattern	Adapter (Serialization)	        to_dict() / from_dict() methods act as an adapter for converting between object and dictionary representation for serialization.
# Architecture	    Pluggable Architecture	        New implicators can be added without modifying core logic----just register via decorator, enabling extensibility.
# Clean Code	    Single Responsibility Principle	Each class and method does one thing (e.g., Frank handles only Frank logic; validate_params handles only validation).
# Clean Code	    Open/Closed Principle	        Framework is open for extension (add new Implicators) but closed for modification (no need to change base class).
# Clean Code	    DRY (Don't Repeat Yourself)	    Common logic (e.g. alias filtering, validation pattern) is centralized and reused via helper methods.
# Clean Code	    Encapsulation & Abstraction	    Abstract base class hides implementation details from users and enforces a consistent API.
# Clean Code	    Liskov Substitution Principle	All Implicator subclasses can be used wherever an Implicator object is expected without breaking functionality.
# Clean Code	    Docstring Documentation	        Doxygen-style docstrings enable IDE and tool-friendly introspection and maintainability.
# Design Support	Reflection/Introspection	    inspect.signature() in _filter_args() allows dynamic and safe instantiation of subclasses with correct parameters.


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
        """
        pass


    def _validate_inputs(self, a, b):
        """
        Ensures inputs a and b are in the range [0, 1].

        @param a: scalar float
        @param b: scalar float
        @raises ValueError: if inputs are out of range
        """
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError(f"Inputs must be in [0, 1]: {a}, {b}")

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        @brief Apply the T-norm to two arrays element-wise.

        @param a: First input array.
        @param b: Second input array.
        @return: Element-wise result of the T-norm.
        """
        pass

    @abstractmethod
    def _compute_scalar(self, a: float, b: float) -> float:
        """
        @brief Perform element-wise implicator operation on arrays of any shape.

        @param a: scalar float
        @param b: scalar float
        @return: scalar float
        """
        pass
    
    # def apply_pairwise_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    #     """
    #     @brief Apply the implicator element-wise between two matrices of equal shape.

    #     @param a: An (n x n) matrix of values (e.g., similarities).
    #     @param b: An (n x n) matrix of values (e.g., mask or labels).
    #     @return: An (n x n) matrix of implicator outputs.
    #     """
    #     if a.shape != b.shape:
    #         raise ValueError("Input matrices must have the same shape.")
    #     vec_call = np.vectorize(self.__call__)
    #     return vec_call(a, b)

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


    def _get_params(self) -> dict:
        """
        @brief Get serializable parameters for reconstruction.

        Override this in subclasses that use constructor parameters.
        @return: Dictionary of parameters.
        """
        return {
            k: getattr(self, k)
            for k in inspect.signature(self.__init__).parameters
            if k != "self" and hasattr(self, k)
        }

    def describe_params_detailed(self) -> dict:
        """
        @brief Returns a dictionary describing the parameters used in this implicator instance.

        @return: Dictionary mapping parameter names to their type and current value.
        """
        sig = inspect.signature(self.__init__)
        params = {}
        for name in sig.parameters:
            if name == "self":
                continue
            if hasattr(self, name):
                val = getattr(self, name)
                params[name] = {
                    "type": type(val).__name__,
                    "value": val
                }
        return params

    def help(self) -> str:
        """
        Returns the docstring of the implicator class.

        @return: String documentation.
        """
        return inspect.getdoc(self.__class__) or "No documentation available."

    @property
    def name(self) -> str:
        """
        @brief Returns the registered name of the tnorm class.

        @return: The class name as a lowercase string without TNorm prefix.
        """
        return self.__class__.__name__.replace("Implicator", "").lower()

# Non-parameterized implicators
@Implicator.register("gaines")
class GainesImplicator(Implicator):
    """
    Gaines fuzzy implicator:
    - If a <= b: returns 1
    - If a > b and a > 0: returns b / a
    - If a == 0: returns 0
    """
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError("Input arrays 'a' and 'b' must have the same shape.")

        result = np.ones_like(a, dtype=np.float64)

        # Case where a <= b
        mask_le = a <= b
        result[mask_le] = 1.0

        # Case where a > b and a > 0
        mask_gt = (a > b) & (a > 0)
        result[mask_gt] = np.minimum(1.0, b[mask_gt] / a[mask_gt])

        # Case where a == 0 (return 1 by convention)
        # Already initialized to 1.0

        return result
    
    def _compute_scalar(self, a: float, b: float) -> float:
        self._validate_inputs(a, b)
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
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape: 
            raise ValueError("Input arrays must have the same shape.") 
        return np.where(a <= b, 1.0, b)
    
    def _compute_scalar(self, a: float, b: float) -> float:
        self._validate_inputs(a, b)
        return 1.0 if a <= b else b

@Implicator.register("kleene", "kleene-dienes")
class KleeneDienesImplicator(Implicator):
    """
    Kleene-Dienes fuzzy implicator:
    - Computes max(1 - a, b)
    """
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError("Input arrays must have the same shape.")
        return np.maximum(1 - a, b)

    def _compute_scalar(self, a: float, b: float) -> float:
        self._validate_inputs(a, b)
        return max(1.0 - a, b)

@Implicator.register("reichenbach")
class ReichenbachImplicator(Implicator):
    """
    Reichenbach fuzzy implicator:
    - Computes 1 - a + a * b
    """
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError("Input arrays must have the same shape.")
        return 1.0 - a + a * b

    def _compute_scalar(self, a: float, b: float) -> float:
        self._validate_inputs(a, b)
        return 1.0 - a + a * b

@Implicator.register("lukasiewicz","luk")
class LukasiewiczImplicator(Implicator):
    """
    Łukasiewicz fuzzy implicator:
    - Computes min(1, 1 - a + b)
    """
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError("Input arrays must have the same shape.")
        return np.minimum(1.0, 1 - a + b)

    def _compute_scalar(self, a: float, b: float) -> float:
        self._validate_inputs(a, b)
        return min(1.0, 1.0 - a + b)

# # Parameterized implicators
# @Implicator.register("yager")
# class YagerImplicator(Implicator):
#     """
#     Yager fuzzy implicator:
#     - Computes min(1, (1 - a)^p + b^p)^(1/p)

#     @param p: Exponent parameter > 0 (default 2)
#     """
#     def __init__(self, p: float = 2.0):
#         self.p = p

#     def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
#         if a.shape != b.shape:
#             raise ValueError("Input arrays must have the same shape.")
#         result = np.ones_like(a)
#         mask = a > b
#         result[mask] = np.power(
#             np.maximum(0.0, 1 - np.power(a[mask], self.p) + np.power(b[mask], self.p)),
#             1.0 / self.p
#         )
#         return result

#     def _compute_scalar(self, a: float, b: float) -> float:
#         if not (0 <= a <= 1 and 0 <= b <= 1):
#             raise ValueError("Inputs must be in range [0, 1].")
#         return min(1.0, ((1 - a) ** self.p + b ** self.p) ** (1 / self.p))

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")


#     def _validate_inputs(self, a, b):
#         """
#         Ensures inputs a and b are in the range [0, 1].

#         @param a: scalar float
#         @param b: scalar float
#         @raises ValueError: if inputs are out of range
#         """
#         self._validate_inputs(a, b)

#         if p is None:
#             raise ValueError("Missing required parameter: p")
#         if not isinstance(p, (int, float)) or p <= 0:
#             raise ValueError("Parameter 'p' must be a positive number.")

# @Implicator.register("weber")
# class WeberImplicator(Implicator):
#     """
#     Weber fuzzy implicator:
#     - Computes min(1, (b^p) / (a^p + (1 - a)^p))

#     @param p: Exponent parameter > 0 (default 2)
#     """
#     def __init__(self, p: float = 2.0):
#         self.p = p
    
#     def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
#         if a.shape != b.shape:
#             raise ValueError("Input arrays must have the same shape.")
#         result = np.ones_like(a, dtype=np.float64)
#         mask = a > 0
#         result[mask] = np.minimum(1.0, b[mask] / a[mask])
#         return result

#     def _compute_scalar(self, a: float, b: float) -> float:
#         denom = (a ** self.p + (1 - a) ** self.p)
#         return min(1.0, b ** self.p / denom if denom != 0 else 1.0)

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")


#     def _validate_inputs(self, a, b):
#         """
#         Ensures inputs a and b are in the range [0, 1].

#         @param a: scalar float
#         @param b: scalar float
#         @raises ValueError: if inputs are out of range
#         """
#         self._validate_inputs(a, b)

#         if p is None:
#             raise ValueError("Missing required parameter: p")
#         if not isinstance(p, (int, float)) or p <= 0:
#             raise ValueError("Parameter 'p' must be a positive number.")

# # @Implicator.register("frank")
# # class FrankImplicator(Implicator):
# #     """
# #     Frank fuzzy implicator:
# #     - Computes log_b(1 + ((b^p - 1)(1 - a^p)) / (p - 1))

# #     @param p: Base parameter, p > 0 and p != 1 (default: 2.0)
# #     """
# #     def __init__(self, p: float = 2.0):
# #         self.p = p
#     # def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     #     if a.shape != b.shape:
#     #         raise ValueError("Input arrays must have the same shape.")
#     #     numerator = (np.power(self.s, b) - 1) * (np.power(self.s, 1 - a) - 1)
#     #     denominator = self.s - 1
#     #     result = 1 + numerator / denominator
#     #     return np.log(result) / np.log(self.s)

# #     def _compute_elementwise(self, a: float, b: float) -> float:
# #         if self.p == 1:
# #             return 1.0 - a + a * b
# #         num = (self.p ** b - 1) * (1 - self.p ** a)
# #         denom = self.p - 1
# #         result = 1 + num / denom
# #         return np.clip(np.log(result) / np.log(self.p), 0, 1)

# #     @classmethod
# #     def validate_params(cls, **kwargs):
# #         p = kwargs.get("p")


# #     def _validate_inputs(self, a, b):
# #         """
# #         Ensures inputs a and b are in the range [0, 1].

# #         @param a: scalar float
# #         @param b: scalar float
# #         @raises ValueError: if inputs are out of range
# #         """
# #         self._validate_inputs(a, b)

# #         if p is None or not isinstance(p, (int, float)) or p <= 0 or p == 1:
# #             raise ValueError("Parameter 'p' must be > 0 and != 1")

# # @Implicator.register("sugeno-weber", "sw")
# # class SugenoWeberImplicator(Implicator):
# #     """
# #     Sugeno-Weber fuzzy implicator:
# #     - Computes min(1, (b - a + a * b) / (1 + p * (1 - a) * b))

# #     @param p: Interaction parameter (default: 1.0)
# #     """
# #     def __init__(self, p: float = 1.0):
# #         self.p = p

# #     def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
# #         if a.shape != b.shape:
# #             raise ValueError("Input arrays must have the same shape.")
# #         numerator = 1 - a + b - self.lambd * a * (1 - b)
# #         denominator = 1 + self.lambd
# #         return np.minimum(1.0, numerator / denominator)

# #     def _compute_elementwise(self, a: float, b: float) -> float:
# #         denom = 1 + self.p * (1 - a) * b
# #         return min(1.0, (b - a + a * b) / denom if denom != 0 else 1.0)

# #     @classmethod
# #     def validate_params(cls, **kwargs):
# #         p = kwargs.get("p")


# #     def _validate_inputs(self, a, b):
# #         """
# #         Ensures inputs a and b are in the range [0, 1].

# #         @param a: scalar float
# #         @param b: scalar float
# #         @raises ValueError: if inputs are out of range
# #         """
# #         self._validate_inputs(a, b)

# #         if p is None or not isinstance(p, (int, float)):
# #             raise ValueError("Parameter 'p' must be a number")
