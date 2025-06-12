"""
@file tnorms.py
@brief Fuzzy T-norms Framework

Provides a pluggable architecture for defining T-norm operators used in fuzzy rough set theory.
Implements factory registration, serialization, validation, and multi-input support.

##############################################
# ✅ Quick Summary of Features
# Feature				Description
# ----------------------------------------------------------------------------------
# register(*names)		Register T-norm with aliases
# create(name, **kwargs)	Instantiate T-norm from registry
# list_available()		Returns registered T-norms
# to_dict() / from_dict()	Serialization / deserialization
# help()				Returns class-level documentation
# validate_params()		Validates constructor parameters
# name				Returns lowercase class name
# get_params()			Introspect parameter structure and values
# __call__()				Handles scalar, vector, matrix application
# reduce()				Aggregation operation

# ✅ Summary Table of Design Patterns
# Category				Name			Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern		Factory Method		TNorm.create(name, **kwargs)
# Design Pattern		Registry Pattern	TNorm._registry and register()
# Design Pattern		Template Method		Defines abstract __call__ and reduce methods
# Design Pattern		Strategy Pattern	Each subclass defines its logic
# Design Pattern		Decorator		    @TNorm.register(...)
# Design Pattern		Adapter			    Serialization via to_dict/from_dict
# Architecture		    Pluggable			New T-norms extend base class via registration
# Clean Code			SRP, DRY, LSP, Fail-Fast, Reflection
##############################################
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from FRsutils.core.registry_factory_mixin import RegistryFactoryMixin

class TNorm(ABC, RegistryFactoryMixin):
    """
    @brief Abstract base class for all T-norms.

    Provides registration, factory instantiation, serialization, and support for
    scalar/vector/matrix input handling. Subclasses must define `_compute_elementwise` and `reduce`.
    """
    
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
    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """
        @brief Reduce an array using the T-norm.

        @param arr: 2D array of shape (n_samples, n_features).
        @return: Reduced array along axis=0.
        """
        pass

@TNorm.register('minimum', 'min')
class MinTNorm(TNorm):
    """
    @brief Minimum T-norm: min(a, b)
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.min(arr, axis=0)


@TNorm.register('product', 'prod', 'algebraic')
class ProductTNorm(TNorm):
    """
    @brief Product T-norm: a * b
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.prod(arr, axis=0)


@TNorm.register('lukasiewicz', 'luk', 'bounded')
class LukasiewiczTNorm(TNorm):
    """
    @brief Łukasiewicz T-norm: max(0, a + b - 1)
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, a + b - 1.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = max(0.0, result + x - 1.0)
        return result


@TNorm.register('yager', 'yg')
class YagerTNorm(TNorm):
    """
    @brief Yager T-norm: 
    1 - min(1, [(1 - a)^p + (1 - b)^p]^(1/p))

    @param p: Exponent parameter that controls the shape (default = 2.0).
    """
    def __init__(self):
        self.validate_params()

    def __init__(self, p: float = 2.0):
        self.p = p

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Compute Yager T-norm element-wise
        return 1.0 - np.minimum(
            1.0, ((1.0 - a) ** self.p + (1.0 - b) ** self.p) ** (1.0 / self.p)
        )

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        # Reduce across axis using Yager logic
        return 1.0 - np.minimum(
            1.0, np.sum((1.0 - arr) ** self.p, axis=0) ** (1.0 / self.p)
        )

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get('p')
        if p is None:
            raise ValueError("Missing required parameter: p")
        if not isinstance(p, (int, float)):
            raise ValueError("Parameter 'p' must be a float or int")
        if p <= 0:
            raise ValueError("Parameter 'p' must be greater than 0")
        if p is None:
            raise ValueError("Missing required parameter: p")
        if not isinstance(p, (int, float)):
            raise ValueError("Parameter 'p' must be a float or int")
        if p <= 0:
            raise ValueError("Parameter 'p' must be greater than 0")

    def _get_params(self) -> dict:
        return {"p": self.p}



@TNorm.register("drastic", "drastic_product")
class DrasticProductTNorm(TNorm):
    """
    @brief Drastic Product T-norm:
    - a if b == 1
    - b if a == 1
    - 0 otherwise
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(b == 1.0, a, np.where(a == 1.0, b, 0.0))

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = np.where(x == 1.0, result, np.where(result == 1.0, x, 0.0))
        return result


@TNorm.register("einstein", "einstein_product")
class EinsteinProductTNorm(TNorm):
    """
    @brief Einstein Product T-norm:
    T(a, b) = (a * b) / (2 - (a + b - a * b))
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = 2 - (a + b - a * b)
        return np.where(denom != 0, (a * b) / denom, 0.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = (result * x) / (2 - (result + x - result * x))
        return result


@TNorm.register("hamacher", "hamacher_product")
class HamacherProductTNorm(TNorm):
    """
    @brief Hamacher Product T-norm:
    T(a, b) = (a * b) / (a + b - a * b) if (a + b - a * b) != 0 else 0
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = a + b - a * b
            result = np.divide(a * b, denom, out=np.zeros_like(a), where=denom != 0)
        return result


    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            denom = result + x - result * x
            result = (result * x) / denom if denom != 0 else 0.0
        return result


@TNorm.register("nilpotent", "nilpotent_minimum")
class NilpotentMinimumTNorm(TNorm):
    """
    @brief Nilpotent Minimum T-norm:
    T(a, b) = min(a, b) if (a + b) > 1 else 0
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where((a + b) > 1.0, np.minimum(a, b), 0.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = np.where((result + x) > 1.0, np.minimum(result, x), 0.0)
        return result


@TNorm.register("lambda", "lambda_tnorm")
class LambdaTNorm(TNorm):
    """
    @brief Lambda T-norm:
    T(a, b) = lambda * a * b / (1 + (lambda - 1) * (a + b - a * b))

    @param l: Lambda parameter (must be > 0)
    """
    def __init__(self):
        self.validate_params()

    def __init__(self, l: float = 1.0):
        self.l = l

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        numerator = self.l * a * b
        denominator = 1 + (self.l - 1) * (a + b - a * b)
        return np.where(denominator != 0, numerator / denominator, 0.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = self.__call__(result, x)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        l = kwargs.get("l")
        if l is None:
            raise ValueError("Missing required parameter: l")
        if not isinstance(l, (float, int)) or l <= 0:
            raise ValueError("Lambda parameter 'l' must be a positive number")
        if l is None:
            raise ValueError("Missing required parameter: l")
        if not isinstance(l, (float, int)) or l <= 0:
            raise ValueError("Lambda parameter 'l' must be a positive number")

    def _get_params(self) -> dict:
        return {"l": self.l}


# @TNorm.register("function", "function_based")
# class FunctionNormTNorm(TNorm):
#     """
#     @brief Function-based T-norm:
#     Custom function supplied at runtime.

#     @param func: Callable taking two floats or arrays and returning array
#     """
#     def __init__(self, func):
#         if not callable(func):
#             raise ValueError("FunctionNorm requires a callable 'func'")
#         self.func = func

#     def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
#         return self.func(a, b)

#     def reduce(self, arr: np.ndarray) -> np.ndarray:
#         result = arr[0]
#         for x in arr[1:]:
#             result = self.__call__(result, x)
#         return result

#     def _get_params(self) -> dict:
#         return {}  # Not serializable by default
