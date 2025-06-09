"""
@file fuzzy_quantifiers.py
@brief Framework for parameterized fuzzy quantifiers used in fuzzy logic systems.

Supports registration, instantiation via alias, and computation of linear and quadratic fuzzy quantifiers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type, List
import inspect


def _filter_args(cls, kwargs: dict) -> dict:
    """
    Filters keyword arguments to only those accepted by the class constructor.

    @param cls: Class whose constructor is inspected.
    @param kwargs: Supplied keyword arguments.
    @return: Filtered arguments.
    """
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class FuzzyQuantifier(ABC):
    """
    @brief Abstract base class for fuzzy quantifiers.

    Provides registry mechanism, interface, and parameter validation support.
    """

    _registry: Dict[str, Type['FuzzyQuantifier']] = {}
    _aliases: Dict[Type['FuzzyQuantifier'], List[str]] = {}

    @classmethod
    def register(cls, *names: str):
        """
        @brief Registers a fuzzy quantifier under one or more names.

        @param names: Aliases to register.
        """
        def decorator(subclass: Type['FuzzyQuantifier']):
            if not names:
                raise ValueError("At least one alias is required.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"FuzzyQuantifier alias '{key}' already registered.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, strict: bool = False, **kwargs) -> 'FuzzyQuantifier':
        """
        @brief Instantiates a fuzzy quantifier by alias.

        @param name: Name or alias of the quantifier.
        @param strict: Whether to raise error on unused kwargs.
        @param kwargs: Parameters to pass to constructor.
        @return: FuzzyQuantifier instance.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown fuzzy quantifier alias: {name}")
        quant_cls = cls._registry[name]
        quant_cls.validate_params(**kwargs)
        ctor_args = _filter_args(quant_cls, kwargs)
        if strict:
            unused = set(kwargs) - set(ctor_args)
            if unused:
                raise ValueError(f"Unused parameters in strict mode: {unused}")
        return quant_cls(**ctor_args)

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """@return Dictionary of all registered fuzzy quantifier names and aliases."""
        return {names[0]: names for names in cls._aliases.values()}

    @classmethod
    def validate_params(cls, **kwargs):
        """Optional parameter validation hook for subclasses."""
        pass

    def to_dict(self) -> dict:
        """
        @return Serialized dictionary representation.
        """
        return {
            "type": self.__class__.__name__.replace("FuzzyQuantifier", "").lower(),
            **self._get_params()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FuzzyQuantifier':
        """
        @param data: Dictionary with type and parameters.
        @return: Deserialized FuzzyQuantifier instance.
        """
        data = data.copy()
        name = data.pop("type")
        return cls.create(name, **data)

    def _get_params(self) -> dict:
        """Override in subclasses to return constructor parameters."""
        return {}

    def help(self) -> str:
        """@return Class docstring or fallback."""
        return self.__class__.__doc__.strip() if self.__class__.__doc__ else "No documentation available."

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        @param x: Input value(s), typically in [0, 1].
        @return: Degree(s) of membership.
        """
        pass


@FuzzyQuantifier.register("linear")
class LinearFuzzyQuantifier(FuzzyQuantifier):
    """
    @brief Linear fuzzy quantifier: piecewise linear increase from alpha to beta.

    Q(p) = 0            if p <= alpha  
           1            if p >= beta  
           (p - alpha)/(beta - alpha) otherwise
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x <= self.alpha, 0.0,
                        np.where(x >= self.beta, 1.0,
                                 (x - self.alpha) / (self.beta - self.alpha)))

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def validate_params(cls, **kwargs):
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")
        if alpha is None or beta is None:
            raise ValueError("Both alpha and beta must be provided.")
        if not (0 <= alpha < beta <= 1):
            raise ValueError("Require 0 <= alpha < beta <= 1")


@FuzzyQuantifier.register("quadratic", "quad")
class QuadraticFuzzyQuantifier(FuzzyQuantifier):
    """
    @brief Quadratic fuzzy quantifier: smooth increase using a parabola.

    Q(p) = 0                           if p <= alpha  
           2*((p-alpha)/(beta-alpha))^2          if alpha < p <= mid  
           1 - 2*((p-beta)/(beta-alpha))^2       if mid < p <= beta  
           1                           if p > beta
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        mid = (self.alpha + self.beta) / 2
        denom = (self.beta - self.alpha) ** 2

        result = np.zeros_like(x)
        mask2 = (x > self.alpha) & (x <= mid)
        mask3 = (x > mid) & (x <= self.beta)
        mask4 = x > self.beta

        result[mask2] = 2 * ((x[mask2] - self.alpha) ** 2) / denom
        result[mask3] = 1 - 2 * ((x[mask3] - self.beta) ** 2) / denom
        result[mask4] = 1.0
        return result

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def validate_params(cls, **kwargs):
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")
        if alpha is None or beta is None:
            raise ValueError("Both alpha and beta must be provided.")
        if not (0 <= alpha < beta <= 1):
            raise ValueError("Require 0 <= alpha < beta <= 1")
