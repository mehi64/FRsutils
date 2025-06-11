"""
@file fuzzy_quantifiers.py
@brief Framework for parameterized fuzzy quantifiers used in fuzzy logic systems.

Supports registration, instantiation via alias, and computation of linear and quadratic fuzzy quantifiers.
"""

import numpy as np
from abc import ABC, abstractmethod
from FRsutils.core.registry_factory_mixin import RegistryFactoryMixin

class FuzzyQuantifier(ABC, RegistryFactoryMixin):
    """
    @brief Abstract base class for fuzzy quantifiers.

    Provides registry mechanism, interface, and parameter validation support.
    """

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
