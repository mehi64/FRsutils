"""
@file fuzzy_quantifiers.py
@brief Framework for parameterized fuzzy quantifiers used in fuzzy logic systems.

Supports registration, instantiation via alias, and computation of linear and quadratic fuzzy quantifiers.
"""

import numpy as np
from abc import ABC, abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin

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

    @classmethod
    def validate_params(self, **kwargs):
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")

        if alpha is None or not isinstance(alpha, (float, int)):
            raise ValueError(f"Missing or invalid parameter: {alpha}. It must be provided and be an int or a float number")
        if beta is None or not isinstance(beta, (float, int)):
            raise ValueError(f"Missing or invalid parameter: {beta}. It must be provided and be an int or a float number")

        if not (0 <= alpha < beta <= 1):
            raise ValueError("Require 0 <= alpha < beta <= 1")

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}


@FuzzyQuantifier.register("linear")
class LinearFuzzyQuantifier(FuzzyQuantifier):
    """
    @brief Linear fuzzy quantifier: piecewise linear increase from alpha to beta.

    Q(x) = 0            if x <= alpha  
           1            if x >= beta  
           (x - alpha)/(beta - alpha) otherwise
    """

    def __init__(self, alpha: float, beta: float):

        self.validate_params(alpha=alpha,
                             beta=beta)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x <= self.alpha, 0.0,
                        np.where(x >= self.beta, 1.0,
                                 (x - self.alpha) / (self.beta - self.alpha)))

    def to_dict(self) -> dict:
        return {"type": 'linear',
                "alpha": self.alpha,
                "beta": self.beta}
    

@FuzzyQuantifier.register("quadratic", "quad")
class QuadraticFuzzyQuantifier(FuzzyQuantifier):
    """
    @brief Quadratic fuzzy quantifier: smooth increase using a parabola.

    Q(x) =  0                                     if x <= alpha  
            2*((x-alpha)/(beta-alpha))^2          if alpha < x <= mid  
            1 - 2*((x-beta)/(beta-alpha))^2       if mid < x <= beta  
            1                                     if x > beta
    """

    def __init__(self, alpha: float, beta: float):
        
        self.validate_params(alpha=alpha,
                             beta=beta)
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

    def to_dict(self) -> dict:
        return {"type": 'quadratic',
                "alpha": self.alpha,
                "beta": self.beta}
