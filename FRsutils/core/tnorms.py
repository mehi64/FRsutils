"""
T-norm System

Provides an extensible and optimized framework to compute T-norms (using
function-call wrappers).

@author: Mehran Amiri
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

# ------------------------------------------------------------------------------
# T-Norm Function Wrapper
# ------------------------------------------------------------------------------
class TNorm:
    """
    Wrapper for T-norm operations that allows parameterized calls.

    @param func: A function that takes two ndarrays and optional kwargs
    @param kwargs: Optional keyword arguments passed to the function
    """
    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.func(a, b, **self.kwargs)


# ------------------------------------------------------------------------------
# Common T-norm Functions
# ------------------------------------------------------------------------------

def tnorm_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

def tnorm_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def tnorm_lukasiewicz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, a + b - 1.0)

def tnorm_yager(a: np.ndarray, b: np.ndarray, p: float = 2.0) -> np.ndarray:
    return 1.0 - np.minimum(1.0, ((1 - a) ** p + (1 - b) ** p) ** (1.0 / p))
