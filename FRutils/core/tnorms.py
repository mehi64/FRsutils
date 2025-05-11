# frutil/tnorms.py
"""
Collection of common T-norm functions.
"""
import numpy as np

def tn_minimum(values: np.ndarray) -> float:
    return np.min(values, axis=-1)

def tn_product(values: np.ndarray) -> float:
    return np.prod(values, axis=-1)

# def tn_lukasiewicz(values: np.ndarray) -> float:
#     return max(0.0, 1.0 - np.sum(1.0 - values))

# def tn_drastic(values: np.ndarray) -> float:
#     if np.all(values == 1):
#         return 1.0
#     elif np.any(values == 0):
#         return 0.0
#     return np.min(values)