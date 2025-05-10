# frutil/implicators.py
"""
Collection of fuzzy implicator functions.
"""
import numpy as np

def imp_goedel(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return 1.0 if a <= b else b

def imp_lukasiewicz(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return min(1.0, 1.0 - a + b)

def imp_product(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return 1.0 if a <= b else b / a if a != 0 else 1.0

def imp_kleene_dienes(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return max(1.0 - a, b)

def imp_reichenbach(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return 1.0 - a + a * b

def imp_zadeh(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return max(min(a, b), 1.0 - a)