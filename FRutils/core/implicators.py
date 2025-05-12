# frutil/implicators.py
"""
Collection of fuzzy implicator functions.
"""
import numpy as np

def imp_gaines(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    if a <= b:
        return 1.0
    elif a > 0:
        return b / a
    else:
        return 0.0

def imp_goedel(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return 1.0 if a <= b else b

def imp_kleene_dienes(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return max(1.0 - a, b)

def imp_reichenbach(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return 1.0 - a + a * b

def imp_lukasiewicz(a: float, b: float) -> float:
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("Inputs must be in range [0.0, 1.0].")
    return min(1.0, 1.0 - a + b)
