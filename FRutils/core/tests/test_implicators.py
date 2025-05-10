import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import implicators

@pytest.mark.parametrize("func", [
    implicators.imp_goedel,
    implicators.imp_lukasiewicz,
    implicators.imp_product,
    implicators.imp_kleene_dienes,
    implicators.imp_reichenbach,
    implicators.imp_zadeh
])
def test_implicators_valid_range(func):
    a = 0.3
    b = 0.7
    result = func(a, b)
    assert 0.0 <= result <= 1.0, f"{func.__name__} produced value out of range"

@pytest.mark.parametrize("func", [
    implicators.imp_goedel,
    implicators.imp_lukasiewicz,
    implicators.imp_product,
    implicators.imp_kleene_dienes,
    implicators.imp_reichenbach,
    implicators.imp_zadeh
])
@pytest.mark.parametrize("a,b", [
    (-0.1, 0.5),
    (1.1, 0.5),
    (0.5, -0.2),
    (0.5, 1.2)
])
def test_implicators_invalid_input(func, a, b):
    with pytest.raises(ValueError):
        func(a, b)

def test_imp_goedel_behavior():
    assert implicators.imp_goedel(0.3, 0.5) == 1.0
    assert implicators.imp_goedel(0.8, 0.8) == 1.0
    assert implicators.imp_goedel(0.8, 0.5) == 0.5
    assert implicators.imp_goedel(0.8, 0.1) == 0.1

def test_imp_lukasiewicz_behavior():
    assert implicators.imp_lukasiewicz(0.3, 0.5) == min(1.0, 1.0 - 0.3 + 0.5)

def test_imp_product_behavior():
    assert implicators.imp_product(0.0, 0.5) == 1.0
    assert implicators.imp_product(0.5, 0.25) == 0.5
    assert implicators.imp_product(0.5, 0.9) == 1.0
    assert implicators.imp_product(0.9, 0.3) == 3.0/9.0

def test_imp_kleene_dienes_behavior():
    assert implicators.imp_kleene_dienes(0.6, 0.3) == max(1.0 - 0.6, 0.3)

def test_imp_reichenbach_behavior():
    assert implicators.imp_reichenbach(0.4, 0.7) == 1.0 - 0.4 + 0.4 * 0.7

def test_imp_zadeh_behavior():
    assert implicators.imp_zadeh(0.4, 0.6) == max(min(0.4, 0.6), 1.0 - 0.4)
