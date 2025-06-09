"""
Unit tests for T-norm classes in tnorms.py
"""

import numpy as np
import pytest
from FRsutils.core.tnorms import TNorm
from tests.syntetic_data_for_tests import syntetic_dataset_factory

# Fixtures
@pytest.fixture
def scalar_data():
    return syntetic_dataset_factory().tnorm_scalar_testing_data()

@pytest.fixture
def matrix_data():
    return syntetic_dataset_factory().tnorm_nxnx2_testing_dataset()

# Parametrize test cases: (alias, expected_output_key)
scalar_cases = [
    ("minimum", "minimum_outputs"),
    ("product", "product_outputs"),
    ("lukasiewicz", "luk_outputs"),
]

matrix_cases = [
    ("minimum", "minimum_outputs"),
    ("product", "product_outputs"),
    ("lukasiewicz", "luk_outputs"),
]

# Basic behavior tests
@pytest.mark.parametrize("alias, output_key", scalar_cases)
def test_tnorm_scalar_behavior(alias, output_key, scalar_data):
    tn = TNorm.create(alias)
    a_b = scalar_data["a_b"]
    expected = scalar_data[output_key]
    actual = np.array([tn(np.array([a]), np.array([b]))[0] for a, b in a_b])
    np.testing.assert_allclose(actual, expected, rtol=1e-3, err_msg=f"{alias} scalar call failed")

@pytest.mark.parametrize("alias, output_key", matrix_cases)
def test_tnorm_matrix_behavior(alias, output_key, matrix_data):
    tn = TNorm.create(alias)
    a = matrix_data["similarity_matrix"]
    b = matrix_data["label_mask"]
    expected = matrix_data[output_key]
    actual = tn(a, b)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, err_msg=f"{alias} matrix call failed")

@pytest.mark.parametrize("alias", [c[0] for c in scalar_cases])
def test_tnorm_reduce_output_shape(alias, matrix_data):
    tn = TNorm.create(alias)
    reduced = tn.reduce(matrix_data["similarity_matrix"])
    assert reduced.shape == (matrix_data["similarity_matrix"].shape[1],)

# Factory & Registry tests
def test_factory_create_and_aliases():
    aliases = TNorm.list_available()
    for primary, names in aliases.items():
        for alias in names:
            tn = TNorm.create(alias)
            assert isinstance(tn, TNorm)

def test_invalid_alias_raises():
    with pytest.raises(ValueError):
        TNorm.create("unknown_tnorm")

def test_strict_mode_extra_param_raises():
    with pytest.raises(ValueError):
        TNorm.create("minimum", strict=True, bogus_param=123)

# Serialization
@pytest.mark.parametrize("alias", [c[0] for c in scalar_cases])
def test_serialization_roundtrip(alias):
    tn = TNorm.create(alias)
    tn_dict = tn.to_dict()
    tn2 = TNorm.from_dict(tn_dict)
    assert isinstance(tn2, TNorm)

# Help
@pytest.mark.parametrize("alias", [c[0] for c in scalar_cases])
def test_help_returns_docstring(alias):
    tn = TNorm.create(alias)
    assert isinstance(tn.help(), str)
    assert len(tn.help()) > 0

# Special parameterized T-norms
def test_yager_param_validation():
    with pytest.raises(ValueError): TNorm.create("yager")  # missing p
    with pytest.raises(ValueError): TNorm.create("yager", p="bad")
    with pytest.raises(ValueError): TNorm.create("yager", p=-1)
    tn = TNorm.create("yager", p=2)
    assert isinstance(tn, TNorm)

def test_lambda_param_validation():
    with pytest.raises(ValueError): TNorm.create("lambda")  # missing l
    with pytest.raises(ValueError): TNorm.create("lambda", l="bad")
    with pytest.raises(ValueError): TNorm.create("lambda", l=-2)
    tn = TNorm.create("lambda", l=1.2)
    assert isinstance(tn, TNorm)

def test_drastic_product_behavior():
    tn = TNorm.create("drastic")
    a = np.array([0.8, 1.0, 0.3])
    b = np.array([1.0, 0.5, 0.4])
    expected = np.array([0.8, 0.5, 0.0])
    result = tn(a, b)
    np.testing.assert_allclose(result, expected)

def test_nilpotent_minimum_behavior():
    tn = TNorm.create("nilpotent")
    a = np.array([0.6, 0.7])
    b = np.array([0.6, 0.2])
    expected = np.array([0.6, 0.0])
    result = tn(a, b)
    np.testing.assert_allclose(result, expected)

def test_hamacher_zero_denom_behavior():
    tn = TNorm.create("hamacher")
    a = np.array([0.0, 0.5])
    b = np.array([0.0, 0.5])
    result = tn(a, b)
    assert result.shape == a.shape
