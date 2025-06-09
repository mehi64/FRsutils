"""
Unit tests for individual T-norm classes using synthetic scalar data testsets.
"""

import numpy as np
import pytest
from FRsutils.core.tnorms import TNorm
from tests.synthetic_data_store import get_tnorm_scalar_testsets

@pytest.fixture(scope="module")
def scalar_testsets():
    return get_tnorm_scalar_testsets()


# Map aliases to canonical TNorm creation info
alias_map = {
    "minimum": ("minimum", {}),
    "product": ("product", {}),
    "lukasiewicz": ("lukasiewicz", {}),
    "drastic_product": ("drastic", {}),
    "hamacher_product": ("hamacher", {}),
    "einstein": ("einstein", {}),
    "nilpotent_min": ("nilpotent", {}),
    "yager_p=0.835": ("yager", {"p": 0.835}),
}

@pytest.mark.parametrize("testset", get_tnorm_scalar_testsets())
@pytest.mark.parametrize("alias_key", list(alias_map.keys()))
def test_tnorm_scalar_by_class(alias_key, testset):
    if alias_key not in testset["expected"]:
        pytest.skip(f"Testset '{testset['name']}' has no expected outputs for '{alias_key}'")

    alias, kwargs = alias_map[alias_key]
    tn = TNorm.create(alias, **kwargs)
    a_b = testset["a_b"]
    expected = testset["expected"][alias_key]
    actual = np.array([tn(np.array([a]), np.array([b]))[0] for a, b in a_b])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, err_msg=f"{alias_key} failed for testset '{testset['name']}'")
