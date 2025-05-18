# import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRsutils/core')))

# import tnorms
# import syntetic_data_for_tests as sds
# # Data used for running tests

# data = np.array([0.5, 0.7, 0.34, 0.98, 1.2])

# def test_tn_minimum():
#     assert tnorms.tn_minimum(data) == 0.34

# def test_tn_product():
#     assert np.isclose(tnorms.tn_product(np.array([0.5, 0.5])), 0.25)
#     assert np.isclose(tnorms.tn_product(data), 0.139944)
    
# # def test_tn_lukasiewicz():
# #     assert np.isclose(tnorms.tn_lukasiewicz(np.array([0.9, 0.3])), 0.2)

# def test_tn_minimum_scalar_values():
#     data_dict = sds.syntetic_dataset_factory().tnorm_scalar_testing_data()
#     a_b = data_dict["a_b"]
#     expected = data_dict["minimum_outputs"]
#     temp_tnorm = tnorms.tn_minimum

#     result = []

#     l = len(a_b)
#     for i in range(l):
#         result.append(temp_tnorm(a_b[i]))
    
#     closeness = np.isclose(result, expected)
#     assert np.all(closeness), "outputs are not the expected values"

# def test_tn_product_scalar_values():
#     data_dict = sds.syntetic_dataset_factory().tnorm_scalar_testing_data()
#     a_b = data_dict["a_b"]
#     expected = data_dict["product_outputs"]
#     temp_tnorm = tnorms.tn_product

#     result = []

#     l = len(a_b)
#     for i in range(l):
#         result.append(temp_tnorm(a_b[i]))
    
#     closeness = np.isclose(result, expected)
#     assert np.all(closeness), "outputs are not the expected values"

# # def test_tn_luk_scalar_values():
# #     data_dict = sds.syntetic_dataset_factory().tnorm_scalar_testing_data()
# #     a_b = data_dict["a_b"]
# #     expected = data_dict["luk_outputs"]
# #     temp_tnorm = tnorms.tn_luk

# #     result = []

# #     l = len(a_b)
# #     for i in range(l):
# #         result.append(temp_tnorm(a_b[i]))
    
# #     closeness = np.isclose(result, expected)
# #     assert np.all(closeness), "outputs are not the expected values"

# def test_tn_minimum_nxnx2_map_values():
#     data_dict = sds.syntetic_dataset_factory().tnorm_nxnx2_testing_dataset()
#     nxnx2_map = data_dict["nxnx2_map"]
#     expected = data_dict["minimum_outputs"]
#     temp_tnorm = tnorms.tn_minimum

#     result = temp_tnorm(nxnx2_map)
    
#     closeness = np.isclose(result, expected)
#     assert np.all(closeness), "outputs are not the expected values"

# def test_tn_product_nxnx2_map_values():
#     data_dict = sds.syntetic_dataset_factory().tnorm_nxnx2_testing_dataset()
#     nxnx2_map = data_dict["nxnx2_map"]
#     expected = data_dict["product_outputs"]
#     temp_tnorm = tnorms.tn_product

#     result = temp_tnorm(nxnx2_map)
    
#     closeness = np.isclose(result, expected)
#     assert np.all(closeness), "outputs are not the expected values"

import numpy as np
import pytest
import tnorms as tn

@pytest.fixture
def sample_arrays():
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.2, 0.4, 0.8])
    arr2d = np.array([
        [0.1, 0.5, 0.9],
        [0.2, 0.4, 0.8],
        [0.9, 0.9, 0.9]
    ])
    return a, b, arr2d

def test_min_tnorm(sample_arrays):
    a, b, arr2d = sample_arrays
    t = tn.MinTNorm()
    assert np.all(t(a, b) == np.minimum(a, b))
    assert np.all(t.reduce(arr2d) == np.min(arr2d, axis=1))

def test_product_tnorm(sample_arrays):
    a, b, arr2d = sample_arrays
    t = tn.ProductTNorm()
    np.testing.assert_allclose(t(a, b), a * b)
    np.testing.assert_allclose(t.reduce(arr2d), np.prod(arr2d, axis=1))

def test_lukasiewicz_tnorm(sample_arrays):
    a, b, arr2d = sample_arrays
    t = tn.LukasiewiczTNorm()
    expected_call = np.maximum(0.0, a + b - 1.0)
    expected_reduce = np.maximum(0.0, np.sum(arr2d, axis=1) - (arr2d.shape[1] - 1))
    np.testing.assert_allclose(t(a, b), expected_call)
    np.testing.assert_allclose(t.reduce(arr2d), expected_reduce)

def test_yager_tnorm_default(sample_arrays):
    a, b, arr2d = sample_arrays
    t = tn.YagerTNorm()
    p = 2.0
    expected_call = 1.0 - np.minimum(1.0, ((1.0 - a)**p + (1.0 - b)**p)**(1.0 / p))
    expected_reduce = 1.0 - np.minimum(1.0, np.sum((1.0 - arr2d)**p, axis=1)**(1.0 / p))
    np.testing.assert_allclose(t(a, b), expected_call)
    np.testing.assert_allclose(t.reduce(arr2d), expected_reduce)

def test_yager_tnorm_custom_p(sample_arrays):
    a, b, arr2d = sample_arrays
    p = 3.0
    t = tn.YagerTNorm(p=p)
    expected_call = 1.0 - np.minimum(1.0, ((1.0 - a)**p + (1.0 - b)**p)**(1.0 / p))
    expected_reduce = 1.0 - np.minimum(1.0, np.sum((1.0 - arr2d)**p, axis=1)**(1.0 / p))
    np.testing.assert_allclose(t(a, b), expected_call)
    np.testing.assert_allclose(t.reduce(arr2d), expected_reduce)