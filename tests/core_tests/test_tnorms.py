"""
✅ Test Objectives (for each T-norm class)
Test registration & creation
✔ TNorm.create(name) instantiates correct subclass
✔ TNorm.from_dict() reconstructs accurately
✔ TNorm.to_dict() has expected format

Test __call__() correctness
✔ Against scalar pairs
✔ Against vector pairs (from get_tnorm_call_testsets())
✔ Against large matrix pairs like a_matrix, b_matrix

Test reduce() correctness
✔ Compare row-wise __call__() + np.stack to reduce() on same data

Validate parameter introspection
✔ describe_params_detailed() returns all internal parameters
✔ _get_params() works for from_dict() roundtrip
"""

import numpy as np
import pytest
from FRsutils.core.tnorms import TNorm
from tests import synthetic_data_store as ds
from FRsutils.utils.logger.logger_util import get_logger

logger = get_logger(env="test",
                    experiment_name="test_tnorms1")

call_testsets = ds.get_tnorm_call_testsets()
registered_tnorms = TNorm.list_available()

#region <test output correctness>
###############################################
###                                         ###
###         test output correctness         ###
###                                         ###
###############################################

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_scalar_inputs(tnorm_name):
    """
    cheks if the scalar inputs can be handeled correctly by tnorms
    """
    obj = TNorm.create(tnorm_name, **({"p": 0.835} if tnorm_name == "yager" else {}))
    a, b = 0.73, 0.18
    result = obj(a, b)
    logger.info(tnorm_name + ', ' + str(result))
    assert np.isscalar(result) or np.shape(result) == ()

    
@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_tnorm_call_output_matches_expected(tnorm_name, testset):
    """
    tests if generated outputs of __call__ are the same as calculated by hand
    this test uses get_tnorm_call_testsets() in synthetic_data_store
    Yager tnorm is not tested here
    """
    if "yager" in tnorm_name and "p=" not in tnorm_name:
        return
    obj = TNorm.create(tnorm_name)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]
    expected_key = tnorm_name
    if "yager" in tnorm_name:
        if "p=" in tnorm_name:
            expected_key = tnorm_name
        else:
            return
        
    calc = obj(a, b)
    exp = testset["expected"][expected_key]
    np.testing.assert_allclose(calc, exp, atol=1e-6)

@pytest.mark.parametrize("tnorm_name", ["yager"])
@pytest.mark.parametrize("p", [0.835, 5.0])
@pytest.mark.parametrize("testset", call_testsets)
def test_yager_parametrized_behavior(tnorm_name, p, testset):
    """
    tests yager __call__ function with get_tnorm_call_testsets()
    from shnthetic_data_store.py
    """
    obj = TNorm.create("yager", p=p)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]
    key = f"yager_p={p}" if p == 0.835 else "yager_p=5.0"
    result = obj(a, b)
    np.testing.assert_allclose(result, testset["expected"][key], atol=1e-5)


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_reduce_consistency_with_call(tnorm_name):
    """
    tests if the output the __call__ on a random 1D array
    is the same as output for reduce"""
    
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    data_ = np.random.rand(200,200)

    reduced = obj.reduce(data_)
    data_ = data_.T

    results = []
    for row in data_:
        res = row[0]
        for i in range(1, len(row)):
            res = obj(np.array(res), np.array(row[i]))
        results.append(float(res))

    np.testing.assert_allclose(reduced, results, atol=1e-7)

#endregion

#region <test non-calculational aspects>
###############################################
###                                         ###
###     test non-calculational aspects      ###
###                                         ###
###############################################

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_create_and_to_dict_from_dict_roundtrip(tnorm_name):
    """
    tests to_dict(), from_dict() and create()
    """
    obj = TNorm.create(tnorm_name)
    assert isinstance(obj, TNorm)

    data = obj.to_dict()
    assert "type" in data
    assert "params" in data

    obj2 = TNorm.from_dict(data)
    assert isinstance(obj2, TNorm)
    assert obj2.name == obj.name

    logger.info(tnorm_name + ', 1st obj to_dict:' + str(data)+ ', 2nd obj from_dict:' + str(obj2.to_dict()))


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_describe_params_detailed(tnorm_name):
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)
    for k in obj._get_params():
        assert k in details
    
    logger.info(tnorm_name + ', params:' + str(obj._get_params())+ ', detailed params:' + str(details))

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_registry_get_class_and_name(tnorm_name):
    cls = TNorm.get_class(tnorm_name)
    instance = cls(**({"p": 2.0} if tnorm_name == "yager" else {}))
    name = TNorm.get_registered_name(instance)
    assert isinstance(name, str)
    logger.info(tnorm_name + ', registered_name:' + str(name))

#endregion