"""
✅ Test Objectives (for each Implicator class)
✔ Implicator.create(name) instantiates correct subclass
✔ Implicator.from_dict() reconstructs accurately
✔ Implicator.to_dict() has expected format

✔ __call__() against scalar a, b
✔ __call__() against test vector a and b from get_implicator_scalar_testsets()

✔ describe_params_detailed() introspection
✔ _get_params() works for from_dict() roundtrip
"""

import numpy as np
import pytest
from FRsutils.core.implicators import Implicator
from tests import synthetic_data_store as ds
from FRsutils.utils.logger.logger_util import get_logger

logger = get_logger(env="test", experiment_name="test_implicators")
call_testsets = ds.get_implicator_scalar_testsets()
registered_implicators = Implicator.list_available()

#region <Output correctness>
###############################################
###           Output correctness            ###
###############################################

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_scalar_call_valid(implicator_name):
    """
    checks if implicator can be called with scalars and return scalar
    """
    obj = Implicator.create(implicator_name)
    a, b = 0.73, 0.18
    result = obj(a, b)
    logger.info(f"{implicator_name} result for (0.73, 0.18): {result}")
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_implicator_call_vector_output(implicator_name, testset):
    obj = Implicator.create(implicator_name)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]

    if implicator_name not in testset["expected"]:
        pytest.skip(f"Expected output not available for {implicator_name}")

    expected = testset["expected"][implicator_name]
    calculated = obj(a, b)
    np.testing.assert_allclose(calculated, expected, atol=1e-6)

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_exhaustive_grid_no_exception(implicator_name):
    """
    @brief Tests whether implicator handles full [0,1] grid without exceptions.

    Uses meshgrid of 0.0 to 1.0 with 101 values for each axis (step=0.01).
    Applies implicator elementwise on all combinations.
    """
    obj = Implicator.create(implicator_name)
    
    values = np.linspace(0.0, 1.0, 1001)
    a_grid, b_grid = np.meshgrid(values, values)
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()

    try:
        result = obj(a_flat, b_flat)
        assert result.shape == a_flat.shape
        assert np.all((0.0 <= result) & (result <= 1.0)), f"Out-of-range result from {implicator_name}"
    except Exception as e:
        pytest.fail(f"{implicator_name} raised an exception during grid evaluation: {e}")

#endregion

#region <Non-calculational behaviors>
###############################################
###       Non-calculational behaviors       ###
###############################################

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_create_and_to_dict_from_dict_roundtrip(implicator_name):
    """
    tests to_dict(), from_dict() and create() 
    """
    obj = Implicator.create(implicator_name)
    serialized = obj.to_dict()

    assert "type" in serialized
    assert "name" in serialized
    assert "params" in serialized

    reconstructed = Implicator.from_dict(serialized)
    assert isinstance(reconstructed, Implicator)
    assert reconstructed.name == obj.name


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_describe_params_detailed_keys(implicator_name):
    """
    tests values of params and deteriled params. check them in log
    """
    obj = Implicator.create(implicator_name)
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)
    for param in obj._get_params().keys():
        assert param in details

    logger.info(implicator_name + ', params:' + str(obj._get_params())+ ', detailed params:' + str(details))

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_registry_get_class_and_name(implicator_name):
    cls = Implicator.get_class(implicator_name)
    instance = cls()
    name = Implicator.get_registered_name(instance)
    assert isinstance(name, str)
    logger.info(f"{implicator_name}, registered_name: {name}")

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_help(implicator_name):
    """
    tests values of params and deteriled params. check them in log
    """
    obj = Implicator.create(implicator_name)
    details = obj.help()
    assert isinstance(details, str)
    
    logger.info(implicator_name + ', class docstring:' + details)

#endregion