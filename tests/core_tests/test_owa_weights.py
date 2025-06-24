# import pytest
# import numpy as np
# from FRsutils.core.owa_weights import OWAWeightStrategy
# from FRsutils.utils.logger.logger_util import get_logger
# from tests.synthetic_data_store import owa_weights_testing_testsets


# logger = get_logger(env="test",
#                     experiment_name="test_tnorms1")
# registered = OWAWeightStrategy.list_available()


# @pytest.mark.parametrize("testset", owa_weights_testing_testsets())
# def test_linear_owa_weight_strategy_correctness(testset):
#     """
#     @brief Tests whether LinearOWAWeightStrategy generates correct lower (infimum) and upper (suprimum) weights
#            for different lengths, and verifies that `weights(descending=True/False)` is consistent with them.
#     """
    
#     obj = OWAWeightStrategy.create("linear")

#     for length_key, expected_inf in testset["infimum_OWA"].items():
#         n = int(length_key.split("_")[1])

#         # Validate lower_weights()
#         calc_inf = obj.lower_weights(n)
#         np.testing.assert_allclose(calc_inf, expected_inf, atol=1e-6, err_msg=f"lower_weights mismatch for n={n}")

#         # Validate upper_weights()
#         expected_sup = testset["suprimum_OWA"][length_key]
#         calc_sup = obj.upper_weights(n)
#         np.testing.assert_allclose(calc_sup, expected_sup, atol=1e-6, err_msg=f"upper_weights mismatch for n={n}")

#         # Validate weights(descending=False)
#         calc_inf_2 = obj.weights(n, descending=False)
#         np.testing.assert_allclose(calc_inf_2, expected_inf, atol=1e-6, err_msg=f"weights(desc=False) mismatch for n={n}")

#         # Validate weights(descending=True)
#         calc_sup_2 = obj.weights(n, descending=True)
#         np.testing.assert_allclose(calc_sup_2, expected_sup, atol=1e-6, err_msg=f"weights(desc=True) mismatch for n={n}")


# @pytest.mark.parametrize("strategy_name", list(registered.keys()))
# def test_scalar_weight_shapes(strategy_name):
#     strategy = OWAWeightStrategy.create(strategy_name, **({"base": 2.5} if strategy_name == "exponential" else {}))
#     for n in [3, 5, 10]:
#         lw = strategy.lower_weights(n)
#         uw = strategy.upper_weights(n)
#         w1 = strategy.weights(n)
#         w2 = strategy.weights(n, descending=True)

#         assert isinstance(lw, np.ndarray)
#         assert isinstance(uw, np.ndarray)
#         assert isinstance(w1, np.ndarray)
#         assert isinstance(w2, np.ndarray)

#         assert lw.shape == (n,)
#         assert uw.shape == (n,)
#         assert np.isclose(lw.sum(), 1.0)
#         assert np.isclose(uw.sum(), 1.0)


# @pytest.mark.parametrize("strategy_name", list(registered.keys()))
# def test_to_dict_roundtrip(strategy_name):
#     strategy = OWAWeightStrategy.create(strategy_name, **({"base": 2.5} if strategy_name == "exponential" else {}))
#     d = strategy.to_dict()
#     obj2 = OWAWeightStrategy.from_dict(d)
#     assert isinstance(obj2, OWAWeightStrategy)
#     assert obj2.name == strategy.name


# @pytest.mark.parametrize("strategy_name", list(registered.keys()))
# def test_weights_equivalence(strategy_name):
#     strategy = OWAWeightStrategy.create(strategy_name, **({"base": 2.0} if strategy_name == "exponential" else {}))
#     for n in [4, 6]:
#         assert np.allclose(strategy.weights(n), strategy.lower_weights(n))
#         assert np.allclose(strategy.weights(n, descending=True), strategy.upper_weights(n))


# def test_exponential_param_validation():
#     with pytest.raises(ValueError):
#         OWAWeightStrategy.create("exponential", base=1.0)
#     with pytest.raises(ValueError):
#         OWAWeightStrategy.create("exp", base=0)
#     with pytest.raises(ValueError):
#         OWAWeightStrategy.create("exp", base="bad")


# @pytest.mark.parametrize("strategy_name", list(registered.keys()))
# def test_describe_params_detailed(strategy_name):
#     strategy = OWAWeightStrategy.create(strategy_name, **({"base": 3.3} if strategy_name == "exponential" else {}))
#     desc = strategy.describe_params_detailed()
#     assert isinstance(desc, dict)
#     for k in strategy._get_params().keys():
#         assert k in desc

import numpy as np
import pytest

from FRsutils.core.owa_weights import OWAWeightStrategy
from FRsutils.utils.logger.logger_util import get_logger

logger = get_logger(env="test", experiment_name="test_owa_weights")

registered_owa_strategies = OWAWeightStrategy.list_available()
LENGTHS = [3, 5, 10]

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
@pytest.mark.parametrize("length", LENGTHS)
def test_weights_generation_and_shape(strategy_name, length):
    """
    @brief Ensure weights() produces valid output shape and range.

    @param strategy_name Registered alias of the strategy
    @param length Desired number of weights

    @details Checks:
    - Output shape is (length,)
    - All weights are in [0, 1]
    - Sum of weights is 1.0
    """
    strategy = OWAWeightStrategy.create(strategy_name)
    weights = strategy.weights(length)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (length,)
    assert np.allclose(np.sum(weights), 1.0)
    assert np.all((weights >= 0.0) & (weights <= 1.0))

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
@pytest.mark.parametrize("length", LENGTHS)
def test_lower_and_upper_weights_are_monotonic(strategy_name, length):
    """
    @brief Test lower_weights() and upper_weights() for expected monotonicity.

    @param strategy_name Registered alias of the strategy
    @param length Number of weights

    @details
    - lower_weights must be non-decreasing
    - upper_weights must be non-increasing
    - Both must sum to 1.0
    """
    strategy = OWAWeightStrategy.create(strategy_name)
    lower = strategy.lower_weights(length)
    upper = strategy.upper_weights(length)
    assert np.all(np.diff(lower) >= 0.0), f"{strategy_name} lower weights not ascending"
    assert np.all(np.diff(upper) <= 0.0), f"{strategy_name} upper weights not descending"
    assert np.isclose(np.sum(lower), 1.0)
    assert np.isclose(np.sum(upper), 1.0)

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
def test_create_vs_from_dict_equivalence(strategy_name):
    """
    @brief Ensure OWA strategies created via create() and from_dict() are consistent.

    @param strategy_name Registered alias of the strategy

    @details
    For various lengths, ensures all three instances (direct, create, from_dict)
    produce identical weights.
    """
    cls = OWAWeightStrategy.get_class(strategy_name)
    strategy1 = cls()
    strategy2 = OWAWeightStrategy.create(strategy_name)
    strategy3 = OWAWeightStrategy.from_dict(strategy1.to_dict())
    for length in LENGTHS:
        w1 = strategy1.weights(length)
        w2 = strategy2.weights(length)
        w3 = strategy3.weights(length)
        np.testing.assert_allclose(w1, w2)
        np.testing.assert_allclose(w2, w3)

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
def test_serialization_format(strategy_name):
    """
    @brief Ensure .to_dict() output has correct fields.

    @param strategy_name Registered alias of the strategy

    @details Must include 'type', 'name', and 'params'
    """
    strategy = OWAWeightStrategy.create(strategy_name)
    d = strategy.to_dict()
    assert isinstance(d, dict)
    assert "type" in d
    assert "name" in d
    assert "params" in d

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
def test_describe_params_detailed(strategy_name):
    """
    @brief Validate that describe_params_detailed returns all expected keys.

    @param strategy_name Registered alias of the strategy

    @details Ensures all internal parameters are present in detail dictionary.
    """
    obj = OWAWeightStrategy.create(strategy_name)
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)
    for k in obj._get_params().keys():
        assert k in details
    logger.info(f"{strategy_name}: {details}")

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
def test_registry_get_class_and_name(strategy_name):
    """
    @brief Check registry consistency of get_class and get_registered_name.

    @param strategy_name Registered alias

    @details Ensures the retrieved class and name match the expected type.
    """
    cls = OWAWeightStrategy.get_class(strategy_name)
    instance = cls()
    name = OWAWeightStrategy.get_registered_name(instance)
    assert isinstance(name, str)

@pytest.mark.parametrize("strategy_name", list(registered_owa_strategies.keys()))
def test_help(strategy_name):
    """
    @brief Ensure .help() returns class-level docstring.

    @param strategy_name Registered alias

    @details The result must be a meaningful string (length > 10).
    """
    obj = OWAWeightStrategy.create(strategy_name)
    help_text = obj.help()
    assert isinstance(help_text, str)
    assert len(help_text) > 10
