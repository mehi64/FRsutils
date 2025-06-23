import pytest
import numpy as np
from FRsutils.core.owa_weights import OWAWeightStrategy
from FRsutils.utils.logger.logger_util import get_logger
from tests.synthetic_data_store import owa_weights_testing_testsets


logger = get_logger(env="test",
                    experiment_name="test_tnorms1")
registered = OWAWeightStrategy.list_available()


@pytest.mark.parametrize("testset", owa_weights_testing_testsets())
def test_linear_owa_weight_strategy_correctness(testset):
    """
    @brief Tests whether LinearOWAWeightStrategy generates correct lower (infimum) and upper (suprimum) weights
           for different lengths, and verifies that `weights(descending=True/False)` is consistent with them.
    """
    
    obj = OWAWeightStrategy.create("linear")

    for length_key, expected_inf in testset["infimum_OWA"].items():
        n = int(length_key.split("_")[1])

        # Validate lower_weights()
        calc_inf = obj.lower_weights(n)
        np.testing.assert_allclose(calc_inf, expected_inf, atol=1e-6, err_msg=f"lower_weights mismatch for n={n}")

        # Validate upper_weights()
        expected_sup = testset["suprimum_OWA"][length_key]
        calc_sup = obj.upper_weights(n)
        np.testing.assert_allclose(calc_sup, expected_sup, atol=1e-6, err_msg=f"upper_weights mismatch for n={n}")

        # Validate weights(descending=False)
        calc_inf_2 = obj.weights(n, descending=False)
        np.testing.assert_allclose(calc_inf_2, expected_inf, atol=1e-6, err_msg=f"weights(desc=False) mismatch for n={n}")

        # Validate weights(descending=True)
        calc_sup_2 = obj.weights(n, descending=True)
        np.testing.assert_allclose(calc_sup_2, expected_sup, atol=1e-6, err_msg=f"weights(desc=True) mismatch for n={n}")


@pytest.mark.parametrize("strategy_name", list(registered.keys()))
def test_scalar_weight_shapes(strategy_name):
    strategy = OWAWeightStrategy.create(strategy_name, **({"base": 2.5} if strategy_name == "exponential" else {}))
    for n in [3, 5, 10]:
        lw = strategy.lower_weights(n)
        uw = strategy.upper_weights(n)
        w1 = strategy.weights(n)
        w2 = strategy.weights(n, descending=True)

        assert isinstance(lw, np.ndarray)
        assert isinstance(uw, np.ndarray)
        assert isinstance(w1, np.ndarray)
        assert isinstance(w2, np.ndarray)

        assert lw.shape == (n,)
        assert uw.shape == (n,)
        assert np.isclose(lw.sum(), 1.0)
        assert np.isclose(uw.sum(), 1.0)


@pytest.mark.parametrize("strategy_name", list(registered.keys()))
def test_to_dict_roundtrip(strategy_name):
    strategy = OWAWeightStrategy.create(strategy_name, **({"base": 2.5} if strategy_name == "exponential" else {}))
    d = strategy.to_dict()
    obj2 = OWAWeightStrategy.from_dict(d)
    assert isinstance(obj2, OWAWeightStrategy)
    assert obj2.name == strategy.name


@pytest.mark.parametrize("strategy_name", list(registered.keys()))
def test_weights_equivalence(strategy_name):
    strategy = OWAWeightStrategy.create(strategy_name, **({"base": 2.0} if strategy_name == "exponential" else {}))
    for n in [4, 6]:
        assert np.allclose(strategy.weights(n), strategy.lower_weights(n))
        assert np.allclose(strategy.weights(n, descending=True), strategy.upper_weights(n))


def test_exponential_param_validation():
    with pytest.raises(ValueError):
        OWAWeightStrategy.create("exponential", base=1.0)
    with pytest.raises(ValueError):
        OWAWeightStrategy.create("exp", base=0)
    with pytest.raises(ValueError):
        OWAWeightStrategy.create("exp", base="bad")


@pytest.mark.parametrize("strategy_name", list(registered.keys()))
def test_describe_params_detailed(strategy_name):
    strategy = OWAWeightStrategy.create(strategy_name, **({"base": 3.3} if strategy_name == "exponential" else {}))
    desc = strategy.describe_params_detailed()
    assert isinstance(desc, dict)
    for k in strategy._get_params().keys():
        assert k in desc