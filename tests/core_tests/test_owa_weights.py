"""
✅ Test Objectives for OWAWeightStrategy
✔ test registration and alias mapping
✔ test create() and to_dict()/from_dict() roundtrip
✔ test lower_weights, upper_weights, and weights(descending)
✔ test param validation (e.g., base > 1 for exponential)
✔ test describe_params_detailed
"""

import pytest
import numpy as np
from FRsutils.core.owa_weights import OWAWeightStrategy

registered = OWAWeightStrategy.list_available()


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