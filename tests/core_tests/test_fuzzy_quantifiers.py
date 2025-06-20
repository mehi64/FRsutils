import pytest
import numpy as np
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier

# ----------------------------
# Functional Behavior Testing
# ----------------------------

@pytest.mark.parametrize("quant_type, alpha, beta", [
    ("linear", 0.25, 0.75),
    ("quadratic", 0.25, 0.75),
    ("linear", 0.1, 0.9),
    ("quadratic", 0.1, 0.9)
])
def test_quantifier_output_shape_and_type(quant_type, alpha, beta):
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    x = np.linspace(0, 1, 500)
    result = fq(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert (0.0 <= result).all() and (result <= 1.0).all()


@pytest.mark.parametrize("quant_type, alpha, beta, x, expected", [
    ("linear", 0.2, 0.8, np.array([0.0, 0.2, 0.5, 0.8, 1.0]), np.array([0.0, 0.0, 0.5, 1.0, 1.0])),
    ("quadratic", 0.2, 0.8, np.array([0.0, 0.2, 0.5, 0.8, 1.0]), np.array([0.0, 0.0, 0.5, 1.0, 1.0]))
])
def test_quantifier_known_outputs(quant_type, alpha, beta, x, expected):
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    result = fq(x)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# ----------------------------
# Factory and Serialization
# ----------------------------

@pytest.mark.parametrize("quant_type", ["linear", "quadratic"])
def test_create_to_dict_from_dict(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    d = fq.to_dict()
    fq2 = FuzzyQuantifier.from_dict(d)
    assert isinstance(fq2, FuzzyQuantifier)
    assert fq2.name == fq.name
    np.testing.assert_allclose(fq2._get_params()["alpha"], fq._get_params()["alpha"])
    np.testing.assert_allclose(fq2._get_params()["beta"], fq._get_params()["beta"])


# ----------------------------
# Validation and Fail-Fast
# ----------------------------

@pytest.mark.parametrize("params", [
    {"alpha": None, "beta": 0.6},
    {"alpha": 0.2, "beta": None},
    {"alpha": "a", "beta": 0.6},
    {"alpha": 0.2, "beta": "b"},
    {"alpha": 0.7, "beta": 0.6},
    {"alpha": -0.1, "beta": 1.2}
])
def test_invalid_alpha_beta(params):
    with pytest.raises(ValueError):
        FuzzyQuantifier.create("linear", **params)


# ----------------------------
# Metadata & Reflection
# ----------------------------

@pytest.mark.parametrize("quant_type", ["linear", "quadratic"])
def test_describe_and_params_match(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    described = fq.describe_params_detailed()
    params = fq._get_params()
    for k in params:
        assert k in described
        assert described[k]["value"] == params[k]
