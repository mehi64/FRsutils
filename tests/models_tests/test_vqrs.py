# import numpy as np
# import tests.syntetic_data_for_tests as sds

# from FRsutils.core.models.vqrs import VQRS
# import FRsutils.core.tnorms as tn
# import FRsutils.core.implicators as imp


# def test_vqrs_lower_upper_approximations_quadratic_fuzzyquantifier():
#     data_dict = sds.syntetic_dataset_factory().VQRS_testing_dataset()
#     expected_lowerBound = data_dict["lower_bound"]
#     expected_upperBound = data_dict["upper_bound"]
#     sim_matrix = data_dict["sim_matrix"]
#     y = data_dict["y"]

#     alpha_lower = data_dict["alpha_lower"]
#     beta_lower  = data_dict["beta_lower"]
#     alpha_upper = data_dict["alpha_upper"]
#     beta_upper  = data_dict["beta_upper"]

#     model = VQRS(sim_matrix,
#                      y, 
#                      alpha_lower= alpha_lower,
#                      beta_lower= beta_lower,
#                      alpha_upper= alpha_upper,
#                      beta_upper= beta_upper
#                      )
    
#     upper = model.upper_approximation()
#     lower = model.lower_approximation()

#     assert lower.shape == (5,)
#     assert upper.shape == (5,)
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     closeness_LB = np.isclose(lower, expected_lowerBound)
#     assert np.all(closeness_LB), "LB outputs are not the expected values"

#     closeness_UB = np.isclose(upper, expected_upperBound)
#     assert np.all(closeness_UB), "UB outputs are not the expected values"


import numpy as np
import pytest
from FRsutils.core.models.vqrs import VQRS
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.utils.logger.logger_util import get_logger
from tests import synthetic_data_store as ds

@pytest.fixture
def synthetic_data():
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])
    fq = FuzzyQuantifier.create("linear", alpha=0.1, beta=0.6)
    return sim_matrix, labels, fq

@pytest.fixture
def model_instance(synthetic_data):
    sim, lbl, fq = synthetic_data
    return VQRS(sim, lbl, fq, fq, logger=get_logger("test"))

def test_lower_approximation(model_instance):
    lower = model_instance.lower_approximation()
    assert isinstance(lower, np.ndarray)
    assert lower.shape == (3,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))

def test_upper_approximation(model_instance):
    upper = model_instance.upper_approximation()
    assert isinstance(upper, np.ndarray)
    assert upper.shape == (3,)
    assert np.all((0.0 <= upper) & (upper <= 1.0))

def test_boundary_region(model_instance):
    boundary = model_instance.boundary_region()
    expected = model_instance.upper_approximation() - model_instance.lower_approximation()
    np.testing.assert_allclose(boundary, expected)

def test_positive_region(model_instance):
    pos = model_instance.positive_region()
    expected = model_instance.lower_approximation()
    np.testing.assert_allclose(pos, expected)

def test_to_dict_include_data(model_instance):
    data = model_instance.to_dict(include_data=True)
    assert "type" in data
    assert "fuzzy_quantifier_lower" in data
    assert "fuzzy_quantifier_upper" in data
    assert "similarity_matrix" in data
    assert "labels" in data

def test_to_dict_exclude_data(model_instance):
    data = model_instance.to_dict(include_data=False)
    assert "type" in data
    assert "fuzzy_quantifier_lower" in data
    assert "fuzzy_quantifier_upper" in data
    assert "similarity_matrix" not in data
    assert "labels" not in data

def test_from_dict_roundtrip(synthetic_data):
    sim, lbl, fq = synthetic_data
    model = VQRS(sim, lbl, fq, fq)
    d = model.to_dict(include_data=True)
    restored = VQRS.from_dict(d)
    np.testing.assert_array_equal(restored.similarity_matrix, model.similarity_matrix)
    np.testing.assert_array_equal(restored.labels, model.labels)
    np.testing.assert_array_equal(restored.lower_approximation(), model.lower_approximation())
    np.testing.assert_array_equal(restored.upper_approximation(), model.upper_approximation())

def test_from_config_equivalence(synthetic_data):
    sim, lbl, fq = synthetic_data
    config = {
        "fuzzy_quantifier_lower": fq.to_dict(),
        "fuzzy_quantifier_upper": fq.to_dict()
    }
    model = VQRS.from_config(config, similarity_matrix=sim, labels=lbl)
    assert isinstance(model, VQRS)
    assert model.similarity_matrix.shape == (3, 3)
    assert model.labels.shape == (3,)

def test_describe_params_detailed(model_instance):
    desc = model_instance.describe_params_detailed()
    assert isinstance(desc, dict)
    assert "fuzzy_quantifier_lower" in desc
    assert "fuzzy_quantifier_upper" in desc

def test_get_params_internal(model_instance):
    params = model_instance._get_params()
    assert "fuzzy_quantifier_lower" in params
    assert "fuzzy_quantifier_upper" in params
    assert "similarity_matrix" in params
    assert "labels" in params

def test_validate_params_invalid_quantifiers(synthetic_data):
    sim, lbl, _ = synthetic_data
    with pytest.raises(ValueError):
        VQRS.validate_params(fuzzy_quantifier_lower=None, fuzzy_quantifier_upper=None)

def test_logger_works(model_instance):
    model_instance.logger.info("Logger test message")

#################################
###                           ###
###      Tests with data      ###
###                           ###
#################################

@pytest.mark.parametrize("test_case", ds.get_VQRS_testing_testsets())
def test_vqrs_model_with_all_settings(test_case):
    sim = test_case["sim_matrix"]
    y = test_case["y"]
    config = {
        "fuzzy_quantifier_lower": {
            "type": "quadratic",
            "alpha": test_case["alpha_lower"],
            "beta": test_case["beta_lower"]
        },
        "fuzzy_quantifier_upper": {
            "type": "quadratic",
            "alpha": test_case["alpha_upper"],
            "beta": test_case["beta_upper"]
        }
    }

    expected = test_case["expected"]

    model = VQRS.from_config(config, similarity_matrix=sim, labels=y)
    actual_lower = model.lower_approximation()
    actual_upper = model.upper_approximation()

    np.testing.assert_allclose(actual_lower, expected["lower_bound_quadratic"], atol=1e-5)
    np.testing.assert_allclose(actual_upper, expected["upper_bound_quadratic"], atol=1e-5)