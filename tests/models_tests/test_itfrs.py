# import numpy as np

# import tests.syntetic_data_for_tests as sds
# from FRsutils.core.models.itfrs import ITFRS
# import FRsutils.core.tnorms as tn
# import FRsutils.core.implicators as imp

# def test_itfrs_approximations_reichenbach_imp_product_tnorm():
#     data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
#     expected_lowerBound = data_dict["Reichenbach_lowerBound"]
#     expected_upperBound = data_dict["prod_tn_upperBound"]
#     sim_matrix = data_dict["sim_matrix"]
#     y = data_dict["y"]

#     tnrm = tn.ProductTNorm()

#     model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_reichenbach)
#     lower = model.lower_approximation()
#     upper = model.upper_approximation()

#     assert lower.shape == (5,)
#     assert upper.shape == (5,)
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     closeness_LB = np.isclose(lower, expected_lowerBound)
#     assert np.all(closeness_LB), "outputs are not similatr to the expected values"

#     closeness_UB = np.isclose(upper, expected_upperBound)
#     assert np.all(closeness_UB), "outputs are not similar to the expected values"

# def test_itfrs_approximations_KD_imp_product_tnorm():
#     data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
#     expected_lowerBound = data_dict["KD_lowerBound"]
#     expected_upperBound = data_dict["prod_tn_upperBound"]
#     sim_matrix = data_dict["sim_matrix"]
#     y = data_dict["y"]

#     tnrm = tn.ProductTNorm()
    
#     model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_kleene_dienes)
#     lower = model.lower_approximation()
#     upper = model.upper_approximation()

#     assert lower.shape == (5,)
#     assert upper.shape == (5,)
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     closeness_LB = np.isclose(lower, expected_lowerBound)
#     assert np.all(closeness_LB), "outputs are not the expected values"

#     closeness_UB = np.isclose(upper, expected_upperBound)
#     assert np.all(closeness_UB), "outputs are not similar to the expected values"


# def test_itfrs_approximations_Luk_imp_product_tnorm():
#     data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
#     expected_lowerBound = data_dict["Luk_lowerBound"]
#     expected_upperBound = data_dict["prod_tn_upperBound"]
#     sim_matrix = data_dict["sim_matrix"]
#     y = data_dict["y"]

#     tnrm = tn.ProductTNorm()

#     model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_lukasiewicz)
#     lower = model.lower_approximation()
#     upper = model.upper_approximation()

#     assert lower.shape == (5,)
#     assert upper.shape == (5,)
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     closeness_LB = np.isclose(lower, expected_lowerBound)
#     assert np.all(closeness_LB), "outputs are not the expected values"

#     closeness_UB = np.isclose(upper, expected_upperBound)
#     assert np.all(closeness_UB), "outputs are not similar to the expected values"


# def test_itfrs_approximations_Goedel_imp_product_tnorm():
#     data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
#     expected_lowerBound = data_dict["Goedel_lowerBound"]
#     expected_upperBound = data_dict["prod_tn_upperBound"]
#     sim_matrix = data_dict["sim_matrix"]
#     y = data_dict["y"]

#     tnrm = tn.ProductTNorm()

#     model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_goedel)
#     lower = model.lower_approximation()
#     upper = model.upper_approximation()

#     assert lower.shape == (5,)
#     assert upper.shape == (5,)
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     closeness_LB = np.isclose(lower, expected_lowerBound)
#     assert np.all(closeness_LB), "outputs are not the expected values"

#     closeness_UB = np.isclose(upper, expected_upperBound)
#     assert np.all(closeness_UB), "outputs are not similar to the expected values"


# def test_itfrs_approximations_Gaines_imp_product_tnorm():
#     data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
#     expected_lowerBound = data_dict["Gaines_lowerBound"]
#     expected_upperBound = data_dict["prod_tn_upperBound"]
#     sim_matrix = data_dict["sim_matrix"]
#     y = data_dict["y"]

#     tnrm = tn.ProductTNorm()

#     model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_gaines)
#     lower = model.lower_approximation()
#     upper = model.upper_approximation()

#     assert lower.shape == (5,)
#     assert upper.shape == (5,)
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     closeness_LB = np.isclose(lower, expected_lowerBound)
#     assert np.all(closeness_LB), "outputs are not the expected values"

#     closeness_UB = np.isclose(upper, expected_upperBound)
#     assert np.all(closeness_UB), "outputs are not similar to the expected values"


import pytest
import numpy as np
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.tnorms import MinTNorm
from FRsutils.core.implicators import LukasiewiczImplicator
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
    return sim_matrix, labels

@pytest.fixture
def model_instance(synthetic_data):
    sim, lbl = synthetic_data
    tnorm = MinTNorm()
    implicator = LukasiewiczImplicator()
    logger = get_logger("test")
    return ITFRS(sim, lbl, tnorm, implicator, logger=logger)

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
    assert "ub_tnorm" in data
    assert "lb_implicator" in data
    assert "similarity_matrix" in data
    assert "labels" in data

def test_to_dict_exclude_data(model_instance):
    data = model_instance.to_dict(include_data=False)
    assert "type" in data
    assert "ub_tnorm" in data
    assert "lb_implicator" in data
    assert "similarity_matrix" not in data
    assert "labels" not in data

def test_from_dict_roundtrip(synthetic_data):
    sim, lbl = synthetic_data
    model = ITFRS(sim, lbl, MinTNorm(), LukasiewiczImplicator())
    d = model.to_dict(include_data=True)
    restored = ITFRS.from_dict(d)
    np.testing.assert_array_equal(restored.similarity_matrix, model.similarity_matrix)
    np.testing.assert_array_equal(restored.labels, model.labels)
    np.testing.assert_array_equal(restored.lower_approximation(), model.lower_approximation())
    np.testing.assert_array_equal(restored.upper_approximation(), model.upper_approximation())

def test_from_config_equivalence(synthetic_data):
    sim, lbl = synthetic_data
    config = {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz"
    }
    model = ITFRS.from_config(config, similarity_matrix=sim, labels=lbl)
    assert isinstance(model, ITFRS)
    assert model.similarity_matrix.shape == (3, 3)
    assert model.labels.shape == (3,)

def test_describe_params_detailed(model_instance):
    desc = model_instance.describe_params_detailed()
    assert isinstance(desc, dict)
    assert "ub_tnorm" in desc
    assert "lb_implicator" in desc

def test_get_params_internal(model_instance):
    params = model_instance._get_params()
    assert "ub_tnorm" in params
    assert "lb_implicator" in params
    assert "similarity_matrix" in params
    assert "labels" in params

def test_validate_params_invalid_tnorm(synthetic_data):
    sim, lbl = synthetic_data
    with pytest.raises(ValueError):
        ITFRS.validate_params(lb_implicator=LukasiewiczImplicator(), ub_tnorm=None)

def test_validate_params_invalid_implicator(synthetic_data):
    sim, lbl = synthetic_data
    with pytest.raises(ValueError):
        ITFRS.validate_params(lb_implicator=None, ub_tnorm=MinTNorm())

def test_logger_works(model_instance):
    model_instance.logger.info("Logger test message")


#################################
###                           ###
###      Tests with data      ###
###                           ###
#################################

@pytest.mark.parametrize("test_case", ds.get_ITFRS_testing_testsets())
@pytest.mark.parametrize("implicator_name, expected_lower_key", [
    ("reichenbach", "Reichenbach_lowerBound"),
    ("kleene-dienes", "KD_lowerBound"),
    ("lukasiewicz", "Luk_lowerBound"),
    ("goedel", "Goedel_lowerBound"),
    ("gaines", "Gaines_lowerBound")
])
@pytest.mark.parametrize("tnorm_name, expected_upper_key", [
    ("product", "prod_tn_upperBound"),
    ("minimum", "min_tn_upperBound")
])
def test_itfrs_model_with_all_settings(test_case, implicator_name, expected_lower_key, tnorm_name, expected_upper_key):
    sim = test_case["sim_matrix"]
    y = test_case["y"]
    expected = test_case["expected"]

    model = ITFRS.from_config({
        "ub_tnorm_name": tnorm_name,
        "lb_implicator_name": implicator_name
    }, similarity_matrix=sim, labels=y)

    actual_lower = model.lower_approximation()
    actual_upper = model.upper_approximation()

    np.testing.assert_allclose(actual_lower, expected[expected_lower_key], atol=1e-5, err_msg=f"Failed for {implicator_name}")
    np.testing.assert_allclose(actual_upper, expected[expected_upper_key], atol=1e-5, err_msg=f"Failed for {tnorm_name}")

    