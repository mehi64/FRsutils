import pytest
import numpy as np
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.tnorms import MinTNorm
from FRsutils.core.implicators import LukasiewiczImplicator
from FRsutils.utils.logger.logger_util import get_logger
from tests import synthetic_data_store as ds

@pytest.fixture
def synthetic_data_():
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])
    return sim_matrix, labels

@pytest.fixture
def model_instance(synthetic_data_):
    sim, lbl = synthetic_data_
    tnorm = MinTNorm()
    implicator = LukasiewiczImplicator()
    logger = get_logger("test")
    return ITFRS(sim, lbl, tnorm, implicator, logger=logger)

def test_lower_approximation(model_instance):
    """
    @brief Test for `lower_approximation` method of ITFRS model.
    """
    lower = model_instance.lower_approximation()
    assert isinstance(lower, np.ndarray)
    assert lower.shape == (3,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))

def test_upper_approximation(model_instance):
    """
    @brief Test for `upper_approximation` method of ITFRS model.
    """
    upper = model_instance.upper_approximation()
    assert isinstance(upper, np.ndarray)
    assert upper.shape == (3,)
    assert np.all((0.0 <= upper) & (upper <= 1.0))

def test_boundary_region(model_instance):
    """
    @brief Test for `boundary_region` method of ITFRS model.
    """
    boundary = model_instance.boundary_region()
    expected = model_instance.upper_approximation() - model_instance.lower_approximation()
    np.testing.assert_allclose(boundary, expected)

def test_positive_region(model_instance):
    """
    @brief Test for `positive_region` method of ITFRS model.
    """
    pos = model_instance.positive_region()
    expected = model_instance.lower_approximation()
    np.testing.assert_allclose(pos, expected)

def test_to_dict_include_data(model_instance):
    """
    @brief Test for `to_dict_include_data` method of ITFRS model.
    """
    data = model_instance.to_dict(include_data=True)
    assert "type" in data
    assert "ub_tnorm" in data
    assert "lb_implicator" in data
    assert "similarity_matrix" in data
    assert "labels" in data

def test_to_dict_exclude_data(model_instance):
    """
    @brief Test for `to_dict_exclude_data` method of ITFRS model.
    """
    data = model_instance.to_dict(include_data=False)
    assert "type" in data
    assert "ub_tnorm" in data
    assert "lb_implicator" in data
    assert "similarity_matrix" not in data
    assert "labels" not in data

def test_from_dict_roundtrip(synthetic_data_):
    """
    @brief Test for `from_dict_roundtrip` method of ITFRS model.
    """
    sim, lbl = synthetic_data_
    model = ITFRS(sim, lbl, MinTNorm(), LukasiewiczImplicator())
    d = model.to_dict(include_data=True)
    restored = ITFRS.from_dict(d)
    np.testing.assert_array_equal(restored.similarity_matrix, model.similarity_matrix)
    np.testing.assert_array_equal(restored.labels, model.labels)
    np.testing.assert_array_equal(restored.lower_approximation(), model.lower_approximation())
    np.testing.assert_array_equal(restored.upper_approximation(), model.upper_approximation())

def test_from_config_equivalence(synthetic_data_):
    """
    @brief Test for `from_config_equivalence` method of ITFRS model.
    """
    sim, lbl = synthetic_data_
    config = {
        "ub_tnorm_name": "yager",
        "lb_implicator_name": "lukasiewicz",
        "p": 0.83
    }
    model = ITFRS.from_config(similarity_matrix=sim, labels=lbl, **config)
    assert isinstance(model, ITFRS)
    assert model.similarity_matrix.shape == (3, 3)
    assert model.labels.shape == (3,)

def test_describe_params_detailed(model_instance):
    """
    @brief Test for `describe_params_detailed` method of ITFRS model.
    """
    desc = model_instance.describe_params_detailed()
    assert isinstance(desc, dict)
    assert "ub_tnorm" in desc
    assert "lb_implicator" in desc

def test_get_params_internal(model_instance):
    """
    @brief Test for `get_params_internal` method of ITFRS model.
    """
    params = model_instance._get_params()
    assert "ub_tnorm" in params
    assert "lb_implicator" in params
    assert "similarity_matrix" in params
    assert "labels" in params

def test_validate_params_invalid_tnorm(synthetic_data_):
    """
    @brief Test for `validate_params_invalid_tnorm` method of ITFRS model.
    """
    sim, lbl = synthetic_data_
    with pytest.raises(ValueError):
        ITFRS.validate_params(lb_implicator=LukasiewiczImplicator(), ub_tnorm=None)

def test_validate_params_invalid_implicator(synthetic_data_):
    """
    @brief Test for `validate_params_invalid_implicator` method of ITFRS model.
    """
    sim, lbl = synthetic_data_
    with pytest.raises(ValueError):
        ITFRS.validate_params(lb_implicator=None, ub_tnorm=MinTNorm())

def test_logger_works(model_instance):
    """
    @brief Test for `logger_works` method of ITFRS model.
    """
    model_instance.logger.info("Logger test message")


#################################
###                           ###
###      Tests with data      ###
###                           ###
#################################

@pytest.mark.parametrize("test_case", ds.get_ITFRS_testing_testsets())
@pytest.mark.parametrize("implicator_name, expected_lower_key", [
    ("reichenbach", "Reichenbach_lowerBound"),
    ("kleenedienes", "KD_lowerBound"),
    ("lukasiewicz", "Luk_lowerBound"),
    ("goedel", "Goedel_lowerBound"),
    # ("gaines", "Gaines_lowerBound"),
    ("goguen", "Goguen_lowerBound"),
    ("rescher", "Rescher_lowerBound"),
    ("weber", "Weber_lowerBound"),
    ("fodor", "Fodor_lowerBound"),
    ("yager", "Yager_lowerBound")
])
@pytest.mark.parametrize("tnorm_name, expected_upper_key", [
    # ("product", "prod_tn_upperBound"),
    # ("minimum", "min_tn_upperBound"),
    # ("lukasiewicz", "luk_tn_upperBound"),
    # ("einstein", "einstein_tn_upperBound"),
    # ("drastic", "drastic_tn_upperBound"),
    # ("nilpotent", "nilpotent_tn_upperBound"),
    # ("hamacher", "hamacher_tn_upperBound"),
    ("yager", "yager_tn_upperBound_p_0_83")
])
def test_itfrs_model_with_all_settings(test_case, implicator_name, expected_lower_key, tnorm_name, expected_upper_key):
    """
    @brief Test for `itfrs_model_with_all_settings` method of ITFRS model.
    """
    sim = test_case["sim_matrix"]
    y = test_case["y"]
    expected = test_case["expected"]

    config={
        "ub_tnorm_name": tnorm_name,
        "lb_implicator_name": implicator_name,
        "p": 0.83
    }

    model = ITFRS.from_config(similarity_matrix=sim, labels=y, **config)

    actual_lower = model.lower_approximation()
    actual_upper = model.upper_approximation()

    np.testing.assert_allclose(actual_lower, expected[expected_lower_key], atol=1e-5, err_msg=f"Failed for {implicator_name}")
    np.testing.assert_allclose(actual_upper, expected[expected_upper_key], atol=1e-5, err_msg=f"Failed for {tnorm_name}")

@pytest.fixture
def synthetic_data():
    return ds.get_ITFRS_testing_testsets()[0]

@pytest.mark.parametrize("implicator_name", ['reichenbach', 'kleenedienes', 'lukasiewicz', 'goedel', 'goguen'])
@pytest.mark.parametrize("tnorm_name", ['product', 'minimum'])
@pytest.mark.parametrize("similarity_name", ['gaussian', 'linear'])
def test_itfrs_all_combinations(implicator_name, tnorm_name, similarity_name, synthetic_data):
    sim_matrix_raw = synthetic_data["sim_matrix"]
    labels = synthetic_data["y"]
    expected = synthetic_data["expected"]

    
    sim_matrix = sim_matrix_raw

    model = ITFRS.from_config(
        similarity_matrix=sim_matrix,
        labels=labels,
        lb_implicator_name=implicator_name,
        ub_tnorm_name=tnorm_name
    )

    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == labels.shape
    assert upper.shape == labels.shape
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    expected_lower_keys = {
        'reichenbach': "Reichenbach_lowerBound",
        'kleene-dienes': "KD_lowerBound",
        'lukasiewicz': "Luk_lowerBound",
        'goedel': "Goedel_lowerBound",
        'gaines': "Gaines_lowerBound"
    }
    expected_upper_keys = {
        'product': "prod_tn_upperBound",
        'minimum': "min_tn_upperBound"
    }

    if implicator_name in expected_lower_keys:
        expected_key = expected_lower_keys[implicator_name]
        if expected_key in expected:
            np.testing.assert_allclose(lower, expected[expected_key], atol=1e-5)

    if tnorm_name in expected_upper_keys:
        expected_key = expected_upper_keys[tnorm_name]
        if expected_key in expected:
            np.testing.assert_allclose(upper, expected[expected_key], atol=1e-5)

def test_logger_and_params_describe(synthetic_data):
    sim_matrix = synthetic_data["sim_matrix"]
    labels = synthetic_data["y"]
    model = ITFRS.from_config(similarity_matrix=sim_matrix, labels=labels,
                              lb_implicator_name="lukasiewicz", ub_tnorm_name="minimum")
    params = model.describe_params_detailed()
    assert "ub_tnorm" in params
    assert "lb_implicator" in params
    model.logger.info("Logger test message")

def test_to_dict_and_from_dict_roundtrip(synthetic_data):
    sim_matrix = synthetic_data["sim_matrix"]
    labels = synthetic_data["y"]
    model = ITFRS.from_config(similarity_matrix=sim_matrix, labels=labels,
                              lb_implicator_name="lukasiewicz", ub_tnorm_name="minimum")
    model_dict = model.to_dict(include_data=True)
    reconstructed = ITFRS.from_dict(model_dict)
    np.testing.assert_allclose(reconstructed.lower_approximation(), model.lower_approximation())
    np.testing.assert_allclose(reconstructed.upper_approximation(), model.upper_approximation())    