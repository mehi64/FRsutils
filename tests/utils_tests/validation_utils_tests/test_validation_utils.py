import pytest
from FRsutils.utils.validation_utils import (
    validate_choice,
    validate_strategy_compatibility,
    validate_fr_model_params,
    ALLOWED_FR_MODELS,
    ALLOWED_SIMILARITIES,
    ALLOWED_TNORMS,
    ALLOWED_RANKING_STRATEGIES
)

# ------------------------------
# validate_choice
# ------------------------------

def test_validate_choice_valid():
    """
    @brief Tests that validate_choice returns the value when it's valid.
    """
    assert validate_choice("fr_model_name", "ITFRS", ALLOWED_FR_MODELS) == "ITFRS"

def test_validate_choice_invalid():
    """
    @brief Tests that validate_choice raises ValueError for invalid values.
    """
    with pytest.raises(ValueError):
        validate_choice("similarity_name", "weird_sim", ALLOWED_SIMILARITIES)

# ------------------------------
# validate_strategy_compatibility
# ------------------------------

def test_strategy_compatibility_valid():
    """
    @brief Tests that valid sampling strategies are accepted.
    """
    validate_strategy_compatibility("MySampler", "auto", {"auto", "balanced"})

def test_strategy_compatibility_invalid():
    """
    @brief Tests that an invalid strategy raises a ValueError.
    """
    with pytest.raises(ValueError):
        validate_strategy_compatibility("MySampler", "random", {"auto", "balanced"})

# ------------------------------
# ITFRS
# ------------------------------

def test_itfrs_valid():
    """
    @brief Tests correct ITFRS parameter configuration.
    """
    validate_fr_model_params("ITFRS", {
        "lb_tnorm": lambda a, b: a * b,
        "ub_implicator": lambda a, b: max(1 - a, b)
    })

def test_itfrs_missing_param():
    """
    @brief Tests that missing required ITFRS parameters raise an error.
    """
    with pytest.raises(ValueError):
        validate_fr_model_params("ITFRS", {
            "lb_tnorm": lambda a, b: a * b
        })

def test_itfrs_invalid_type():
    """
    @brief Tests that non-callable ITFRS parameters raise a TypeError.
    """
    with pytest.raises(TypeError):
        validate_fr_model_params("ITFRS", {
            "lb_tnorm": None,
            "ub_implicator": 123
        })

# ------------------------------
# OWAFRS
# ------------------------------

def test_owafrs_valid():
    """
    @brief Tests valid OWAFRS parameter configuration.
    """
    validate_fr_model_params("OWAFRS", {
        "lb_tnorm": lambda a, b: a * b,
        "ub_implicator": lambda a, b: max(1 - a, b),
        "owa_weighting_strategy": "linear"
    })

def test_owafrs_invalid_strategy_string():
    """
    @brief Tests that an invalid owa_weighting_strategy raises ValueError.
    """
    with pytest.raises(ValueError):
        validate_fr_model_params("OWAFRS", {
            "lb_tnorm": lambda a, b: a * b,
            "ub_implicator": lambda a, b: max(1 - a, b),
            "owa_weighting_strategy": "custom"
        })

def test_owafrs_missing_strategy():
    """
    @brief Tests that missing 'owa_weighting_strategy' raises an error.
    """
    with pytest.raises(ValueError):
        validate_fr_model_params("OWAFRS", {
            "lb_tnorm": lambda a, b: a * b,
            "ub_implicator": lambda a, b: max(1 - a, b)
        })

# ------------------------------
# VQRS
# ------------------------------

def test_vqrs_valid():
    """
    @brief Tests valid float parameter configuration for VQRS.
    """
    validate_fr_model_params("VQRS", {
        "alpha_Q_lower": 0.1,
        "beta_Q_lower": 0.4,
        "alpha_Q_upper": 0.6,
        "beta_Q_upper": 0.9
    })

def test_vqrs_missing_param():
    """
    @brief Tests that missing float parameters raise ValueError.
    """
    with pytest.raises(ValueError):
        validate_fr_model_params("VQRS", {
            "alpha_Q_lower": 0.1,
            "beta_Q_lower": 0.4,
            "alpha_Q_upper": 0.6
        })

def test_vqrs_type_error():
    """
    @brief Tests that non-float types raise TypeError for VQRS.
    """
    with pytest.raises(TypeError):
        validate_fr_model_params("VQRS", {
            "alpha_Q_lower": "0.1",
            "beta_Q_lower": 0.4,
            "alpha_Q_upper": 0.6,
            "beta_Q_upper": 0.9
        })

def test_vqrs_range_error():
    """
    @brief Tests that out-of-range float values raise ValueError.
    """
    with pytest.raises(ValueError):
        validate_fr_model_params("VQRS", {
            "alpha_Q_lower": -0.2,
            "beta_Q_lower": 1.5,
            "alpha_Q_upper": 0.6,
            "beta_Q_upper": 0.9
        })

# ------------------------------
# Unknown model
# ------------------------------

def test_unknown_model():
    """
    @brief Tests that an unknown model name raises ValueError.
    """
    with pytest.raises(ValueError):
        validate_fr_model_params("NotARealModel", {})
