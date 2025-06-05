"""
@file
@brief Validation utilities for fuzzy rough oversamplers and models.

@details This module provides:
- Choice validation for string parameters.
- Strategy compatibility checks.
- Schema-based validation for fuzzy rough model parameters.
"""

import numpy as np

# ------------------------------
# Allowed values for string parameters
# ------------------------------

ALLOWED_FR_MODELS = {'ITFRS', 'VQRS', 'OWAFRS'}
ALLOWED_SIMILARITIES = {'linear', 'gaussian'}
ALLOWED_TNORMS = {'lukasiewicz', 'product', 'minimum'}
ALLOWED_IMPLICATORS = {'gaines', 'goedel', 'kleene_dienes', 'lukasiewicz', 'reichenbach'}
ALLOWED_FUZZY_QUANTIFIERS = {'linear', 'quadratic'}
ALLOWED_OWA_WEIGHTING_STRATEGIES = {'linear'}
ALLOWED_RANKING_STRATEGIES = {'pos', 'lower', 'upper'}

# ------------------------------
# validate_choice
# ------------------------------

def _validate_string_param_choice(param_name: str, 
                                  param_value: str, 
                                  allowed: set[str]) -> str:
    """
    @brief Validates a string parameter against allowed values.

    @param name Parameter name (for error reporting).
    @param value Actual user-provided value.
    @param allowed Allowed set of values.

    @return The same value if valid.

    @throws ValueError If value is not allowed.
    """
    if param_value not in allowed:
        raise ValueError(
            f"Invalid value '{param_value}' for parameter '{param_name}'. "
            f"Allowed values are: {sorted(allowed)}."
        )
    return param_value

# ------------------------------
# validate_strategy_compatibility
# ------------------------------


def validate_ranking_strategy_choice(name: str):
    """
    @brief Validates a ranking strategy choice.
    """
    return _validate_string_param_choice("ranking_strategy", 
                                         name, 
                                         ALLOWED_RANKING_STRATEGIES)

# ------------------------------
# validate_fr_model_params
# ------------------------------

def validate_fr_model_choice(name: str):
    """
    @brief Validates a fuzzy rough model choice.
    """
    return _validate_string_param_choice("fuzzy-rough_model", 
                                         name, 
                                         ALLOWED_FR_MODELS)

def validate_tnorm_choice(name: str):
    """
    @brief Validates a T-norm choice.
    """
    return _validate_string_param_choice("t-norm", 
                                         name, 
                                         ALLOWED_TNORMS)

def validate_similarity_choice(name: str) -> str:
    """
    @brief Validates a similarity function name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_SIMILARITIES.
    """
    return _validate_string_param_choice("similarity_name", 
                                         name, 
                                         ALLOWED_SIMILARITIES)


def validate_implicator_choice(name: str) -> str:
    """
    @brief Validates an implicator name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_IMPLICATORS.
    """
    return _validate_string_param_choice("implicator", 
                                         name, 
                                         ALLOWED_IMPLICATORS)


def validate_fuzzy_quantifier_choice(name: str) -> str:
    """
    @brief Validates a fuzzy quantifier name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_FUZZY_QUANTIFIERS.
    """
    return _validate_string_param_choice("fuzzy_quantifier", 
                                         name, 
                                         ALLOWED_FUZZY_QUANTIFIERS)


def validate_owa_weighting_strategy_choice(name: str) -> str:
    """
    @brief Validates an OWA weighting strategy name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_OWA_WEIGHTING_STRATEGIES.
    """
    return _validate_string_param_choice("owa_weighting_strategy", 
                                        name, 
                                        ALLOWED_OWA_WEIGHTING_STRATEGIES)

# TODO: check correctness
def validate_range_0_1(x, name="value"):
    
    if isinstance(x, float):
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"{name} must be in range [0.0, 1.0], but got {x}")
    elif isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"{name} must be an array of floats")
        if np.any(x < 0.0) or np.any(x > 1.0):
            raise ValueError(f"All elements of {name} must be in range [0.0, 1.0]")
    else:
        raise TypeError(f"{name} must be a float or a numpy.ndarray, but got {type(x).__name__}")

    return x
##############################################################################


