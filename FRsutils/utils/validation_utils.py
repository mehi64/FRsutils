"""
@file
@brief Validation utilities for fuzzy rough oversamplers and models.

@details This module provides:
- Choice validation for string parameters.
- Strategy compatibility checks.
- Schema-based validation for fuzzy rough model parameters.
"""

# ------------------------------
# Allowed values for string parameters
# ------------------------------

ALLOWED_FR_MODELS = {'ITFRS', 'VQRS', 'OWAFRS'}
ALLOWED_SIMILARITIES = {'linear', 'gaussian'}
ALLOWED_TNORMS = {'lukasiewicz', 'product', 'minimum'}
ALLOWED_IMPLICATORS = {'goedel', 'kleene_dienes', 'lukasiewicz', 'reichenbach'}
ALLOWED_FUZZY_QUANTIFIERS = {'linear', 'quad'}
ALLOWED_OWA_WEIGHTING_STRATEGIES = {'linear_sup', 'linear_inf'}

ALLOWED_RANKING_STRATEGIES = {'pos', 'lower', 'upper'}

# ------------------------------
# validate_choice
# ------------------------------

def validate_choice(name: str, value: str, allowed: set[str]) -> str:
    """
    @brief Validates a string parameter against allowed values.

    @param name Parameter name (for error reporting).
    @param value Actual user-provided value.
    @param allowed Allowed set of values.

    @return The same value if valid.

    @throws ValueError If value is not allowed.
    """
    if value not in allowed:
        raise ValueError(
            f"Invalid value '{value}' for parameter '{name}'. "
            f"Allowed values are: {sorted(allowed)}."
        )
    return value

# ------------------------------
# validate_strategy_compatibility
# ------------------------------

def validate_strategy_compatibility(class_name: str, strategy: str, supported_strategies: set[str]):
    """
    @brief Validates whether a sampling strategy is compatible with a class.

    @param class_name Name of the oversampler class.
    @param strategy Sampling strategy to check.
    @param supported_strategies Set of valid strategies.

    @throws ValueError If strategy is not supported.
    """
    if strategy not in supported_strategies:
        raise ValueError(
            f"Sampling strategy '{strategy}' is not supported by '{class_name}'. "
            f"Supported strategies are: {sorted(supported_strategies)}."
        )

# ------------------------------
# validate_fr_model_params
# ------------------------------

def validate_fr_model_params(model_name: str, params: dict):
    """
    @brief Validates parameters passed to a fuzzy rough model against its schema.

    @param model_name Name of the fuzzy rough model.
    @param params Dictionary of user-provided parameters.

    @throws ValueError or TypeError If keys are missing, types are incorrect, or values are invalid.
    """
    schema = _get_fr_model_param_schema(model_name)

    for key, spec in schema.items():
        # Check presence
        if spec.get('required', False) and key not in params:
            raise ValueError(f"{model_name} requires parameter '{key}'.")

        val = params.get(key)

        # Type: float
        if spec['type'] == 'float':
            if not isinstance(val, float):
                raise TypeError(f"Parameter '{key}' must be a float.")
            lo, hi = spec.get('range', (None, None))
            if lo is not None and (val < lo or val > hi):
                raise ValueError(f"Parameter '{key}' must be in range [{lo}, {hi}].")

        # Type: str with allowed values
        elif spec['type'] == 'str':
            if not isinstance(val, str):
                raise TypeError(f"Parameter '{key}' must be a string.")
            if 'allowed' in spec and val not in spec['allowed']:
                raise ValueError(f"Parameter '{key}' must be one of {sorted(spec['allowed'])}.")

        # Type: tnorm / implicator should be callable
        elif spec['type'] in {'tnorm', 'implicator'}:
            if not callable(val):
                raise TypeError(f"Parameter '{key}' must be a callable (type={spec['type']}).")


def validate_tnorm_params(name: str, params: dict):
    """
    @brief Validates parameters provided for a named T-norm.

    @param name Name of the T-norm.
    @param params Parameter dictionary to validate.

    @throws ValueError or TypeError for missing or invalid values.
    """
    schema = _get_tnorm_param_schema(name)

    for key, spec in schema.items():
        if spec.get('required', False) and key not in params:
            raise ValueError(f"T-norm '{name}' requires parameter '{key}'.")

        val = params.get(key)

        if spec['type'] == 'float':
            if not isinstance(val, float):
                raise TypeError(f"Parameter '{key}' must be a float.")
            lo, hi = spec.get('range', (None, None))
            if lo is not None and (val < lo or val > hi):
                raise ValueError(f"Parameter '{key}' must be in range [{lo}, {hi}].")


def validate_similarity_choice(name: str) -> str:
    """
    @brief Validates a similarity function name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_SIMILARITIES.
    """
    return validate_choice("similarity_name", name, ALLOWED_SIMILARITIES)


def validate_implicator_choice(name: str) -> str:
    """
    @brief Validates an implicator name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_IMPLICATORS.
    """
    return validate_choice("implicator", name, ALLOWED_IMPLICATORS)


def validate_quantifier_choice(name: str) -> str:
    """
    @brief Validates a fuzzy quantifier name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_FUZZY_QUANTIFIERS.
    """
    return validate_choice("fuzzy_quantifier", name, ALLOWED_FUZZY_QUANTIFIERS)


def validate_owa_strategy_choice(name: str) -> str:
    """
    @brief Validates an OWA weighting strategy name.

    @param name Name to check.
    @return The name if valid.

    @throws ValueError If not in ALLOWED_OWA_WEIGHTING_STRATEGIES.
    """
    return validate_choice("owa_weighting_strategy", name, ALLOWED_OWA_WEIGHTING_STRATEGIES)

##############################################################################


# ------------------------------
# get_fr_model_param_schema
# ------------------------------

def _get_fr_model_param_schema(model_name: str) -> dict:
    """
    @brief Returns a schema defining the parameters required for a fuzzy rough model.

    @param model_name Name of the fuzzy rough model.

    @return Schema dictionary with keys as param names and values as specs.

    @throws ValueError If the model name is unknown.
    """
    if model_name == 'ITFRS':
        return {
            'lb_tnorm': {'type': 'tnorm', 'required': True},
            'ub_implicator': {'type': 'implicator', 'required': True}
        }
    elif model_name == 'OWAFRS':
        return {
            'lb_tnorm': {'type': 'tnorm', 'required': True},
            'ub_implicator': {'type': 'implicator', 'required': True},
            'owa_weighting_strategy': {
                'type': 'str', 'required': True, 'allowed': {'linear', 'quad'}
            }
        }
    elif model_name == 'VQRS':
        return {
            'alpha_Q_lower': {'type': 'float', 'required': True, 'range': (0.0, 1.0)},
            'beta_Q_lower': {'type': 'float', 'required': True, 'range': (0.0, 1.0)},
            'alpha_Q_upper': {'type': 'float', 'required': True, 'range': (0.0, 1.0)},
            'beta_Q_upper': {'type': 'float', 'required': True, 'range': (0.0, 1.0)}
        }
    else:
        raise ValueError(f"Unknown fuzzy rough model '{model_name}'.")


# ------------------------------
# T-norm Parameter Schema
# ------------------------------

def _get_tnorm_param_schema(name: str) -> dict:
    """
    @brief Returns the expected parameter schema for the given T-norm.

    @param name Name of the T-norm ('minimum', 'product', 'lukasiewicz', etc.)
    @return Dictionary of expected parameters and specifications.
            If the T-norm does not require parameters, an empty dict is returned.

    @throws ValueError If the T-norm name is not supported.
    """
    name = name.lower()

    if name in {'minimum', 'product', 'lukasiewicz'}:
        return {}  # No parameters required

    # elif name == 'yager':
    #     return {
    #         'p': {'type': 'float', 'required': True, 'range': (1.0, float('inf'))}
    #     }

    else:
        raise ValueError(f"Unknown or unsupported T-norm: '{name}'")

# ------------------------------
# implicator Parameter Schema
# ------------------------------

def _get_implicator_param_schema(name: str) -> dict:
    """
    @brief Returns the parameter schema for a named fuzzy implicator.

    @param name Name of the implicator function (e.g., 'goedel', 'lukasiewicz', etc.).
    @return A dictionary describing the expected parameters. Most are parameterless.

    @throws ValueError If the implicator name is not supported.
    """
    name = name.lower()

    if name in {'goedel', 'kleene_dienes', 'lukasiewicz', 'reichenbach'}:
        return {}  # All current implicators are parameterless

    # Future support for parameterized implicators can go here
    else:
        raise ValueError(f"Unknown or unsupported implicator: '{name}'")

# ------------------------------
# similarity Parameter Schema
# ------------------------------

def _get_similarity_param_schema(name: str) -> dict:
    """
    @brief Returns the parameter schema for a similarity function.

    @param name Name of the similarity function (e.g., 'linear', 'gaussian').

    @return Dictionary describing the required parameters.

    @throws ValueError If similarity function name is unknown.
    """
    name = name.lower()
    if name == 'linear':
        return {}
    elif name == 'gaussian':
        return {
            'sigma': {'type': 'float', 'required': True, 'range': (0.0, float('inf'))}
        }
    else:
        raise ValueError(f"Unknown similarity function: '{name}'")

# ------------------------------
# OWA weights Parameter Schema
# ------------------------------

def _get_owa_weight_param_schema(name: str) -> dict:
    """
    @brief Returns the parameter schema for an OWA weighting strategy.

    @param name Name of the OWA strategy ('linear_sup', 'linear_inf').

    @return Parameter schema dict. All current strategies require `n` as int > 0.

    @throws ValueError If strategy name is unknown.
    """
    name = name.lower()
    if name in {'linear_sup', 'linear_inf'}:
        return {
            'n': {'type': 'int', 'required': True, 'range': (1, float('inf'))}
        }
    else:
        raise ValueError(f"Unknown OWA weighting strategy: '{name}'")

# ------------------------------
# Fuzzy Quantifier Parameter Schema
# ------------------------------

def _get_fuzzy_quantifier_param_schema(name: str) -> dict:
    """
    @brief Returns the parameter schema for a fuzzy quantifier.

    @param name Name of the quantifier ('linear', 'quad').

    @return Dictionary with parameter specs.

    @throws ValueError If the quantifier name is not recognized.
    """
    name = name.lower()
    if name == 'linear':
        return {
            'alpha': {'type': 'float', 'required': True, 'range': (0.0, 1.0)},
            'beta':  {'type': 'float', 'required': True, 'range': (0.0, 1.0)},
            'increasing': {'type': 'bool', 'required': False}
        }
    elif name == 'quad':
        return {
            'alpha': {'type': 'float', 'required': True, 'range': (0.0, 1.0)},
            'beta':  {'type': 'float', 'required': True, 'range': (0.0, 1.0)}
        }
    else:
        raise ValueError(f"Unknown fuzzy quantifier: '{name}'")


# ------------------------------
# unified Dipatcher for all Parameter Schemas
# ------------------------------

def get_param_schema(component_type: str, name: str) -> dict:
    """
    @brief Returns the parameter schema for any supported component.

    @param component_type One of ['fr_model', 'tnorm', 'implicator', 'similarity', 'owa_weight', 'fuzzy_quantifier'].
    @param name The component name (e.g., 'ITFRS', 'linear', 'lukasiewicz').

    @return A dictionary schema describing the required/optional parameters.

    @throws ValueError If the component_type or name is not supported.
    """
    component_type = component_type.lower()

    if component_type == 'fr_model':
        return _get_fr_model_param_schema(name)
    elif component_type == 'tnorm':
        return _get_tnorm_param_schema(name)
    elif component_type == 'implicator':
        return _get_implicator_param_schema(name)
    elif component_type == 'similarity':
        return _get_similarity_param_schema(name)
    elif component_type == 'owa_weight':
        return _get_owa_weight_param_schema(name)
    elif component_type == 'fuzzy_quantifier':
        return _get_fuzzy_quantifier_param_schema(name)
    else:
        raise ValueError(f"Unsupported component_type '{component_type}'.")
