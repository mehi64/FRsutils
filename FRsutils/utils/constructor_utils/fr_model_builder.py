"""
@file
@brief Factory functions for building fuzzy-rough models by name.

@details These functions instantiate fuzzy-rough model objects using validated parameters.
They decouple model construction logic from the model class definitions.
"""

from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.models.vqrs import VQRS
from FRsutils.core.approximations import BaseFuzzyRoughModel
from FRsutils.utils.constructor_utils.tnorm_builder import build_tnorm
from FRsutils.utils.constructor_utils.function_registry import IMPLICATOR_REGISTRY

def build_fuzzy_rough_model(model_name: str, similarity_matrix, labels, fr_model_params: dict) -> BaseFuzzyRoughModel:
    """
    @brief Factory to instantiate a fuzzy-rough model from its name and validated parameters.

    @param model_name Name of the fuzzy-rough model ('ITFRS', 'OWAFRS', or 'VQRS').
    @param fr_model_params Dictionary of constructor parameters required by the model.
           Assumes values are already validated externally.

    @return Instance of FuzzyRoughModel_Base subclass.

    @throws ValueError If the model_name is not recognized.
    """
    if model_name == 'ITFRS':
        return ITFRS(
            similarity_matrix=similarity_matrix,
            labels=labels,
            tnorm=build_tnorm(fr_model_params['lb_tnorm']),
            implicator=IMPLICATOR_REGISTRY[fr_model_params['ub_implicator']]
        )
    
    elif model_name == 'OWAFRS':
        return OWAFRS(
            similarity_matrix=similarity_matrix,
            labels=labels,
            tnorm=fr_model_params['lb_tnorm'],
            implicator=fr_model_params['ub_implicator'],
            owa_weighting_strategy=fr_model_params['owa_weighting_strategy']
        )
    
    elif model_name == 'VQRS':
        return VQRS(
            similarity_matrix=similarity_matrix,
            labels=labels,
            alpha_Q_lower=fr_model_params['alpha_Q_lower'],
            beta_Q_lower=fr_model_params['beta_Q_lower'],
            alpha_Q_upper=fr_model_params['alpha_Q_upper'],
            beta_Q_upper=fr_model_params['beta_Q_upper']
        )
    
    else:
        raise ValueError(f"Unsupported fuzzy rough model name: '{model_name}'")
