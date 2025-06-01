"""
@file
@brief Factory function to build T-norm instances from string names.
"""

from FRsutils.core.tnorms import MinTNorm, ProductTNorm, LukasiewiczTNorm, TNorm
from FRsutils.utils.validation_utils import (
    _validate_string_param_choice,
    ALLOWED_TNORMS
)

def build_tnorm(name: str) -> TNorm:
    """
    @brief Instantiates a T-norm object based on its name.

    @param name Name of the T-norm ('minimum', 'product', 'lukasiewicz').

    @return An instance of a subclass of TNorm.

    @throws ValueError If the name is not recognized.
    """
    name = _validate_string_param_choice("tnorm", name, ALLOWED_TNORMS)

    if name == 'minimum':
        return MinTNorm()
    elif name == 'product':
        return ProductTNorm()
    elif name == 'lukasiewicz':
        return LukasiewiczTNorm()
    else:
        raise ValueError(f"Unknown T-norm: {name}")
