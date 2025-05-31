"""
@file
@brief Utility to auto-generate parameter schemas using class __init__ signatures and type hints.
"""

import inspect
from typing import get_type_hints, Union

def infer_param_schema_from_init(cls) -> dict:
    """
    @brief Infers a parameter schema from a class's __init__ method using type hints and defaults.

    @param cls The class to analyze.
    @return A dictionary schema where each key is a parameter and each value is a spec dict.
    """
    signature = inspect.signature(cls.__init__)
    type_hints = get_type_hints(cls.__init__)
    schema = {}

    for name, param in signature.parameters.items():
        if name == 'self':
            continue

        param_type = type_hints.get(name, None)
        has_default = param.default is not inspect.Parameter.empty

        param_schema = {
            'type': _type_name(param_type),
            'required': not has_default
        }

        if has_default:
            param_schema['default'] = param.default

        schema[name] = param_schema

    return schema


def _type_name(py_type: Union[type, None]) -> str:
    """
    @brief Returns a string representation of a Python type.

    @param py_type The Python type object (e.g., float, int).
    @return String name for schema.
    """
    if py_type is None:
        return 'unknown'
    if hasattr(py_type, '__name__'):
        return py_type.__name__
    return str(py_type)
