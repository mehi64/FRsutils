"""
@file
@brief Factory function to build similarity functions by name.
"""

from FRsutils.core.similarities import LinearSimilarity, GaussianSimilarity, SimilarityFunction
from FRsutils.utils.validation_utils import (
    validate_similarity_choice,
    get_param_schema
)

def build_similarity(name: str, **kwargs) -> SimilarityFunction:
    """
    @brief Instantiates a similarity function object.

    @param name Name of the similarity function ('linear', 'gaussian').
    @param kwargs Optional keyword args (e.g., sigma for GaussianSimilarity)

    @return Instance of SimilarityFunction.

    @throws ValueError If the name or arguments are invalid.
    """
    name = validate_similarity_choice(name)
    schema = get_param_schema('similarity', name)

    # Validate and extract parameters
    validated_kwargs = {}
    for param, spec in schema.items():
        if spec.get('required', False) and param not in kwargs:
            raise ValueError(f"Missing required parameter '{param}' for similarity '{name}'.")

        value = kwargs.get(param, spec.get('default'))

        if value is None:
            raise ValueError(f"Parameter '{param}' must be provided.")

        if spec['type'] == 'float':
            if not isinstance(value, float):
                raise TypeError(f"Parameter '{param}' must be a float.")
            lo, hi = spec.get('range', (None, None))
            if lo is not None and not (lo <= value <= hi):
                raise ValueError(f"Parameter '{param}' must be in range [{lo}, {hi}].")

        validated_kwargs[param] = value

    # Instantiate the similarity object
    if name == 'linear':
        return LinearSimilarity()
    elif name == 'gaussian':
        return GaussianSimilarity(**validated_kwargs)
    else:
        raise ValueError(f"Unknown similarity function: {name}")
