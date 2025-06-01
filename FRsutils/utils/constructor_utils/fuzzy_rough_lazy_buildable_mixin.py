
"""
@file
@brief Mixin class that adds lazy building logic for fuzzy-rough models based on configuration.
"""

from abc import ABC
from FRsutils.utils.validation_utils import (
    validate_fr_model_params,
    _validate_string_param_choice,
    ALLOWED_FR_MODELS, ALLOWED_SIMILARITIES, ALLOWED_TNORMS
)
from FRsutils.utils.constructor_utils.tnorm_builder import build_tnorm
from FRsutils.utils.constructor_utils.similarity_builder import build_similarity
from FRsutils.utils.constructor_utils.fr_model_builder import build_fuzzy_rough_model
from FRsutils.core.similarities import calculate_similarity_matrix

class FuzzyRoughLazyBuildableMixin(ABC):
    """
    @brief Mixin for fuzzy-rough based oversamplers or estimators to support lazy model building.

    @details This mixin handles:
     - Storing configuration params
     - Validating parameters
     - Lazily building the fuzzy-rough model and similarity matrix only when needed
    """

    def _initialize_fr_config(self,
                              fr_model_name,
                              similarity_name,
                              similarity_tnorm_name,
                              fr_model_params):
        """
        @brief Initializes and validates fuzzy rough configuration.

        @param fr_model_name Name of fuzzy rough model.
        @param similarity_name Name of similarity function.
        @param similarity_tnorm_name Name of t-norm.
        @param fr_model_params Dict of model parameters.
        """
        self.fr_model_name = _validate_string_param_choice("fr_model_name", fr_model_name, ALLOWED_FR_MODELS)
        self.similarity_name = _validate_string_param_choice("similarity_name", similarity_name, ALLOWED_SIMILARITIES)
        self.similarity_tnorm_name = _validate_string_param_choice("similarity_tnorm", similarity_tnorm_name, ALLOWED_TNORMS)

        validate_fr_model_params(self.fr_model_name, fr_model_params)
        self.fr_model_params = fr_model_params
        self._is_built = False

    def _build_internal_objects(self, X, y):
        """
        @brief Constructs the similarity function, and similarity t-norm, and similarity matrix with 
        them. Plus the fuzzy rough model instance.

        @details Should be called before using self.fr_model. Sets _is_built = True.
        """
        similarity_func = build_similarity(self.similarity_name)
        similarity_tnorm = build_tnorm(self.similarity_tnorm_name)

        self.similarity_matrix = calculate_similarity_matrix(X, similarity_func, similarity_tnorm)

        self.fr_model = build_fuzzy_rough_model(model_name=self.fr_model_name,
                                                similarity_matrix=self.similarity_matrix,
                                                labels=y,
                                                fr_model_params= self.fr_model_params)

        self.lower_app = self.fr_model.lower_approximation()
        self.upper_app = self.fr_model.upper_approximation()

        # # TODO: check this. Is that correct?what about all models?
        self.positive_region = self.lower_app
        
        self._is_built = True

    def ensure_built(self, X, y):
        """
        @brief Public method to build internal components if not already built.
        """
        if not getattr(self, '_is_built', False):
            self._build_internal_objects(X, y)