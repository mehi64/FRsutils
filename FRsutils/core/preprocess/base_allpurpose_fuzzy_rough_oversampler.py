import warnings
from imblearn.over_sampling.base import BaseOverSampler
from abc import ABC, abstractmethod
from FRsutils.utils.init_helpers import assign_allowed_kwargs
from FRsutils.utils.validation_utils import get_fr_model_param_schema
from FRsutils.utils.constructor_utils.fuzzy_rough_lazy_buildable_mixin import FuzzyRoughLazyBuildableMixin

from FRsutils.utils.validation_utils import (
    _validate_string_param_choice,
    validate_fr_model_params,
    get_similarity_param_schema,
    ALLOWED_RANKING_STRATEGIES
)

class BaseAllPurposeFuzzyRoughOversampler(FuzzyRoughLazyBuildableMixin, ABC, BaseOverSampler):
    """
    @brief Abstract base class for oversampling using Fuzzy Rough Sets.

    @details This base class is intended to be inherited by resamplers that either:
     - Use fuzzy-rough set theory directly for ranking or selecting instances.
     - Combine fuzzy-rough logic with generative models (e.g., VAE, GAN).
     - Use fuzzy-rough sets as a preprocessing step for other resamplers.

    It should not be used directly.

    @warning Do not instantiate or use this class directly. Use one of its concrete subclasses instead.
    """

    def __init__(self,
                 fr_model_name='ITFRS',
                 similarity_name='linear',
                 similarity_tnorm_name='minimum',
                 instance_ranking_strategy_name='pos',
                 sampling_strategy='auto',
                 **kwargs):

        super().__init__(sampling_strategy=sampling_strategy)

        # Validate and store fuzzy rough config parameters
        validate_fr_model_params(fr_model_name, kwargs)
        fr_model_schema = get_fr_model_param_schema(fr_model_name)
        assign_allowed_kwargs(self, kwargs, fr_model_schema)
        
        # similarity_name_schema = get_similarity_param_schema(similarity_name)
        # assign_allowed_kwargs(self, kwargs, similarity_name_schema)

        self._initialize_fr_config(
            fr_model_name=fr_model_name,
            similarity_name=similarity_name,
            similarity_tnorm_name=similarity_tnorm_name,
            fr_model_params=kwargs
        )

        self.instance_ranking_strategy_name = _validate_string_param_choice(
            "instance_ranking_strategy", instance_ranking_strategy_name, ALLOWED_RANKING_STRATEGIES)

# _get_target_classes and _get_num_samples remain the same as in FRSMOTE
    def _get_target_classes(self):
        """Determine which classes to oversample based on sampling_strategy."""
        
        if self.instance_ranking_strategy_name == 'pos':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        elif isinstance(self.instance_ranking_strategy_name, dict):
            return list(self.instance_ranking_strategy_name.keys())
        # Add more strategy handling if needed (float, list, callable)
        else:
            warnings.warn(f"Unsupported sampling_strategy: {self.instance_ranking_strategy}. Using 'auto'.")
            return [cls for cls in self.classes_ if cls != max(self.target_stats_, key=self.target_stats_.get)]

    def _get_num_samples(self, class_label):
        """Determine number of samples to generate for a class."""
        if self.instance_ranking_strategy_name == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
        elif isinstance(self.instance_ranking_strategy_name, dict):
            # Ensure target count is not less than current count
            target_count = max(self.target_stats_[class_label], self.instance_ranking_strategy_name[class_label])
        else: # Default to balancing against majority if strategy is unclear
             warnings.warn(f"Interpreting sampling_strategy '{self.instance_ranking_strategy_name}' as 'auto'.")
             majority_class = max(self.target_stats_, key=self.target_stats_.get)
             target_count = self.target_stats_[majority_class]

        current_count = self.target_stats_[class_label]
        return max(0, target_count - current_count)
    
        
    @abstractmethod
    def _check_params(self):
        """
        checks correctness of parameters specific to this object.
        Each derived class must implements its own
        """
        raise NotImplementedError("Subclasses must implement _check_params.")
        
    @abstractmethod 
    def fit_resample(self, X, y):
        """
        @brief Fits the resampler to the data and returns the resampled data.
        all classes should implement this method
        
        @param X The input data.
        @param y The target labels.
        @return A tuple of the resampled data and labels.
        """
        raise NotImplementedError("Subclasses must implement fit_resample.")
 