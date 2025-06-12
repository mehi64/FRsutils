"""
@file base_allpurpose_fuzzy_rough_oversampler.py
@brief Abstract base class for oversampling using Fuzzy Rough Sets.

Provides lazy instantiation and configuration support via LazyBuildableFromConfigMixin.

##############################################
# ✅ Features
# - Stores all fuzzy-rough config lazily via kwargs
# - Defers model building until `ensure_built(X, y)` is called
# - Compatible with any registered fuzzy rough model via .from_config()

# ✅ Design Principles
# - Mixin Reuse: uses LazyBuildableFromConfigMixin
# - Fail-fast validation via validation_utils
# - Open/Closed: supports all fuzzy rough models and extensions
##############################################
"""

import warnings
from imblearn.over_sampling.base import BaseOverSampler
from abc import ABC, abstractmethod
from FRsutils.utils.constructor_utils.lazy_buildable_from_config_mixin import LazyBuildableFromConfigMixin
import FRsutils.utils.validation_utils as valutil
from FRsutils.core.models.base_fuzzy_rough_model import BaseFuzzyRoughModel

class BaseAllPurposeFuzzyRoughOversampler(LazyBuildableFromConfigMixin, ABC, BaseOverSampler):
    """
    @brief Abstract base class for oversampling using Fuzzy Rough Sets.

    This class builds all fuzzy rough components lazily using configuration parameters.
    It supports multiple models (e.g., ITFRS, OWAFRS, VQRS).

    @warning Do not instantiate or use this class directly. Use a concrete subclass instead.
    """

    def __init__(self, sampling_strategy='auto', **fr_config_kwargs):
        super().__init__(sampling_strategy=sampling_strategy)

        # Store the config for fuzzy rough model (e.g., type='itfrs', tnorm_name='product', ...)
        self._initialize_lazy_config(BaseFuzzyRoughModel, **fr_config_kwargs)

        # Additional strategy validation
        instance_ranking_strategy = fr_config_kwargs.get('instance_ranking_strategy', 'pos')
        self.instance_ranking_strategy = valutil.validate_ranking_strategy_choice(instance_ranking_strategy)

    def _get_target_classes(self):
        """Determine which classes to oversample based on sampling_strategy."""
        if self.instance_ranking_strategy == 'pos':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        elif isinstance(self.instance_ranking_strategy, dict):
            return list(self.instance_ranking_strategy.keys())
        else:
            warnings.warn(f"Unsupported sampling_strategy: {self.instance_ranking_strategy}. Using 'auto'.")
            return [cls for cls in self.classes_ if cls != max(self.target_stats_, key=self.target_stats_.get)]

    def _get_num_samples(self, class_label):
        """Determine number of samples to generate for a class."""
        if self.instance_ranking_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
        elif isinstance(self.instance_ranking_strategy, dict):
            target_count = max(self.target_stats_[class_label], self.instance_ranking_strategy[class_label])
        else:
            warnings.warn(f"Interpreting sampling_strategy '{self.instance_ranking_strategy}' as 'auto'.")
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]

        current_count = self.target_stats_[class_label]
        return max(0, target_count - current_count)

    @abstractmethod
    def _check_params(self):
        """
        Checks correctness of parameters specific to this object.
        Each derived class must implement this.
        """
        raise NotImplementedError("Subclasses must implement _check_params.")

    @abstractmethod
    def fit_resample(self, X, y):
        """
        @brief Fits the resampler to the data and returns the resampled data.

        @param X: Input feature matrix.
        @param y: Target labels.
        @return: Tuple of resampled (X, y)
        """
        raise NotImplementedError("Subclasses must implement fit_resample.")
