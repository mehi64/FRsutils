"""
@file base_solo_fuzzy_rough_oversampler.py
@brief Base oversampler that uses only fuzzy rough sets (no generative models).

Extends BaseAllPurposeFuzzyRoughOversampler to provide fit and resampling hooks.

##############################################
# ✅ Summary
# - Uses fuzzy-rough approximations for ranking/selection
# - Compatible with LazyBuildableFromConfigMixin
# - Adds fit(), fit_resample(), and abstract resampling logic
##############################################
"""

import numpy as np
from collections import Counter
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils import check_X_y
from abc import ABC, abstractmethod
import warnings

from FRsutils.core.similarities import calculate_similarity_matrix
from FRsutils.core.preprocess.base_allpurpose_fuzzy_rough_oversampler import BaseAllPurposeFuzzyRoughOversampler
from FRsutils.utils.fuzzy_rough_dataset_validation_utils import compatible_dataset_with_FuzzyRough

class BaseSoloFuzzyRoughOversampler(BaseAllPurposeFuzzyRoughOversampler):
    """
    @brief Oversampler that uses fuzzy rough set theory directly (no VAE/GAN).
    """

    def __init__(self,
                 k_neighbors=5,
                 bias_interpolation=False,
                 random_state=None,
                 **fr_config_kwargs):

        super().__init__(**fr_config_kwargs)
        self.k_neighbors = k_neighbors
        self.bias_interpolation = bias_interpolation
        self.random_state = random_state

    def fit(self, X, y):
        """
        @brief Validates input and builds the fuzzy-rough model lazily.

        @param X: Feature matrix
        @param y: Target labels
        @return: self
        """
        compatible_dataset_with_FuzzyRough(X, y)
        self._check_params()

        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)

        self.ensure_built(X, y)
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        X_resampled, y_resampled = self._fit_resample(X, y)
        return X_resampled, y_resampled

    def transformm(self, X, y=None):
        """
        @brief Applies resampling transformation (for sklearn pipelines).

        @param X: Feature matrix
        @param y: Labels (required)
        @return: Tuple (X_resampled, y_resampled)
        """
        if y is None:
            raise ValueError("y cannot be None when using transform().")
        return self.fit_resample(X, y)

    @abstractmethod
    def _check_params(self):
        """Validates hyperparameters specific to derived class."""
        pass

    @abstractmethod
    def _fit_resample(self, X, y):
        """Actual resampling logic to be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement _fit_resample().")

    @abstractmethod
    def _prepare_minority_samples(self, X, y, class_label):
        """
        Selects minority samples used for interpolation.

        @param X: Feature matrix
        @param y: Target labels
        @param class_label: The label of the minority class
        @return: np.ndarray of selected sample indices
        """
        pass

    @abstractmethod
    def _generate_new_samples(self):
        """
        Generates new synthetic samples based on selected neighbors.
        Must be implemented in subclasses.
        """
        pass
