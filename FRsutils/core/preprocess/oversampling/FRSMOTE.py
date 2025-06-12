"""
@file frsmote.py
@brief FRSMOTE: Fuzzy Rough SMOTE-based Oversampler.

This class generates synthetic samples based on fuzzy rough approximations.
Built on top of BaseSoloFuzzyRoughOversampler with LazyBuildableFromConfigMixin.

##############################################
# ✅ Summary
# - Selects minority samples using positive region
# - Applies SMOTE-style interpolation using fuzzy-rough ranked neighbors
# - Configurable bias, neighborhood size, and rough set model
##############################################
"""

import numpy as np
import warnings
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

import FRsutils.utils.math_utils.math_utils as math_utils
from FRsutils.core.preprocess.base_solo_fuzzy_rough_oversampler import BaseSoloFuzzyRoughOversampler


class FRSMOTE(BaseSoloFuzzyRoughOversampler):
    """
    @brief FRSMOTE: SMOTE-like oversampler guided by fuzzy rough approximations.
    """

    def _check_params(self):
        super()._check_params()
        if not isinstance(self.k_neighbors, int) or self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer.")
        if not isinstance(self.bias_interpolation, bool):
            raise ValueError("bias_interpolation must be a boolean.")

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        X_resampled_list = [X.copy()]
        y_resampled_list = [y.copy()]

        for class_label in self._get_target_classes():
            n_samples = self._get_num_samples(class_label)
            if n_samples == 0:
                continue

            p1_candidates, nn_indices, distances, idx_orig = self._prepare_minority_samples(X, y, class_label)
            new_X = self._generate_new_samples(X, n_samples, p1_candidates, idx_orig, nn_indices, random_state)

            if new_X:
                X_resampled_list.append(np.array(new_X))
                y_resampled_list.append(np.full(len(new_X), class_label, dtype=y.dtype))

        return np.vstack(X_resampled_list), np.hstack(y_resampled_list)

    def _prepare_minority_samples(self, X, y, class_label):
        minority_idx = np.where(y == class_label)[0]
        pos_min = self.positive_region[minority_idx]

        if len(minority_idx) <= 1:
            raise ValueError(f"Cannot perform SMOTE for class {class_label} with <= 1 sample.")
        if np.any(pos_min < 0.0):
            raise ValueError("POS values must be non-negative.")

        selectable_mask = pos_min > 0
        selected_idx = minority_idx[selectable_mask]
        pos_sel = pos_min[selectable_mask]
        X_sel = X[selected_idx]

        if len(selected_idx) < 2:
            warnings.warn(f"Only {len(selected_idx)} POS > 0 for class {class_label}. Using all minority if possible.")
            if len(minority_idx) >= 2:
                selected_idx = minority_idx
                pos_sel = pos_min
                X_sel = X[selected_idx]
            else:
                raise ValueError("Insufficient points for neighbor search.")

        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(selected_idx)))
        nn.fit(X_sel)
        distances, indices = nn.kneighbors(X_sel, return_distance=True)

        p1_candidates = list(zip(selected_idx, pos_sel))
        return p1_candidates, indices, distances, selected_idx

    def _generate_new_samples(self, X, n, p1_candidates, idx_sel_orig, nn_indices, rng):
        new_samples = []
        eps = 1e-9

        for _ in range(n):
            p1_idx_orig, _ = math_utils._weighted_random_choice(p1_candidates, rng)
            if p1_idx_orig is None:
                continue

            i_in_sel = np.where(idx_sel_orig == p1_idx_orig)[0]
            if len(i_in_sel) == 0:
                continue
            i_in_sel = i_in_sel[0]

            neighbor_idxs = nn_indices[i_in_sel][1:]
            if len(neighbor_idxs) == 0:
                continue

            j_in_sel = rng.choice(neighbor_idxs)
            p2_idx_orig = idx_sel_orig[j_in_sel]

            p1_pos = self.positive_region[p1_idx_orig]
            p2_pos = self.positive_region[p2_idx_orig]

            if self.bias_interpolation:
                denom = max(p1_pos + p2_pos, eps)
                lam = np.clip(p2_pos / denom, 0.0, 1.0)
            else:
                lam = rng.rand()

            new_sample = X[p1_idx_orig] + lam * (X[p2_idx_orig] - X[p1_idx_orig])
            new_samples.append(new_sample)

        return new_samples

    def supported_strategies(self):
        return {'auto', 'balance_minority'}
 
    def transform(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None when using transform().")
        return self.fit_resample(X, y)

    def get_params(self, deep=True):
        params = {
            'type': self._lazy_model_type,
            'k_neighbors': self.k_neighbors,
            'bias_interpolation': self.bias_interpolation,
            'random_state': self.random_state
        }
        if hasattr(self, '_lazy_model_config'):
            params.update(self._lazy_model_config)
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif hasattr(self, '_lazy_model_config') and k in self._lazy_model_config:
                self._lazy_model_config[k] = v
        return self

    # def get_params(self, deep=True):
    #     """
    #     @brief Returns all parameters including nested fuzzy rough model parameters.

    #     @param deep If True, will return parameters of nested objects.

    #     @return Dictionary of parameter names and values.
    #     """
    #     # Start with known top-level parameters
    #     params = {
    #         'fr_model_name': self.fr_model_type,
    #         'similarity_name': self.similarity_name,
    #         'similarity_tnorm_name': self.similarity_tnorm_name,
    #         'instance_ranking_strategy_name': self.instance_ranking_strategy_name,
    #         'sampling_strategy': self.sampling_strategy,
    #         'k_neighbors': self.k_neighbors,
    #         'bias_interpolation': self.bias_interpolation,
    #         'random_state': self.random_state,
    #         'fr_model_params': self.fr_model_params
    #     }

    #     # Add fuzzy rough model parameters (those passed via **kwargs in init)
    #     if hasattr(self, 'fr_model_params'):
    #         for k, v in self.fr_model_params.items():
    #             params[f'{k}'] = v

    #     return params
    
    # def set_params(self, **params):
    #     """
    #     @brief Sets the parameters including nested fuzzy rough model parameters.

    #     @param params Dictionary of parameters to set.

    #     @return self
    #     """
    #     # Separate top-level and nested fuzzy rough parameters
    #     fr_model_params = self.fr_model_params.copy() if hasattr(self, 'fr_model_params') else {}

    #     for key, value in params.items():
    #         if key.startswith("fr_model_params__"):
    #             inner_key = key[len("fr_model_params__"):]
    #             fr_model_params[inner_key] = value
    #         else:
    #             setattr(self, key, value)

    #     self.fr_model_params = fr_model_params
    #     return self