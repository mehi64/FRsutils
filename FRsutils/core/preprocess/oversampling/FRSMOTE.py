import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import FRsutils.core.preprocess.FR_Helpers as frh
import FRsutils.core.preprocess.Base_solo_FR_resampler as bfrrs
from FRsutils.core.approximations import FuzzyRoughModel_Base


# --- FRSMOTE Implementation ---

class FRSMOTE(bfrrs.BaseSoloFuzzyRoughResampler):
    """
    Fuzzy Rough Set based SMOTE (FRSMOTE) Oversampler.
    """
    def __init__(self,
                 fr_model : FuzzyRoughModel_Base,
                 k_neighbors=5,
                 bias_interpolation=False,
                 random_state=None):

        super().__init__(fr_model=fr_model)
        self.k_neighbors = k_neighbors
        self.bias_interpolation = bias_interpolation
        self.random_state = random_state

    def _check_params(self, X, y):
        super()._check_params(X, y)
        if not isinstance(self.k_neighbors, int) or self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer.")
        if not isinstance(self.bias_interpolation, bool):
             raise ValueError("bias_interpolation must be a boolean.")
    
    def _prepare_minority_samples(self,
                       minoroty_instances_IDX,
                       X_norm,
                       y):
        "returns nns of each sample of minority class + its rank + rmoves non-promissing instances"
        pass

    def _fit_resample(self, X_norm, y):
        """Perform FRSMOTE oversampling"""
        random_state = check_random_state(self.random_state)
        X_resampled_list = [X_norm.copy()]
        y_resampled_list = [y.copy()]

        target_classes = self._get_target_classes()
        for class_label in target_classes:
            num_samples_to_generate = self._get_num_samples(class_label)
            if num_samples_to_generate == 0:
                continue

            minority_indices = np.where(y == class_label)[0]
            X_minority_norm = X_norm[minority_indices]
            pos_minority = self.positive_region[minority_indices]

            if len(minority_indices) <= 1:
                 warnings.warn(f"Cannot perform SMOTE for class {class_label} with <= 1 sample.")
                 continue

            # Filter selectable points (POS > 0) - Use original indices
            selectable_mask = pos_minority > 0
            indices_selectable_orig = minority_indices[selectable_mask]
            pos_selectable = pos_minority[selectable_mask]
            # Need normalized data corresponding to selectable points for KNN
            X_selectable_norm = X_norm[indices_selectable_orig]

            if len(indices_selectable_orig) < 2:
                 warnings.warn(f"Only {len(indices_selectable_orig)} points with POS > 0 found for class {class_label}. SMOTE may be unreliable. Consider FRS params. Using all minority points for neighbor search if possible.")
                 if len(minority_indices) >= 2:
                      X_selectable_norm = X_minority_norm
                      indices_selectable_orig = minority_indices
                      pos_selectable = pos_minority
                 else:
                     continue

            # Fit NN on *normalized* selectable minority points
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(indices_selectable_orig)))
            nn.fit(X_selectable_norm)
            # Find neighbors for each selectable point within the *selectable* set
            distances, nn_indices_in_selectable = nn.kneighbors(X_selectable_norm, return_distance=True)

            # Prepare for weighted selection of base point p1 (using original indices)
            p1_candidates = list(zip(indices_selectable_orig, pos_selectable))

            new_samples = []
            epsilon = 1e-9

            for _ in range(num_samples_to_generate):
                # Weighted selection of p1's original index
                p1_idx_orig, _ = frh._weighted_random_choice(p1_candidates, random_state)
                if p1_idx_orig is None: continue

                # Find index of p1 within the selectable set (indices_selectable_orig)
                p1_idx_in_selectable_set = np.where(indices_selectable_orig == p1_idx_orig)[0]
                if len(p1_idx_in_selectable_set) == 0: continue
                p1_idx_in_selectable_set = p1_idx_in_selectable_set[0]

                # Get neighbors of p1 (these indices are relative to the selectable set)
                p1_neighbors_indices_in_selectable = nn_indices_in_selectable[p1_idx_in_selectable_set][1:] # Exclude self
                if len(p1_neighbors_indices_in_selectable) == 0: continue

                # Randomly choose one neighbor p2 (index relative to selectable set)
                p2_idx_in_selectable_set = random_state.choice(p1_neighbors_indices_in_selectable)

                # Get original index of p2
                p2_idx_orig = indices_selectable_orig[p2_idx_in_selectable_set]

                # Get POS memberships for bias calculation
                p1_pos = self.positive_region[p1_idx_orig]
                p2_pos = self.positive_region[p2_idx_orig]

                # Calculate lambda
                if self.bias_interpolation:
                    denominator = max(p1_pos + p2_pos, epsilon) # Ensure denominator is positive
                    lambda_ = p2_pos / denominator
                    lambda_ = np.clip(lambda_, 0.0, 1.0)
                else:
                    lambda_ = random_state.rand()

                # Interpolate in *original* feature space
                p1_orig = X_norm[p1_idx_orig]
                p2_orig = X_norm[p2_idx_orig]
                new_sample = p1_orig + lambda_ * (p2_orig - p1_orig)
                new_samples.append(new_sample)

            if new_samples:
                 X_resampled_list.append(np.array(new_samples))
                 y_resampled_list.append(np.full(len(new_samples), class_label, dtype=y.dtype))

        return np.vstack(X_resampled_list), np.hstack(y_resampled_list)

    # _get_target_classes and _get_num_samples remain the same as in FRSMOTE
    def _get_target_classes(self):
        """Determine which classes to oversample based on sampling_strategy."""
        self.sampling_strategy = 'auto'
        if self.sampling_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        elif isinstance(self.sampling_strategy, dict):
            return list(self.sampling_strategy.keys())
        # Add more strategy handling if needed (float, list, callable)
        else:
            warnings.warn(f"Unsupported sampling_strategy: {self.sampling_strategy}. Using 'auto'.")
            return [cls for cls in self.classes_ if cls != max(self.target_stats_, key=self.target_stats_.get)]


    def _get_num_samples(self, class_label):
        """Determine number of samples to generate for a class."""
        if self.sampling_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
        elif isinstance(self.sampling_strategy, dict):
            # Ensure target count is not less than current count
            target_count = max(self.target_stats_[class_label], self.sampling_strategy[class_label])
        else: # Default to balancing against majority if strategy is unclear
             warnings.warn(f"Interpreting sampling_strategy '{self.sampling_strategy}' as 'auto'.")
             majority_class = max(self.target_stats_, key=self.target_stats_.get)
             target_count = self.target_stats_[majority_class]

        current_count = self.target_stats_[class_label]
        return max(0, target_count - current_count)

