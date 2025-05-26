import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from FRsutils.core.approximations import FuzzyRoughModel_Base


class BaseSoloFuzzyRoughResampler(BaseEstimator, TransformerMixin):
    """Base class with FRS calculations.
       This class of resamplers just use Fuzzy-rough sets without combining with any other model
    """
    def __init__(self,
                 fr_model : FuzzyRoughModel_Base):
        self.fr_model = fr_model

        self.lower_app = self.fr_model.lower_approximation()
        self.upper_app = self.fr_model.upper_approximation()

# TODO: check this. Is that correct?what about all models?
        self.positive_region = self.lower_app
        # self.boundary_region = self.fr_model.boundary_region()


    def _check_params(self, X, y):
        pass

# TODO: check this. Does it need to be changed?
    def fit(self, X, y):
        """Fit the estimator (mainly checks data)."""
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)
        self._check_params(X, y)
        return self

    def fit_resample(self, X, y):
        """Resample the dataset."""
        self.fit(X, y)
        X_resampled, y_resampled = self._fit_resample(X, y)

        return X_resampled, y_resampled

    def _fit_resample(self, X, y):
        """Placeholder for the actual resampling logic in subclasses."""
        raise NotImplementedError("Subclasses must implement _fit_resample.")

    def _prepare_minority_samples(self,
                       minoroty_instances_IDX,
                       X_norm,
                       y):
        "returns nns of each sample of minority class + its rank + rmoves non-promissing instances"
        pass

    def _select_resampling_candidates(self):
        pass

    def _generate_new_samples(self):
        pass
        