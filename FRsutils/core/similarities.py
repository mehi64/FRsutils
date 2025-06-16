"""
@file similarities.py
@brief Extensible framework for similarity function computation and similarity matrix generation.

Defines a pluggable architecture for similarity functions with dynamic registration, creation,
serialization, and matrix computation based on selected T-norms.

##############################################
# ✅ Summary of Features
# Feature				Description
# ----------------------------------------------------------------------------------
# register(*names)		Register similarity functions with aliases
# create(name, **kwargs)	Instantiate from registered names
# list_available()		List of all similarity types and aliases
# to_dict() / from_dict()	Serialization and deserialization
# help()				Returns class-level documentation
# validate_params()		Input validation hook
# name				Returns canonical name (without 'SimilarityFunction')
##############################################
"""

from typing import Callable
import numpy as np
from abc import ABC, abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from FRsutils.core.tnorms import TNorm

class Similarity(RegistryFactoryMixin):
    """
    @brief Abstract base class for all similarity functions.

    Provides a unified interface and registry for defining scalar similarity measures.
    """


    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        @brief Compute similarity between two vectors.

        @param x: Feature vector.
        @param y: Feature vector.
        @return: Similarity score.
        """
        diff = x - y
        self._validate_diff(diff)
        
        if diff.ndim == 0:
            return self.compute(np.array([[diff]]))[0, 0]
        elif diff.ndim == 1:
            diff = diff[:, None] - diff[None, :]
        return self.compute(diff)

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief Optional parameter validation for subclass-specific settings.
        """
        pass

   



    def _validate_diff(self, diff: np.ndarray):
        """
        @brief Ensure the input is a 2D NumPy array of pairwise differences.
        """
        if not isinstance(diff, np.ndarray):
            raise TypeError("Input 'diff' must be a NumPy array.")
        if diff.ndim != 2:
            raise ValueError("Expected a 2D pairwise difference matrix.")


    @abstractmethod
    def compute(self, diff: np.ndarray) -> np.ndarray:
        self._validate_diff(diff)
        """
        @brief Compute the similarity given pairwise differences.

        @param diff: Pairwise difference matrix (n, n)
        @return: Similarity values.
        """
        pass





@Similarity.register("linear")
class LinearSimilarity(Similarity):
    """
    @brief Linear similarity function: sim = max(0, 1 - |x - y|) given 0<= x,y <=1.0 
    """
    def __init__(self):
        self.validate_params()
        
    def compute(self, diff: np.ndarray) -> np.ndarray:
        self._validate_diff(diff)
        return np.maximum(0.0, 1.0 - np.abs(diff))
    
    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}


@Similarity.register("gaussian", "gauss")
class GaussianSimilarity(Similarity):
    """
    @brief Gaussian similarity: sim = exp(-(x - y)^2 / (2 * sigma^2))

    @param sigma: Standard deviation for the Gaussian kernel.
    """
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def compute(self, diff: np.ndarray) -> np.ndarray:
        self._validate_diff(diff)
        return np.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))

    def _get_params(self) -> dict:
        return {"sigma": self.sigma}

    @classmethod
    def validate_params(cls, **kwargs):
        sigma = kwargs.get("sigma")
        if sigma is None or not isinstance(sigma, (float, int)) or sigma <= 0:
            raise ValueError("Parameter 'sigma' must be provided and be a positive number.")

#     if ENABLE_PLACEHOLDER_SIMILARITIES:
#         @Similarity.register("cosine")
#         class CosineSimilarity(Similarity):
#             """
#             @brief Cosine similarity: sim = dot(x, y) / (||x|| * ||y||)
#             Assumes diff = x - y pairs prepared row-wise.
#             """
#             def compute(self, diff: np.ndarray) -> np.ndarray:
#                 raise NotImplementedError("Cosine similarity is not pairwise on |x - y|, requires full vectors.")


# @Similarity.register("exponential")
# class ExponentialSimilarity(Similarity):
#     """
#     @brief Exponential similarity: sim = exp(-alpha * |x - y|)

#     @param alpha: Scaling parameter
#     """
#     def __init__(self, alpha: float = 1.0):
#         self.alpha = alpha

#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         self._validate_diff(diff)
#         return np.exp(-self.alpha * np.abs(diff))

#     def _get_params(self) -> dict:
#         return {"alpha": self.alpha}

#     @classmethod
#     def validate_params(cls, **kwargs):
#         alpha = kwargs.get("alpha")
#         if alpha is None or not isinstance(alpha, (float, int)) or alpha <= 0:
#             raise ValueError("Parameter 'alpha' must be a positive number.")


# @Similarity.register("yager")
# class YagerSimilarity(Similarity):
#     """
#     @brief Yager similarity: sim = 1 - (|x - y|^p)^(1/p)

#     @param p: Exponent parameter (must be > 0)
#     """
#     def __init__(self, p: float = 2.0):
#         self.p = p

#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         self._validate_diff(diff)
#         return 1.0 - (np.abs(diff) ** self.p) ** (1.0 / self.p)

#     def _get_params(self) -> dict:
#         return {"p": self.p}

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")
#         if p is None or not isinstance(p, (float, int)) or p <= 0:
#             raise ValueError("Parameter 'p' must be a positive number.")


# @Similarity.register("hamming")
# class HammingSimilarity(Similarity):
#     """
#     @brief Hamming similarity: sim = 1 - |x - y|
#     """
#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         self._validate_diff(diff)
#         return 1.0 - np.abs(diff)


# if ENABLE_PLACEHOLDER_SIMILARITIES:
#     @Similarity.register("dice")
#     class DiceSimilarity(Similarity):
#         """
#         @brief Dice similarity (not strictly pairwise on |x - y|): placeholder only.
#         """
#         def compute(self, diff: np.ndarray) -> np.ndarray:
#             raise NotImplementedError("Dice similarity requires full input vectors, not just diff.")


# if ENABLE_PLACEHOLDER_SIMILARITIES:
#     @Similarity.register("jaccard")
#     class JaccardSimilarity(Similarity):
#         """
#         @brief Jaccard similarity (not strictly pairwise on |x - y|): placeholder only.
#         """
#         def compute(self, diff: np.ndarray) -> np.ndarray:
#             raise NotImplementedError("Jaccard similarity requires full input vectors, not just diff.")


# if ENABLE_PLACEHOLDER_SIMILARITIES:
#     @Similarity.register("tversky")
#     class TverskySimilarity(Similarity):
#         """
#         @brief Tversky similarity (not strictly pairwise on |x - y|): placeholder only.

#         @param alpha: Weight for A \ B
#         @param beta: Weight for B \ A
#         """
#         def __init__(self, alpha: float = 0.5, beta: float = 0.5):
#             self.alpha = alpha
#             self.beta = beta

#         def compute(self, diff: np.ndarray) -> np.ndarray:
#             raise NotImplementedError("Tversky similarity requires full vector inputs.")

#         def _get_params(self) -> dict:
#             return {"alpha": self.alpha, "beta": self.beta}

#         @classmethod
#         def validate_params(cls, **kwargs):
#             alpha = kwargs.get("alpha")
#             beta = kwargs.get("beta")
#             if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
#                 raise ValueError("Tversky alpha and beta must be numeric.")


def calculate_similarity_matrix(
    X: np.ndarray,
    similarity_func: Similarity,
    tnorm: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    @brief Compute a pairwise similarity matrix from input features and similarity function.

    @param X: Normalized input matrix of shape (n_samples, n_features)
    @param similarity_func: Instance of SimilarityFunction subclass
    @param tnorm: Binary T-norm operator (e.g. min, product)
    @return: Similarity matrix (n_samples, n_samples)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array")
    if X.size == 0:
        return np.zeros((0, 0))
    n_samples, n_features = X.shape
    sim_matrix = np.ones((n_samples, n_samples), dtype=np.float64)

    for k in range(n_features):
        col = X[:, k].reshape(-1, 1)
        diff = col - col.T
        sim_k = similarity_func(col , col.T)
        sim_matrix = tnorm(sim_matrix, sim_k)

    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix

def build_similarity_matrix(X: np.ndarray, **kwargs) -> np.ndarray:
    """
    @brief Build a pairwise similarity matrix from input features and a config dictionary.

    @param X: Normalized input matrix of shape (n_samples, n_features)
    @param kwargs: Flattened config including:
        - similarity: name of similarity function (e.g., 'gaussian')
        - similarity_tnorm: name of T-norm to use across features (e.g., 'minimum')
        - parameters for similarity function (e.g., sigma=0.2)
        - parameters for tnorm (e.g., p=2.0 for Yager)
    @return: Pairwise similarity matrix (n x n)
    """
    similarity_type = kwargs.get("similarity", "gaussian")
    similarity_params = {
        k: v for k, v in kwargs.items() if k not in {"similarity", "similarity_tnorm"}
    }

    tnorm_type = kwargs.get("similarity_tnorm", "minimum")
    tnorm_params = {
        k: v for k, v in kwargs.items() if k not in {"similarity", "similarity_tnorm"}
    }

    similarity_func = Similarity.create(similarity_type, **similarity_params)
    tnorm_func = TNorm.create(tnorm_type, **tnorm_params)

    return calculate_similarity_matrix(X, similarity_func, tnorm_func)