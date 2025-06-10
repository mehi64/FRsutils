ENABLE_PLACEHOLDER_SIMILARITIES = False

# ✅ Quick Summary of Features
# Feature	Description
# register(*names)	Register similarity functions with multiple aliases
# create(name, **kwargs)	Factory method for dynamic instantiation by alias
# list_available()	List registered similarity functions and aliases
# to_dict() / from_dict()	Support serialization and deserialization
# help()	Return docstring documentation for subclasses
# validate_params()	Hook for parameter checking in subclasses
# __call__()	Smart dispatcher for scalar/vector/matrix input
# compute()	Implemented by subclasses to define similarity logic

# ✅ Summary Table of Design Principles
# Category	Name	Usage & Where Applied
#################################################################################
# Design Pattern	Factory Method	SimilarityFunction.create(name, **kwargs)
# Design Pattern	Registry Pattern	SimilarityFunction._registry and register() decorator
# Design Pattern	Template Method	SimilarityFunction defines abstract compute()
# Design Pattern	Strategy Pattern	Each subclass defines a custom similarity strategy
# Design Pattern	Adapter Pattern	to_dict() / from_dict() serialize/deserialize objects
# Architecture	Pluggable Architecture	Register new functions via decorators
# Clean Code	Single Responsibility	Each class handles a single similarity strategy
# Clean Code	Open/Closed Principle	New classes extend without base modification
# Clean Code	DRY	Common validation and logic abstracted in base
# Clean Code	Liskov Substitution	All subclasses interchangeable via base API
# Clean Code	Doxygen-style Docs	Consistent docstring usage
# Clean Code	Fail-Fast Validation	validate_params() prevents invalid usage
# Clean Code	Type Safety	Explicit param typing in constructors
# Clean Code	Uniform API	All classes support __call__, to_dict, etc.


"""
@file similarities.py
@brief Extensible framework for similarity function computation and similarity matrix generation.

Defines a pluggable architecture for similarity functions with dynamic registration, creation,
serialization, and matrix computation based on selected T-norms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Dict, List, Callable
import inspect


def _filter_args(cls, kwargs: dict) -> dict:
    """
    Filter kwargs to only include parameters accepted by the class constructor.

    @param cls: The class whose constructor will be inspected.
    @param kwargs: Dictionary of keyword arguments.
    @return: Filtered dictionary containing only relevant constructor parameters.
    """
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class Similarity(ABC):
    """
    @brief Abstract base class for all similarity functions.

    Provides a unified interface and registry for defining scalar similarity measures.
    """

    _registry: Dict[str, Type['Similarity']] = {}
    _aliases: Dict[Type['Similarity'], List[str]] = {}

    @classmethod
    def register(cls, *names: str):
        """
        @brief Class decorator to register a similarity function under one or more names.

        @param names: Aliases for the similarity function.
        """
        def decorator(subclass: Type['Similarity']):
            if not names:
                raise ValueError("At least one name must be provided for registration.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"SimilarityFunction alias '{key}' already registered.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, strict: bool = False, **kwargs) -> 'Similarity':
        """
        @brief Factory method to instantiate a similarity function by name or alias.

        @param name: Registered name or alias.
        @param strict: If True, raise error on unused parameters.
        @param kwargs: Arguments for the constructor.
        @return: SimilarityFunction instance.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown similarity function: {name}")
        sim_cls = cls._registry[name]
        sim_cls.validate_params(**kwargs)
        ctor_args = _filter_args(sim_cls, kwargs)
        if strict:
            unused = set(kwargs) - set(ctor_args)
            if unused:
                raise ValueError(f"Unused parameters in strict mode: {unused}")
        return sim_cls(**ctor_args)

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """
        @brief List all registered similarity functions and their aliases.

        @return: Mapping from primary name to all aliases.
        """
        return {names[0]: names for names in cls._aliases.values()}

    @classmethod
    def validate_params(cls, **kwargs):
        """@brief Hook for parameter validation in subclasses."""
        pass

    def to_dict(self) -> dict:
        """
        @brief Serialize the similarity function to a dictionary.

        @return: Dictionary representation including type and parameters.
        """
        return {
            "type": self.__class__.__name__.replace("Similarity", "").lower(),
            **self._get_params()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Similarity':
        """
        @brief Deserialize a similarity function from a dictionary.

        @param data: Dictionary with type and parameters.
        @return: Reconstructed SimilarityFunction instance.
        """
        data = data.copy()
        name = data.pop("type")
        return cls.create(name, **data)

    def help(self) -> str:
        """
        @brief Return class-level documentation string.

        @return: Docstring of the subclass.
        """
        return self.__class__.__doc__.strip() if self.__class__.__doc__ else "No documentation available."

    def _get_params(self) -> dict:
        """Override this in subclasses to support parameter serialization."""
        return {}

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

    def __call__(self, diff: np.ndarray) -> np.ndarray:
        diff = np.asarray(diff)
        if diff.ndim == 0:
            return self.compute(np.array([[diff]]))[0, 0]
        elif diff.ndim == 1:
            diff = diff[:, None] - diff[None, :]
        return self.compute(diff)



    def describe_params_detailed(self) -> dict:
        """
        @brief Returns a dictionary describing the parameters used in this similarity function instance.

        @return: Dictionary mapping parameter names to their type and current value.
        """
        sig = inspect.signature(self.__init__)
        return {
            name: {"type": type(getattr(self, name)).__name__, "value": getattr(self, name)}
            for name in sig.parameters if name != "self" and hasattr(self, name)
        }
    
    @property
    def name(self) -> str:
        """
        @brief Returns the registered name of the tnorm class.

        @return: The class name as a lowercase string without TNorm prefix.
        """
        return self.__class__.__name__.replace("Similarity", "").lower()

@Similarity.register("linear")
class LinearSimilarity(Similarity):
    """
    @brief Linear similarity function: sim = max(0, 1 - |x - y|)
    """
    def compute(self, diff: np.ndarray) -> np.ndarray:
        self._validate_diff(diff)
        return np.maximum(0.0, 1.0 - np.abs(diff))


@Similarity.register("gaussian", "gauss")
class GaussianSimilarity(Similarity):
    """
    @brief Gaussian similarity: sim = exp(-diff^2 / (2 * sigma^2))

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
        sim_k = similarity_func(diff)
        sim_matrix = tnorm(sim_matrix, sim_k)

    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix
