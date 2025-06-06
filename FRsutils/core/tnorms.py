
# ✅ Quick Summary of Features
# Feature	Description
# register(*names)	Decorator to register T-norms with multiple aliases
# create(name, **kwargs)	Factory method to instantiate from name/alias
# list_available()	List of all T-norms with aliases
# to_dict() / from_dict()	Serialization & deserialization
# help()	Returns class docstring as readable help
# validate_params()	Ensures required parameters are present and valid


# ✅ Summary Table of Design Principles
# Category	Name	Usage & Where Applied
# Design Pattern	Factory Method	TNorm.create(name, **kwargs) dynamically creates TNorm objects from registered classes based on name/alias.
# Design Pattern	Registry Pattern	TNorm._registry and register() decorator maintain a central registry of all available TNorm subclasses and aliases.
# Design Pattern	Template Method	Abstract base class TNorm defines method signatures like __call__() and reduce(), which are implemented differently by subclasses.
# Design Pattern	Decorator (Class Registration)	@TNorm.register('name', ...) dynamically registers subclasses with aliases using a class decorator.
# Design Pattern	Strategy Pattern	Each TNorm subclass (e.g., MinTNorm, YagerTNorm) implements its own strategy for computing the T-norm.
# Design Pattern	Adapter (Serialization)	to_dict() / from_dict() methods act as an adapter for converting between object and dictionary representation for serialization.
# Architecture	Pluggable Architecture	New T-norms can be added without modifying core logic—just register via decorator, enabling extensibility.
# Clean Code	Single Responsibility Principle	Each class and method does one thing (e.g., YagerTNorm handles only the Yager logic; validate_params handles only validation).
# Clean Code	Open/Closed Principle	Framework is open for extension (add new TNorms) but closed for modification (no need to change base class).
# Clean Code	DRY (Don't Repeat Yourself)	Common logic (e.g. alias filtering, validation pattern) is centralized and reused via helper methods.
# Clean Code	Encapsulation & Abstraction	Abstract base class hides implementation details from users and enforces a consistent API.
# Clean Code	Liskov Substitution Principle	All TNorm subclasses can be used wherever a TNorm object is expected without breaking functionality.
# Clean Code	Docstring Documentation	Doxygen-style docstrings enable IDE and tool-friendly introspection and maintainability.
# Design Support	Reflection/Introspection	inspect.signature() in _filter_args() allows dynamic and safe instantiation of subclasses with correct parameters.


import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Dict, List
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


class TNorm(ABC):
    """
    @brief Abstract base class for all T-norms.
    
    Provides a unified interface and extensible registry mechanism for defining and using various T-norm operators.
    """

    # Maps T-norm names to their classes.
    _registry: Dict[str, Type['TNorm']] = {}  

    # Stores aliases for each T-norm class.
    _aliases: Dict[Type['TNorm'], List[str]] = {}  

    @classmethod
    def register(cls, *names: str):
        """
        @brief Class decorator to register a T-norm with one or more names.

        @param names: One or more names (aliases) for the T-norm.
        @return: Decorator for subclass registration.
        """
        def decorator(subclass: Type['TNorm']):
            if not names:
                raise ValueError("At least one name must be provided for registration.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"TNorm alias '{key}' is already registered.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """
        @brief Lists all registered T-norms and their aliases.

        @return: Dictionary mapping primary name to list of aliases.
        """
        return {names[0]: names for _, names in cls._aliases.items()}

    @classmethod
    def create(cls, name: str, strict: bool = False, **kwargs) -> 'TNorm':
        """
        @brief Factory method to instantiate a T-norm using its name or alias.

        @param name: Name or alias of the T-norm.
        @param strict: If True, raise error on unused kwargs.
        @param kwargs: Parameters for T-norm constructor.
        @return: Instance of a TNorm subclass.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown T-norm type or alias: {name}")
        tnorm_cls = cls._registry[name]
        tnorm_cls.validate_params(**kwargs)

        # Filter arguments to match __init__ signature
        ctor_args = _filter_args(tnorm_cls, kwargs)
        if strict:
            unused = set(kwargs.keys()) - set(ctor_args.keys())
            if unused:
                raise ValueError(f"Unused parameters in strict mode: {unused}")
        return tnorm_cls(**ctor_args)

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief Optional parameter validation hook for subclasses.

        @param kwargs: Parameters to validate.
        """
        pass

    def to_dict(self) -> dict:
        """
        @brief Serialize the T-norm instance to a dictionary.

        @return: A dictionary representation including type and parameters.
        """
        return {
            "type": self.__class__.__name__.replace("TNorm", "").lower(),
            **self._get_params()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TNorm':
        """
        @brief Deserialize a T-norm instance from a dictionary.

        @param data: Dictionary containing T-norm type and parameters.
        @return: Reconstructed TNorm instance.
        """
        data = data.copy()
        name = data.pop("type")
        return cls.create(name, **data)

    def _get_params(self) -> dict:
        """
        @brief Get serializable parameters for reconstruction.

        Override this in subclasses that use constructor parameters.
        @return: Dictionary of parameters.
        """
        return {}

    def help(self) -> str:
        """
        @brief Return the class-level docstring for this T-norm.

        @return: Docstring or fallback message.
        """
        return self.__class__.__doc__.strip() if self.__class__.__doc__ else "No description available."

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        @brief Apply the T-norm to two arrays element-wise.

        @param a: First input array.
        @param b: Second input array.
        @return: Element-wise result of the T-norm.
        """
        pass

    @abstractmethod
    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """
        @brief Reduce an array using the T-norm.

        @param arr: 2D array of shape (n_samples, n_features).
        @return: Reduced array along axis=0.
        """
        pass


@TNorm.register('minimum', 'min')
class MinTNorm(TNorm):
    """
    @brief Minimum T-norm: min(a, b)
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.min(arr, axis=0)


@TNorm.register('product', 'prod', 'algebraic')
class ProductTNorm(TNorm):
    """
    @brief Product T-norm: a * b
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.prod(arr, axis=0)


@TNorm.register('lukasiewicz', 'luk', 'bounded')
class LukasiewiczTNorm(TNorm):
    """
    @brief Łukasiewicz T-norm: max(0, a + b - 1)
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, a + b - 1.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = max(0.0, result + x - 1.0)
        return result


@TNorm.register('yager', 'yg')
class YagerTNorm(TNorm):
    """
    @brief Yager T-norm: 
    1 - min(1, [(1 - a)^p + (1 - b)^p]^(1/p))

    @param p: Exponent parameter that controls the shape (default = 2.0).
    """

    def __init__(self, p: float = 2.0):
        self.p = p

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Compute Yager T-norm element-wise
        return 1.0 - np.minimum(
            1.0, ((1.0 - a) ** self.p + (1.0 - b) ** self.p) ** (1.0 / self.p)
        )

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        # Reduce across axis using Yager logic
        return 1.0 - np.minimum(
            1.0, np.sum((1.0 - arr) ** self.p, axis=0) ** (1.0 / self.p)
        )

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get('p')
        if p is None:
            raise ValueError("Missing required parameter: p")
        if not isinstance(p, (int, float)):
            raise ValueError("Parameter 'p' must be a float or int")
        if p <= 0:
            raise ValueError("Parameter 'p' must be greater than 0")

    def _get_params(self) -> dict:
        return {"p": self.p}
