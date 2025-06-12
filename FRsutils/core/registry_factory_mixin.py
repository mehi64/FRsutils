"""
@file registry_factory_mixin.py
@brief Provides registration, factory instantiation, and reflection support for pluggable components.

This mixin is designed to be shared among base classes like TNorm, Implicator, and SimilarityFunction
in the FRsutils framework. It encapsulates:
- Registry management via @register
- Factory instantiation with optional strict parameter checking
- Introspection utilities: describe_params_detailed, help
- Serialization support: to_dict / from_dict

##############################################
# ✅ Summary of Clean Code and Design Patterns
# - Registry Pattern: _registry / _aliases with dynamic alias support
# - Factory Method: create(name, **kwargs) with param filtering and instantiation
# - Reflection: Uses inspect to dynamically match __init__ parameters
# - Adapter: to_dict / from_dict for serialization
# - Open/Closed: Easily extendable without modifying the mixin
# - DRY: Removes duplication from base classes like TNorm and Implicator
##############################################
"""

import inspect
from typing import Type, Dict, List, Any

class RegistryFactoryMixin:
    """
    @brief Mixin class for pluggable component registration and instantiation.

    Includes registry management, parameter filtering, dynamic factory creation,
    serialization, and runtime introspection utilities.
    """
    
    def __init_subclass__(cls, **kwargs):
        """
        @brief Automatically initializes per-subclass registry.
        Ensures that each subclass maintains its own `_registry` and `_aliases`.
        This is because tnorms and implicators can have the same name, e.g. yager, luk.
        """
        super().__init_subclass__(**kwargs)
        # a mapping between names and classes. it stores all registered classes
        cls._registry: Dict[str, Type] = {}
        # a mapping between classes and several aliases for each class.
        # for example, Yager could beregistered with 'yager', 'yg', 'yager_implicator', etc.    
        cls._aliases: Dict[Type, List[str]] = {}

    @classmethod
    def register(cls, *names: str):
        """
        @brief Class decorator to register a subclass with one or more aliases.

        Registers the class in the global registry and stores all aliases.

        @param names: One or more alias names for the subclass.
        @return: Class decorator.
        """
        def decorator(subclass: Type):
            if not names:
                raise ValueError("At least one name must be provided for registration.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"Alias '{key}' is already registered in {cls.__name__}.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, strict: bool = False, **kwargs) -> Any:
        """
        @brief Instantiates a subclass by alias name.

        Matches the alias with the registered class, filters the input parameters,
        validates them, and returns an instance.

        @param name: Alias of the subclass.
        @param strict: If True, raises error if unused kwargs are passed.
        @param kwargs: Keyword arguments to pass to the constructor.
        @return: Instantiated subclass object.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown alias: {name}")
        target_cls = cls._registry[name]
        
        target_cls.validate_params(**kwargs)
        
        # ctor_args means constructor arguments
        # filter out unused parameters
        ctor_args = cls._filter_args(target_cls, kwargs)
        if strict:
            unused = set(kwargs) - set(ctor_args)
            if unused:
                raise ValueError(f"Unused parameters: {unused}")
        return target_cls(**ctor_args)

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """
        @brief Lists all registered subclasses and their aliases.

        @return: Dictionary mapping primary alias to all aliases.
        """
        return {names[0]: names for _, names in cls._aliases.items()}

    @staticmethod
    def _filter_args(cls, kwargs: dict) -> dict:
        """
        @brief Filters kwargs to only include those accepted by a class constructor.

        Inspects the constructor signature and removes any extraneous keyword arguments.

        @param cls: Target class.
        @param kwargs: Full dictionary of keyword arguments.
        @return: Filtered dictionary of valid constructor arguments.
        """
        sig = inspect.signature(cls.__init__)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    def get_params_detailed(self) -> dict:
        """
        @brief Returns a dictionary describing the current instance's parameters.

        Uses reflection to enumerate the constructor parameters and their current values.

        @return: Dictionary mapping parameter names to their type and value.
        """
        sig = inspect.signature(self.__init__)
        return {
            name: {"type": type(getattr(self, name)).__name__, "value": getattr(self, name)}
            for name in sig.parameters if name != "self" and hasattr(self, name)
        }

    def _get_params(self):
        return {}

    def to_dict(self) -> dict:
        """
        @brief Serializes the instance to a dictionary.

        @return: Dictionary with "type" and "params" fields.
        """
        return {"type": self.__class__.__name__, "name": self.name, "params": self._get_params()}

    @classmethod
    def from_dict(cls, data: dict) -> Any:
        """
        @brief Deserializes an instance from a dictionary.

        Uses the type key to instantiate the correct registered subclass.

        @param data: Dictionary with "type" and optionally "params".
        @return: Instantiated object.
        """
        return cls.create(data["name"], **data.get("params", {}))

    def help(self) -> str:
        """
        @brief Returns the class-level docstring.

        Useful for introspection and documentation tools.

        @return: String representation of the docstring or fallback text.
        """
        return inspect.getdoc(self.__class__) or "No documentation available."

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief Optional parameter validation hook for subclasses.
        
        @param kwargs: Parameters to validate.
        """
        pass
    
    @property
    def name(self) -> str:
        """
        @brief Returns the registered name of the class (lowercased, with suffix removed).

        Removes suffixes like 'TNorm', 'Implicator', 'SimilarityFunction' from class name.

        @return: Cleaned lowercase name.
        """
        name = self.__class__.__name__
        return name.replace("TNorm", "").replace("Implicator", "").replace("Similarity", "").lower()
    
    
    # def __str__(self) -> str:
    #     return f"{self.__class__.__name__}(n={len(self.labels)})"

    # def __repr__(self) -> str:
    #     return self.__str__()