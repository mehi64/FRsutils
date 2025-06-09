"""
@file fuzzy_rough_lazy_buildable_mixin.py
@brief Provides model registry and factory instantiation for fuzzy rough models.
"""

class FuzzyRoughModelRegistryMixin:
    _registry = {}
    _aliases = {}

    @classmethod
    def register(cls, *names: str):
        def decorator(subclass):
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"Model name '{key}' is already registered.")
                cls._registry[key] = subclass
            cls._aliases[subclass] = list(names)
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, similarity_matrix, labels, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown fuzzy rough model: {name}")
        model_cls = cls._registry[name]
        model_cls.validate_params(**kwargs)
        return model_cls(similarity_matrix, labels, **kwargs)

    @classmethod
    def list_available(cls):
        return {cls.__name__: names for cls, names in cls._aliases.items()}

    @classmethod
    def get_class(cls, name: str):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found.")
        return cls._registry[name]

    @classmethod
    def list_aliases(cls):
        return {alias: klass.__name__ for alias, klass in cls._registry.items()}
