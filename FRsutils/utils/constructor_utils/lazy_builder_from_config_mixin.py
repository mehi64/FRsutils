"""
@file lazy_buildable_from_config_mixin.py
@brief General-purpose mixin to lazily build objects from string-based configuration.

This mixin enables:
- Deferring instantiation of pluggable components until needed
- Central storage of configuration parameters
- Generic interface to integrate with any registry-based factory system

##############################################
# ✅ Design Patterns & Clean Code
# - Factory Method: via `.from_config()` interface
# - Registry Pattern: uses Base class registry lookup
# - Separation of Concerns: separates lazy build from core class logic
# - Open/Closed: supports all pluggable models
##############################################
"""

from abc import ABC

class LazyBuildableFromConfigMixin(ABC):
    """
    @brief Mixin for lazily building models from a string-based config.

    This is intended to be inherited by any class that:
    - Knows its registry (e.g., BaseFuzzyRoughModel)
    - Can be initialized via .from_config(...)
    - Wants to defer instantiation until later (e.g., in fit())
    """

    def _initialize_lazy_config(self, model_class_registry, **config_kwargs):
        """
        @brief Stores the registry and config used to instantiate the model later.

        @param model_class_registry: Registry class supporting get_class().
        @param config_kwargs: Configuration dict, must include key 'type'.
        """
        if 'type' not in config_kwargs:
            raise ValueError("config_kwargs must include a 'type' key to identify the model.")

        self._lazy_model_registry = model_class_registry
        self._lazy_model_config = dict(config_kwargs)
        self._lazy_model_type = config_kwargs['type']
        self._is_built = False

    def _build_lazy_model(self, *args, **kwargs):
        """
        @brief Builds the model using the .from_config() of the resolved class.

        @param args: Positional arguments for from_config (e.g., similarity_matrix, labels)
        @param kwargs: Any additional keyword arguments passed to from_config
        """
        model_cls = self._lazy_model_registry.get_class(self._lazy_model_type)
        self._lazy_model = model_cls.from_config(*args, **kwargs, **self._lazy_model_config)
        self._is_built = True

    def ensure_built(self, *args, **kwargs):
        """
        @brief Ensures that the model is built before use.

        @param args: Passed to _build_lazy_model
        @param kwargs: Passed to _build_lazy_model
        """
        if not getattr(self, '_is_built', False):
            self._build_lazy_model(*args, **kwargs)

    @property
    def lazy_model(self):
        """
        @brief Accessor for the built model.
        @return: The model instance created by _build_lazy_model
        @raises RuntimeError: If ensure_built() was not called first
        """
        if not self._is_built:
            raise RuntimeError("Model has not been built yet. Call ensure_built() first.")
        return self._lazy_model
