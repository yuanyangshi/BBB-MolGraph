"""
Generic registry design pattern for components (models, datasets, optimizers).
"""

from typing import Any, Callable, Dict, Optional


class Registry:
    """
    A registry to map string identifiers to classes or constructor functions.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a class or function.
        """
        def decorator(fn_or_cls: Callable[..., Any]) -> Callable[..., Any]:
            key = name if name is not None else fn_or_cls.__name__
            if key in self._registry:
                raise KeyError(f"'{key}' is already registered in registry '{self.name}'.")
            self._registry[key] = fn_or_cls
            return fn_or_cls

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        """
        Retrieve a registered class or callable by name.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in registry '{self.name}'. "
                f"Available keys: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """
        Instantiate an object using the registered class/factory with provided keyword arguments.
        """
        constructor = self.get(name)
        return constructor(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def list_available(self) -> list:
        return list(self._registry.keys())


# Pre-instantiated registries
MODELS = Registry("models")
DATASETS = Registry("datasets")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
