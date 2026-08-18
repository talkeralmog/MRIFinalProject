# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Name-based registries for swappable components.

Models, losses, and masks register themselves with a string name. Configs then
select a component by name, which keeps `train.py` / `engine.py` free of any
hard-coded references to specific architectures or loss functions. Adding a new
model or loss therefore requires only a new class/function plus one decorator.
"""

from __future__ import annotations

from typing import Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
    """A simple string -> object registry with decorator-based registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers the target under ``name``."""

        def decorator(obj: T) -> T:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable:
        """Look up a registered entry by name, with a helpful error message."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "<none>"
            raise KeyError(
                f"unknown {self.kind} '{name}'. Available: {available}"
            )
        return self._entries[name]

    def build(self, name: str, **kwargs):
        """Look up an entry and instantiate/call it with ``kwargs``."""
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._entries)


MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
MASK_REGISTRY = Registry("mask")


def register_model(name: str):
    return MODEL_REGISTRY.register(name)


def register_loss(name: str):
    return LOSS_REGISTRY.register(name)


def register_mask(name: str):
    return MASK_REGISTRY.register(name)
