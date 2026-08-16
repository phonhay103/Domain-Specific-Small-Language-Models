"""Functional programming primitives and utilities for domain SLM pipelines.

Provides pure higher-order functions, immutable data structures, composition,
and currying helpers for constructing clean, testable data workflows.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, NamedTuple, TypeVar

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
T = TypeVar("T")


def pipe(value: T, *functions: Callable[[Any], Any]) -> Any:
    """Pass a value through a sequence of unary functions from left to right.

    Example:
        >>> pipe(5, lambda x: x + 1, lambda x: x * 2)
        12
    """
    return reduce(lambda acc, fn: fn(acc), functions, value)


def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose multiple functions from right to left: compose(f, g)(x) == f(g(x))."""
    return lambda initial: reduce(lambda acc, fn: fn(acc), reversed(functions), initial)


def curry(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Curry a function to allow partial application step by step."""

    def curried(*args: Any, **kwargs: Any) -> Any:
        if len(args) + len(kwargs) >= fn.__code__.co_argcount:
            return fn(*args, **kwargs)
        return partial(curried, *args, **kwargs)

    return curried


def map_tuple(fn: Callable[[A], B], items: Iterable[A]) -> tuple[B, ...]:
    """Pure mapping that returns an immutable tuple."""
    return tuple(map(fn, items))


def filter_tuple(predicate: Callable[[A], bool], items: Iterable[A]) -> tuple[A, ...]:
    """Pure filtering that returns an immutable tuple."""
    return tuple(filter(predicate, items))


def chunk_list(items: Sequence[T], chunk_size: int) -> tuple[tuple[T, ...], ...]:
    """Purely divide a sequence into immutable chunks of specified size."""
    return tuple(tuple(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size))


def format_percentage(numerator: float, denominator: float) -> float:
    """Safely compute and return percentage float value."""
    return (numerator / denominator * 100.0) if denominator != 0.0 else 0.0


def get_model_memory_bytes(model: Any) -> int:
    """Pure calculation of total model memory usage in bytes across all parameters."""
    return sum(param.numel() * param.element_size() for param in model.parameters())


def calculate_speedup(baseline_latency: float, target_latency: float) -> float:
    """Calculate speedup ratio between baseline and optimized target."""
    return (baseline_latency / target_latency) if target_latency > 0.0 else 1.0
