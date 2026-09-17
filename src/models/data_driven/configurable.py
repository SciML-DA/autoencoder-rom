"""Base class for models that can be saved and restored.

A configurable model is a dataclass whose `__init__` fields are its options. It
reports those options and its trained arrays, and rebuilds itself from them.
`config.model_config` stores both on disk under a hash of the options.

Typical usage example:

  options, arrays = ae.config_options(), ae.trained_arrays()
  restored = AE.from_trained(options, arrays)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, Self, cast

import numpy as np
import numpy.typing as npt

__all__ = ["Configurable", "restore_tuples"]


class Configurable(ABC):
    """A dataclass model that can be rebuilt from its options and trained arrays.

    Subclasses implement `trained_arrays`, `from_trained`, and `from_data`.

    Attributes:
      configurable: Whether `config.model_config` can store the class. Classes
        whose construction needs more than their options set it to `False`.
    """

    configurable: ClassVar[bool] = True
    #: `__init__` fields that `config_options` leaves out.
    _config_exclude: ClassVar[tuple[str, ...]] = ()

    def config_options(self) -> dict[str, Any]:
        """Lists the constructor options as plain Python values.

        Tuples become lists and dtypes become their names.

        Returns:
          Each `__init__` field not in `_config_exclude`, by name.

        Raises:
          ValueError: If an option holds a function.
          TypeError: If an option holds a value with no plain form.
        """
        return {name: _plain(getattr(self, name), name) for name in self.option_names()}

    @classmethod
    def option_names(cls) -> list[str]:
        """Lists the names of the constructor options.

        Returns:
          Each `__init__` field not in `_config_exclude`, in declaration order.

        Raises:
          TypeError: If the class is not a dataclass.
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        return [f.name for f in fields(cls) if f.init and f.name not in cls._config_exclude]

    @classmethod
    def resolve_options(cls, data: Any, **options: Any) -> dict[str, Any]:
        """Computes the options a model built from `data` and `options` reports.

        Args:
          data: The training data.
          **options: Constructor options. Missing options take their defaults.

        Returns:
          The options, as `config_options` reports them.
        """
        del data
        return cls(**options).config_options()

    @abstractmethod
    def trained_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the arrays training produced.

        Returns:
          The arrays, by name.

        Raises:
          RuntimeError: If the model is not trained.
        """

    @classmethod
    @abstractmethod
    def from_trained(cls, options: Mapping[str, Any], arrays: Mapping[str, npt.NDArray[Any]], **overrides: Any) -> Self:
        """Rebuilds a trained model.

        Args:
          options: Options from `config_options`. Options the class no longer
            defines are ignored.
          arrays: Arrays from `trained_arrays`.
          **overrides: Options to use instead of the stored ones.

        Returns:
          The trained model.
        """

    @classmethod
    @abstractmethod
    def from_data(cls, data: Any, **options: Any) -> Self:
        """Builds and trains a model.

        Args:
          data: The training data.
          **options: Constructor options.

        Returns:
          The trained model.
        """

    @classmethod
    def _known_options(cls, options: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Merges stored options and overrides, dropping names the class does not define.

        Args:
          options: Stored options.
          overrides: Options that replace stored ones.

        Returns:
          The merged options.
        """
        known = set(cls.option_names()) | set(overrides)
        return {k: v for k, v in {**options, **overrides}.items() if k in known}


def restore_tuples(value: Any) -> Any:
    """Converts lists, at any depth, to tuples.

    Args:
      value: A value read from a stored configuration.

    Returns:
      `value` with every list replaced by a tuple.
    """
    if isinstance(value, list):
        return tuple(restore_tuples(v) for v in cast(list[Any], value))
    return value


def _plain(value: Any, name: str) -> Any:
    """Converts an option value to plain Python values.

    Args:
      value: The option value.
      name: The option name, for error messages.

    Returns:
      `value` with tuples as lists, NumPy scalars as Python scalars, and dtypes
      as their names.

    Raises:
      ValueError: If `value` holds a function.
      TypeError: If `value` holds a value with no plain form.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, np.dtype):
        return cast(np.dtype[Any], value).name
    if isinstance(value, np.generic):
        return cast(np.generic, value).item()
    if isinstance(value, list | tuple):
        return [_plain(v, name) for v in cast(list[Any] | tuple[Any, ...], value)]
    if isinstance(value, dict):
        return {str(k): _plain(v, name) for k, v in cast(dict[Any, Any], value).items()}
    if callable(value):
        raise ValueError(f"cannot store a function in option {name!r}; pass it by a registered name")
    raise TypeError(f"option {name!r} holds a {type(value).__name__}, which has no plain form")
