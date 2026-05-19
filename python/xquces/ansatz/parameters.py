from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for axis in shape:
        size *= int(axis)
    return int(size)


@dataclass(frozen=True)
class ParameterBlock:
    """A named contiguous block in a flat variational parameter vector."""

    name: str
    start: int
    stop: int
    shape: tuple[int, ...] | None = None
    kind: str = "generic"
    frozen: bool = False

    def __post_init__(self) -> None:
        start = int(self.start)
        stop = int(self.stop)
        if start < 0:
            raise ValueError("start must be nonnegative")
        if stop < start:
            raise ValueError("stop must be greater than or equal to start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

        shape_value = self.shape
        frozen_value = self.frozen
        if isinstance(shape_value, bool):
            # Backwards compatibility for the old GCRParameterBlock positional
            # constructor: (name, start, stop, frozen).
            frozen_value = bool(shape_value)
            shape_value = None

        if shape_value is None:
            shape = (self.size,)
        else:
            shape = tuple(int(axis) for axis in shape_value)
        if any(axis < 0 for axis in shape):
            raise ValueError("shape entries must be nonnegative")
        if _shape_size(shape) != self.size:
            raise ValueError(
                f"shape {shape} is incompatible with block size {self.size}"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "frozen", bool(frozen_value))

    @property
    def size(self) -> int:
        return self.stop - self.start

    def slice(self) -> slice:
        return slice(self.start, self.stop)

    def with_offset(self, offset: int, *, prefix: str | None = None) -> "ParameterBlock":
        name = self.name if prefix is None else f"{prefix}.{self.name}"
        return ParameterBlock(
            name=name,
            start=self.start + int(offset),
            stop=self.stop + int(offset),
            shape=self.shape,
            kind=self.kind,
            frozen=self.frozen,
        )

    def with_frozen(self, frozen: bool = True) -> "ParameterBlock":
        return ParameterBlock(
            name=self.name,
            start=self.start,
            stop=self.stop,
            shape=self.shape,
            kind=self.kind,
            frozen=frozen,
        )


class ParameterView(Mapping[str, np.ndarray]):
    """Named read/write view over a flat variational parameter vector."""

    def __init__(
        self,
        params: np.ndarray,
        blocks: tuple[ParameterBlock, ...] | list[ParameterBlock],
        *,
        copy: bool = False,
    ) -> None:
        vector = np.asarray(params, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("params must be a one-dimensional vector")
        if copy:
            vector = np.array(vector, copy=True)
        self._params = vector
        self._blocks = tuple(blocks)

        by_name: dict[str, ParameterBlock] = {}
        for block in self._blocks:
            if block.name in by_name:
                raise ValueError(f"duplicate parameter block name {block.name!r}")
            if block.stop > vector.size:
                raise ValueError(
                    f"block {block.name!r} extends past vector length {vector.size}"
                )
            by_name[block.name] = block
        self._by_name = by_name

    @property
    def params(self) -> np.ndarray:
        return self._params

    @property
    def blocks(self) -> tuple[ParameterBlock, ...]:
        return self._blocks

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self._blocks)

    def block(self, name: str) -> ParameterBlock:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown parameter block {name!r}") from exc

    def flat(self, name: str) -> np.ndarray:
        block = self.block(name)
        return self._params[block.slice()]

    def __getitem__(self, name: str) -> np.ndarray:
        block = self.block(name)
        return self._params[block.slice()].reshape(block.shape)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def set(self, name: str, values: np.ndarray) -> None:
        block = self.block(name)
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != block.shape:
            raise ValueError(
                f"block {name!r} expects shape {block.shape}, got {arr.shape}"
            )
        self._params[block.slice()] = arr.reshape(block.size)

    def updated(self, **values: np.ndarray) -> np.ndarray:
        out = np.array(self._params, copy=True)
        view = ParameterView(out, self._blocks)
        for name, value in values.items():
            view.set(name, value)
        return out

    def as_dict(self, *, copy: bool = True) -> dict[str, np.ndarray]:
        out = {}
        for name in self.names:
            value = self[name]
            out[name] = np.array(value, copy=True) if copy else value
        return out


def parameter_view(
    params: np.ndarray,
    blocks: tuple[ParameterBlock, ...] | list[ParameterBlock],
    *,
    copy: bool = False,
) -> ParameterView:
    return ParameterView(params, blocks, copy=copy)
