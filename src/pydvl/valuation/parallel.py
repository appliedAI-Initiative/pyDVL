"""
Parallel processing utilities for pyDVL valuation methods.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from multiprocessing import shared_memory

from joblib._parallel_backends import (
    LokyBackend,
    MultiprocessingBackend,
    ThreadingBackend,
)
from joblib.parallel import _get_active_backend


def ensure_backend_has_generator_return():
    backend = _get_active_backend()[0]
    if not backend.supports_return_generator:
        raise ValueError(
            "The current backend does not support generator return. "
            "Please use a different backend. If you are using ray, follow the "
            "instructions in the documentation to enable generator support."
        )
        # TODO: copy this to the docs:
        #
        #  Instead of calling register_ray(), Do this to enable generator support in
        #  Ray. See:
        #  https://github.com/aai-institute/pyDVL/issues/387#issuecomment-1807994301
        #
        # from ray.util.joblib.ray_backend import RayBackend
        # RayBackend.supports_return_generator = True
        # register_parallel_backend("ray", RayBackend)


class Flag(ABC):
    @abstractmethod
    def set(self): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def __call__(self): ...

    def __bool__(self):  # some syntactic sugar
        return self.__call__()

    @abstractmethod
    def unlink(self): ...


class ThreadingFlag(Flag):
    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def reset(self):
        self._flag = False

    def __call__(self):
        return self._flag

    def unlink(self):
        pass


class MultiprocessingFlag(Flag):
    def __init__(self, name: str):
        self._flag = shared_memory.SharedMemory(name, create=False, size=1)

    def set(self):
        self._flag.buf[0] = True

    def reset(self):
        self._flag.buf[0] = False

    def __call__(self):
        return self._flag.buf[0]

    @classmethod
    def create(cls):
        # limit uid to 30 instead of 32 characters to avoid OSError on MacOS
        uid = uuid.uuid4().hex[:30]
        shared_memory.SharedMemory(uid, create=True, size=1)
        return cls(uid)

    def unlink(self):
        self._flag.close()
        self._flag.unlink()


@contextmanager
def make_parallel_flag():
    """A context manager that creates a flag for signalling across parallel processes.
    The type of flag created is based on the active parallel backend."""
    backend = _get_active_backend()[0]

    if isinstance(backend, MultiprocessingBackend) or isinstance(backend, LokyBackend):
        flag = MultiprocessingFlag.create()
    elif isinstance(backend, ThreadingBackend):
        flag = ThreadingFlag()
    else:
        raise NotImplementedError()

    try:
        yield flag
    finally:
        flag.unlink()
