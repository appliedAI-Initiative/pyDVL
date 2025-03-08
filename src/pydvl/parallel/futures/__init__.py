from __future__ import annotations

from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Any, Generator

from deprecate import deprecated

from pydvl.parallel.backend import ParallelBackend
from pydvl.parallel.config import ParallelConfig

__all__ = ["init_executor"]


# TODO: delete this function once it's made redundant in v0.10.0
# This string for the benefit of deprecation searches:
# remove_in="0.10.0"
@contextmanager
@deprecated(
    target=None,
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def init_executor(
    max_workers: int | None = None,
    config: ParallelConfig | None = None,
    **kwargs: Any,
) -> Generator[Executor, None, None]:
    """Initializes a futures executor for the given parallel configuration.

    Args:
        max_workers: Maximum number of concurrent tasks.
        config: instance of [ParallelConfig][pydvl.parallel.config.ParallelConfig]
            with cluster address, number of cpus, etc.
        kwargs: Other optional parameter that will be passed to the executor.


    ??? Examples
        ``` python
        from pydvl.parallel.futures import init_executor, ParallelConfig

        config = ParallelConfig(backend="ray")
        with init_executor(max_workers=1, config=config) as executor:
            future = executor.submit(lambda x: x + 1, 1)
            result = future.result()
        assert result == 2
        ```
        ``` python
        from pydvl.parallel.futures import init_executor
        with init_executor() as executor:
            results = list(executor.map(lambda x: x + 1, range(5)))
        assert results == [1, 2, 3, 4, 5]
        ```
    """
    if config is None:
        config = ParallelConfig()
    try:
        cls = ParallelBackend.BACKENDS[config.backend]
        with cls.executor(max_workers=max_workers, config=config, **kwargs) as e:
            yield e
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
