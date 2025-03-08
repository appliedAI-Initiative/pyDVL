from __future__ import annotations

import logging
import warnings
from concurrent.futures import Executor
from typing import Callable, TypeVar, cast

from deprecate import deprecated
from joblib import delayed, effective_n_jobs
from joblib.externals.loky import get_reusable_executor
from joblib.parallel import get_active_backend

from pydvl.parallel.backend import CancellationPolicy, ParallelBackend
from pydvl.parallel.config import ParallelConfig

__all__ = ["JoblibParallelBackend"]

T = TypeVar("T")

logger = logging.getLogger(__name__)


class JoblibParallelBackend(ParallelBackend, backend_name="joblib"):
    """Class used to wrap joblib to make it transparent to algorithms.

    !!! Example
        ``` python
        from pydvl.parallel import JoblibParallelBackend
        parallel_backend = JoblibParallelBackend()
        ```
    """

    _joblib_backend_name: str = "loky"
    """Name of the backend to use for joblib inside [MapReduceJob][pydvl.parallel.map_reduce.MapReduceJob]."""

    @deprecated(
        target=True,
        args_mapping={"config": None},
        deprecated_in="0.9.0",
        remove_in="0.10.0",
    )
    def __init__(self, config: ParallelConfig | None = None) -> None:
        n_jobs: int | None = None
        if config is not None:
            n_jobs = config.n_cpus_local
        self.config = {
            "n_jobs": n_jobs,
        }

    @classmethod
    def executor(
        cls,
        max_workers: int | None = None,
        *,
        config: ParallelConfig | None = None,
        cancel_futures: CancellationPolicy | bool = CancellationPolicy.NONE,
    ) -> Executor:
        """Returns a futures executor for the parallel backend.

        !!! Example
            ``` python
            from pydvl.parallel import JoblibParallelBackend
            parallel_backend = JoblibParallelBackend()
            with parallel_backend.executor() as executor:
                executor.submit(...)
            ```

        Args:
            max_workers: Maximum number of parallel workers.
            config: (**DEPRECATED**) Object configuring parallel computation,
                with cluster address, number of cpus, etc.
            cancel_futures: Policy to use when cancelling futures
                after exiting an Executor.

        Returns:
            Instance of
                [_ReusablePoolExecutor](https://github.com/joblib/loky/blob/master/loky/reusable_executor.py)
        """
        if config is not None:
            warnings.warn(
                "The `JoblibParallelBackend` uses deprecated arguments: "
                "`config`. They were deprecated since v0.9.0 "
                "and will be removed in v0.10.0.",
                FutureWarning,
            )

        if cancel_futures not in (CancellationPolicy.NONE, False):
            warnings.warn(
                "Cancellation of futures is not supported by the joblib backend",
            )
        return cast(Executor, get_reusable_executor(max_workers=max_workers))

    def get(self, v: T, *args, **kwargs) -> T:
        return v

    def put(self, v: T, *args, **kwargs) -> T:
        return v

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a joblib delayed.

        Args:
            fun: the function to wrap

        Returns:
            The delayed function.
        """
        return delayed(fun)  # type: ignore

    def wait(self, v: list[T], *args, **kwargs) -> tuple[list[T], list[T]]:
        return v, []

    def _effective_n_jobs(self, n_jobs: int) -> int:
        eff_n_jobs: int = effective_n_jobs(n_jobs)
        _, backend_n_jobs = get_active_backend()
        if self.config["n_jobs"] is not None:
            maximum_n_jobs = self.config["n_jobs"]
            eff_n_jobs = min(eff_n_jobs, maximum_n_jobs)
        if backend_n_jobs is not None:
            eff_n_jobs = min(eff_n_jobs, backend_n_jobs)
        return eff_n_jobs
