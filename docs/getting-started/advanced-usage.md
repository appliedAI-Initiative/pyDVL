---
title: Advanced usage
alias: 
  name: advanced-usage
  text: Advanced usage
---

# Advanced usage

Besides the dos and don'ts of data valuation itself, which are the subject of
the examples and the documentation of each method, there are two main things to
keep in mind when using pyDVL namely Parallelization and Caching.

## Parallelization { #setting-up-parallelization }

pyDVL uses parallelization to scale and speed up computations. It does so
using one of Dask, Ray or Joblib. The first is used in
the [influence][pydvl.influence] package whereas the other two
are used in the [value][pydvl.value] package.

### Data valuation  { #setting-up-parallelization-data-valuation }

For data valuation, pyDVL uses [joblib](https://joblib.readthedocs.io/en/latest/) for local
parallelization (within one machine) and supports using
[Ray](https://ray.io) for distributed parallelization (across multiple machines).

The former works out of the box but for the latter you will need to install
additional dependencies (see [Extras][installation-extras])
and to provide a running cluster (or run ray in local mode).

!!! info

    As of v0.9.0 pyDVL does not allow requesting resources per task sent to the
    cluster, so you will need to make sure that each worker has enough resources to
    handle the tasks it receives. A data valuation task using game-theoretic methods
    will typically make a copy of the whole model and dataset to each worker, even
    if the re-training only happens on a subset of the data. This means that you
    should make sure that each worker has enough memory to handle the whole dataset.

We use backend classes for both joblib and ray as well as two types
of executors for the different algorithms: the first uses a map reduce pattern as seen in
the [MapReduceJob][pydvl.parallel.map_reduce.MapReduceJob] class
and the second implements the futures executor interface from [concurrent.futures][].

As a convenience, you can also instantiate a parallel backend class
by using the [init_parallel_backend][pydvl.parallel.init_parallel_backend]
function:

```python
from pydvl.parallel import init_parallel_backend
parallel_backend = init_parallel_backend(backend_name="joblib")
```

!!! info

    The executor classes are not meant to be instantiated and used by users
    of pyDVL. They are used internally as part of the computations of the 
    different methods.

!!! danger "Deprecation notice"

    We are currently planning to deprecate
    [MapReduceJob][pydvl.parallel.map_reduce.MapReduceJob] in favour of the
    futures executor interface because it allows for more diverse computation
    patterns with interruptions.

#### Joblib

Please follow the instructions in Joblib's documentation
for all possible configuration options that you can pass to the
[parallel_config][joblib.parallel_config] context manager.

To use the joblib parallel backend with the `loky` backend and verbosity set to `100`
to compute exact shapley values you would use:

```python
import joblib
from pydvl.parallel import JoblibParallelBackend
from pydvl.value.shapley import combinatorial_exact_shapley
from pydvl.utils.utility import Utility

parallel_backend = JoblibParallelBackend() 
u = Utility(...)

with joblib.parallel_config(backend="loky", verbose=100):
    values = combinatorial_exact_shapley(u, parallel_backend=parallel_backend)
```

#### Ray

!!! warning "Additional dependencies"
   
    The Ray parallel backend requires optional dependencies.
    See [Extras][installation-extras] for more information.

Please follow the instructions in Ray's documentation to
[set up a remote cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html).
You could alternatively use a local cluster and in that case you don't have to set
anything up.

Before starting a computation, you should initialize ray by calling 
[`ray.init`][ray.init] with the appropriate parameters:

To set up and start a local ray cluster with 4 CPUs you would use:

```python
import ray

ray.init(num_cpus=4)
```

Whereas for a remote ray cluster you would use:

```python
import ray

address = "<Hypothetical Ray Cluster IP Address>"
ray.init(address)
```

To use the ray parallel backend to compute exact shapley values you would use:

```python
import ray
from pydvl.parallel import RayParallelBackend
from pydvl.value.shapley import combinatorial_exact_shapley
from pydvl.utils.utility import Utility

ray.init()
parallel_backend = RayParallelBackend()
u = Utility(...)
vaues = combinatorial_exact_shapley(u, parallel_backend=parallel_backend)
```

#### Futures executor

For the futures executor interface, we have implemented an executor 
class for ray in [RayExecutor][pydvl.parallel.futures.ray.RayExecutor]
and rely on joblib's loky [get_reusable_executor][loky.get_reusable_executor]
function to instantiate an executor for local parallelization.

They are both compatibles with the builtin
[ThreadPoolExecutor][concurrent.futures.ThreadPoolExecutor]
and [ProcessPoolExecutor][concurrent.futures.ProcessPoolExecutor]
classes.

```pycon
>>> from joblib.externals.loky import _ReusablePoolExecutor
>>> from pydvl.parallel import JoblibParallelBackend
>>> parallel_backend = JoblibParallelBackend() 
>>> with parallel_backend.executor() as executor:
...     results = list(executor.map(lambda x: x + 1, range(3)))
...
>>> results
[1, 2, 3]
```

#### Map-reduce

The map-reduce interface is older and more limited in the patterns
it allows us to use.

To reproduce the previous example using
[MapReduceJob][pydvl.parallel.map_reduce.MapReduceJob], we would use:

```pycon
>>> from pydvl.parallel import JoblibParallelBackend, MapReduceJob
>>> parallel_backend = JoblibParallelBackend() 
>>> map_reduce_job = MapReduceJob(
...     list(range(3)),
...     map_func=lambda x: x[0] + 1,
...     parallel_backend=parallel_backend,
... )
>>> results = map_reduce_job()
>>> results
[1, 2, 3]
```

### Influence functions

Refer to [Scaling influence computation][scaling-influence-computation] for
explanations about parallelization for Influence Functions.

## Caching { #getting-started-cache }

PyDVL can cache (memoize) the computation of the utility function
and speed up some computations for data valuation.
It is however disabled by default.
When it is enabled it takes into account the data indices passed as argument
and the utility function wrapped into the
[Utility][pydvl.utils.utility.Utility] object. This means that
care must be taken when reusing the same utility function with different data,
see the documentation for the [caching package][pydvl.utils.caching] for more
information.

In general, caching won't play a major role in the computation of Shapley values
because the probability of sampling the same subset twice, and hence needing
the same utility function computation, is very low. However, it can be very
useful when comparing methods that use the same utility function, or when
running multiple experiments with the same data.

pyDVL supports 3 different caching backends:

- [InMemoryCacheBackend][pydvl.utils.caching.memory.InMemoryCacheBackend]:
  an in-memory cache backend that uses a dictionary to store and retrieve
  cached values. This is used to share cached values between threads
  in a single process.

- [DiskCacheBackend][pydvl.utils.caching.disk.DiskCacheBackend]:
  a disk-based cache backend that uses pickled values written to and read from disk.  
  This is used to share cached values between processes in a single machine.
- [MemcachedCacheBackend][pydvl.utils.caching.memcached.MemcachedCacheBackend]:
  a [Memcached](https://memcached.org/)-based cache backend that uses pickled values written to
  and read from a Memcached server. This is used to share cached values
  between processes across multiple machines.

    ??? info "Memcached extras"

         The Memcached backend requires optional dependencies.
         See [Extras][installation-extras] for more information.

As an example, here's how one would use the disk-based cached backend
with a utility:

```python
from pydvl.utils.caching.disk import DiskCacheBackend
from pydvl.utils.utility import Utility

cache_backend = DiskCacheBackend()
u = Utility(..., cache_backend=cache_backend)
```

Please refer to the documentation and examples of each backend class for more details.

!!! tip "When is the cache really necessary?"
    Crucially, semi-value computations with the
    [PermutationSampler][pydvl.value.sampler.PermutationSampler] require caching
    to be enabled, or they will take twice as long as the direct implementation
    in [compute_shapley_values][pydvl.value.shapley.compute_shapley_values].

!!! tip "Using the cache"
    Continue reading about the cache in the documentation
    for the [caching package][pydvl.utils.caching].

### Setting up the Memcached cache { #setting-up-memcached }

[Memcached](https://memcached.org/) is an in-memory key-value store accessible
over the network. pyDVL can use it to cache the computation of the utility function
and speed up some computations (in particular, semi-value computations with the
[PermutationSampler][pydvl.value.sampler.PermutationSampler] but other methods
may benefit as well).

You can either install it as a package or run it inside a docker container (the
simplest). For installation instructions, refer to the [Getting
started](https://github.com/memcached/memcached/wiki#getting-started) section in
memcached's wiki. Then you can run it with:

```shell
memcached -u user
```

To run memcached inside a container in daemon mode instead, use:

```shell
docker container run -d --rm -p 11211:11211 memcached:latest
```
